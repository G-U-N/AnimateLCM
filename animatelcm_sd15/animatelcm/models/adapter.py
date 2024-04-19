import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import numpy as np


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


def conv_nd(dims, in_channels, out_channels, kernel_size, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    elif dims == 2:
        return nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
    elif dims == 3:
        if isinstance(kernel_size, int):
            kernel_size = (1, *((kernel_size,) * 2))
        if 'stride' in kwargs.keys():
            if isinstance(kwargs['stride'], int):
                kwargs['stride'] = (1, *((kwargs['stride'],) * 2))
        if 'padding' in kwargs.keys():
            if isinstance(kwargs['padding'], int):
                kwargs['padding'] = (0, *((kwargs['padding'],) * 2))
        return nn.Conv3d(in_channels, out_channels, kernel_size, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def fixed_positional_embedding(t, d_model):
    position = torch.arange(0, t, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float()
                         * (-np.log(10000.0) / d_model))
    pos_embedding = torch.zeros(t, d_model)
    pos_embedding[:, 0::2] = torch.sin(position * div_term)
    pos_embedding[:, 1::2] = torch.cos(position * div_term)
    return pos_embedding


class Adapter(nn.Module):
    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=3, cin=64, ksize=3, sk=False, use_conv=True):
        super(Adapter, self).__init__()
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for i in range(len(channels)):
            for j in range(nums_rb):
                if (i != 0) and (j == 0):
                    self.body.append(ResnetBlock(
                        channels[i-1], channels[i], down=True, ksize=ksize, sk=sk, use_conv=use_conv))
                else:
                    self.body.append(ResnetBlock(
                        channels[i], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv))
        self.body = nn.ModuleList(self.body)
        self.conv_in = zero_module(nn.Conv2d(cin, channels[0], 3, 1, 1))
        self.motion_scale = 0.8
        self.insertion_weights = [1., 1., 1., 1.]

        self.d_model = channels[0]

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        features = []
        x = self.conv_in(x)

        pos_embedding = fixed_positional_embedding(
            t, self.d_model).to(x.device)
        pos_embedding = pos_embedding.unsqueeze(-1).unsqueeze(-1)
        pos_embedding = pos_embedding.expand(-1, -1, h, w)

        x_pos = pos_embedding.repeat(b, 1, 1, 1)

        x = self.motion_scale*x_pos + x

        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i*self.nums_rb + j
                x = self.body[idx](x)
            features.append(x)
        features = [weight*rearrange(fn, '(b t) c h w -> b c t h w', b=b, t=t)
                    for fn, weight in zip(features, self.insertion_weights)]
        return features


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize//2
        if in_c != out_c or sk == False:
            self.in_conv = zero_module(nn.Conv2d(in_c, out_c, ksize, 1, ps))
        else:
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = zero_module(nn.Conv2d(out_c, out_c, ksize, 1, ps))
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)

        if self.in_conv is not None:
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)

        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels

        kernel_size = (2, 2)

        input_height, input_width = x.size(2), x.size(3)

        padding_height = (
            math.ceil(input_height / kernel_size[0]) * kernel_size[0]) - input_height
        padding_width = (
            math.ceil(input_width / kernel_size[1]) * kernel_size[1]) - input_width

        x = F.pad(x, (0, padding_width, 0, padding_height), mode='replicate')

        return self.op(x)
