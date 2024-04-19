import torch
import numpy as np
from safetensors import safe_open


def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`torch.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


# From LCMScheduler.get_scalings_for_boundary_condition_discrete
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
    return c_skip, c_out


# Compare LCMScheduler.step, Step 4
def predicted_origin(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "v_prediction":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(
            f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0


def scale_for_loss(timesteps, sample, prediction_type, alphas, sigmas):
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        sample = sample * alphas / sigmas
    else:
        raise ValueError(
            f"Prediction type {prediction_type} currently not supported.")

    return sample


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (
            np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        # self.ddim_timesteps = (torch.linspace(100**2,1000**2,30)**0.5).round().numpy().astype(np.int64) - 1
        self.ddim_timesteps_prev = np.asarray(
            [0] + self.ddim_timesteps[:-1].tolist()
        )
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_timesteps_prev = torch.from_numpy(
            self.ddim_timesteps_prev).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(
            self.ddim_alpha_cumprods_prev)

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_timesteps_prev = self.ddim_timesteps_prev.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(
            device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(
            self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev


@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def convert_lcm_lora(unet, path, alpha=1.0):

    if path.endswith(("ckpt",)):
        state_dict = torch.load(path, map_location="cpu")
    else:
        state_dict = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    num_alpha = 0
    for key in state_dict.keys():
        if "alpha" in key:
            num_alpha += 1

    lora_keys = [k for k in state_dict.keys(
    ) if k.endswith("lora_down.weight")]

    updated_state_dict = {}
    for key in lora_keys:
        lora_name = key.split(".")[0]

        if lora_name.startswith("lora_unet_"):
            diffusers_name = key.replace("lora_unet_", "").replace("_", ".")

            if "input.blocks" in diffusers_name:
                diffusers_name = diffusers_name.replace(
                    "input.blocks", "down_blocks")
            else:
                diffusers_name = diffusers_name.replace(
                    "down.blocks", "down_blocks")

            if "middle.block" in diffusers_name:
                diffusers_name = diffusers_name.replace(
                    "middle.block", "mid_block")
            else:
                diffusers_name = diffusers_name.replace(
                    "mid.block", "mid_block")
            if "output.blocks" in diffusers_name:
                diffusers_name = diffusers_name.replace(
                    "output.blocks", "up_blocks")
            else:
                diffusers_name = diffusers_name.replace(
                    "up.blocks", "up_blocks")

            diffusers_name = diffusers_name.replace(
                "transformer.blocks", "transformer_blocks")
            diffusers_name = diffusers_name.replace("to.q.lora", "to_q_lora")
            diffusers_name = diffusers_name.replace("to.k.lora", "to_k_lora")
            diffusers_name = diffusers_name.replace("to.v.lora", "to_v_lora")
            diffusers_name = diffusers_name.replace(
                "to.out.0.lora", "to_out_lora")
            diffusers_name = diffusers_name.replace("proj.in", "proj_in")
            diffusers_name = diffusers_name.replace("proj.out", "proj_out")
            diffusers_name = diffusers_name.replace(
                "time.emb.proj", "time_emb_proj")
            diffusers_name = diffusers_name.replace(
                "conv.shortcut", "conv_shortcut")

            updated_state_dict[diffusers_name] = state_dict[key]
            up_diffusers_name = diffusers_name.replace(".down.", ".up.")
            up_key = key.replace("lora_down.weight", "lora_up.weight")
            updated_state_dict[up_diffusers_name] = state_dict[up_key]

    state_dict = updated_state_dict

    num_lora = 0
    for key in state_dict:
        if "up." in key:
            continue
        up_key = key.replace(".down.", ".up.")
        model_key = key.replace("processor.", "").replace("_lora", "").replace(
            "down.", "").replace("up.", "").replace(".lora", "")
        model_key = model_key.replace("to_out.", "to_out.0.")
        layer_infos = model_key.split(".")[:-1]

        curr_layer = unet
        while len(layer_infos) > 0:
            temp_name = layer_infos.pop(0)
            curr_layer = curr_layer.__getattr__(temp_name)

        weight_down = state_dict[key].to(
            curr_layer.weight.data.device, curr_layer.weight.data.dtype)
        weight_up = state_dict[up_key].to(
            curr_layer.weight.data.device, curr_layer.weight.data.dtype)

        if weight_up.ndim == 2:
            curr_layer.weight.data += 1/8 * alpha * \
                torch.mm(weight_up, weight_down)
        else:
            assert weight_up.ndim == 4
            curr_layer.weight.data += 1/8 * alpha * torch.mm(weight_up.flatten(
                start_dim=1), weight_down.flatten(start_dim=1)).reshape(curr_layer.weight.data.shape)
        num_lora += 1

    return unet
