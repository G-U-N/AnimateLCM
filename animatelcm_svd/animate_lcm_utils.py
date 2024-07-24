import torch
import numpy as np
from safetensors import safe_open



def calculate_probabilities(sigma_values, Pmean=0.7, Pstd=1.6):

    log_sigma_values = torch.log(sigma_values)
    
    erf_diff = torch.erf((log_sigma_values[:-1] - Pmean) / (np.sqrt(2) * Pstd)) - \
               torch.erf((log_sigma_values[1:] - Pmean) / (np.sqrt(2) * Pstd))
    
    probabilities = erf_diff / torch.sum(erf_diff)
    
    return probabilities

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


class SVDSolver():
    def __init__(self, N, sigma_min, sigma_max, rho, Pmean, Pstd):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.N = N
        self.Pmean = Pmean
        self.Pstd = Pstd


        self.indices = torch.arange(0, N, dtype=torch.float)
        self.sigmas = (sigma_max ** (1 / rho) + self.indices / (N - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho)))**rho
        
        self.indices = torch.cat([self.indices, torch.tensor([N])])
        self.sigmas  = torch.cat([self.sigmas, torch.tensor([0])])


        self.probs = torch.ones_like(self.sigmas[:-1])*(1/N)

        self.sigmas = self.sigmas[:,None,None,None,None]
        self.timesteps = torch.Tensor([0.25 * (sigma + 1e-44).log() for sigma in self.sigmas])

        self.weights = (1/(self.sigmas[:-1] - self.sigmas[1:]))**0.1 # This is not optimal and can influence the training dynamics a lot. Wish someone can make it better.
        self.c_out = -self.sigmas / ((self.sigmas**2 + 1)**0.5)
        self.c_skip = 1 / (self.sigmas**2 + 1)
        self.c_in = 1 /((self.sigmas**2 + 1) ** 0.5)

    def sample_params(self, indices):

        sampled_sigmas = self.sigmas[indices]
        sampled_timesteps = self.timesteps[indices]
        sampled_weights = self.weights[torch.where(indices>self.weights.shape[0]-1,self.weights.shape[0]-1,indices)]
        sampled_c_out = self.c_out[indices]
        sampled_c_in = self.c_in[indices]
        sampled_c_skip = self.c_skip[indices]

        return indices, sampled_sigmas, sampled_timesteps, sampled_weights, sampled_c_in, sampled_c_out, sampled_c_skip


    def sample_timesteps(self, bsz):
        
        sampled_indices = torch.multinomial(self.probs, bsz, replacement=True)

        sampled_indices, sampled_sigmas, sampled_timesteps, sampled_weights, sampled_c_in, sampled_c_out, sampled_c_skip = self.sample_params(sampled_indices)

        return sampled_indices, sampled_sigmas, sampled_timesteps, sampled_weights, sampled_c_in, sampled_c_out, sampled_c_skip


    def predicted_origin(self, model_output, indices, sample):
        return model_output * self.c_out[indices] + sample * self.c_skip[indices]

    @torch.no_grad()
    def euler_solver(self, model_output, sample, indices, indices_next):
        x = sample
        denoiser = self.predicted_origin(model_output, indices, sample)
        d = (x - denoiser) / self.sigmas[indices]
        sample = x + d * (self.sigmas[indices_next] - self.sigmas[indices])
        
        return sample

    @torch.no_grad()
    def heun_solver(self, model_output, sample, indices, indices_next, model_fn):
        pass

    def to(self,device,dtype):
        self.indinces = self.indices.to(device,dtype)
        self.sigmas = self.sigmas.to(device,dtype)
        self.timesteps=self.timesteps.to(device,dtype)
        self.probs=self.probs.to(device,dtype)
        self.weights=self.weights.to(device,dtype)
        self.c_out=self.c_out.to(device,dtype)
        self.c_skip=self.c_skip.to(device,dtype)
        self.c_in=self.c_in.to(device,dtype)


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


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



