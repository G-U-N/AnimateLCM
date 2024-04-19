import torch
from safetensors import safe_open
from pipeline import StableVideoDiffusionPipeline
from animatelcm_scheduler import AnimateLCMSVDStochasticIterativeScheduler
from diffusers.utils import load_image, export_to_gif
import os


def run_inference_once(image_path, height, width, inference_time, min_guidance_scale, max_guidance_scale, noise_scheduler, weight = None,):
    if noise_scheduler is not None:
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "/mnt/afs/wangfuyun/SVD-xt/stable-video-diffusion-img2vid-xt",  scheduler = noise_scheduler, torch_dtype=torch.float16, variant="fp16"
        )
    else:
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "/mnt/afs/wangfuyun/SVD-xt/stable-video-diffusion-img2vid-xt",  torch_dtype=torch.float16, variant="fp16"
        )
    pipe.enable_model_cpu_offload()

    if weight is not None:
        state_dict = {}
        with safe_open(weight, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        m,u = pipe.unet.load_state_dict(state_dict,strict=True)
        assert len(u) == 0
        del state_dict

    image = load_image(image_path)
    image = image.resize((height,width))

    generator = torch.manual_seed(42)
    frames = pipe(image, decode_chunk_size=8, generator=generator, num_frames=25, height=height, width=width, num_inference_steps=inference_time, min_guidance_scale = min_guidance_scale, max_guidance_scale = max_guidance_scale).frames[0]
    export_to_gif(frames, f"output_gifs/{image_path[-20:-5]}-{height}-{width}-{inference_time}-{min_guidance_scale}-{max_guidance_scale}-{weight is None}.gif")

if __name__ == "__main__":
    path = "test_imgs"
    weight_path = None 
    assert weight_path is not None
    noise_scheduler = AnimateLCMSVDStochasticIterativeScheduler(
    num_train_timesteps= 40,
    sigma_min = 0.002,
    sigma_max = 700.0,
    sigma_data = 1.0,
    s_noise = 1.0,
    rho = 7,
    clip_denoised = False,
    )
    # noise_scheduler = None
    assert noise_scheduler is not None
    for image_path in os.listdir(path)[5:]:
        image_path = os.path.join(path, image_path)
        for inference_time in [1, 2, 4, 8]:
            for height, width in [(576, 1024)]:
                # for min_scale, max_scale in [(1, 1.5), (1.2, 1.5), (1, 2)]:
                for min_scale, max_scale in [(1.0,1.0)]:
                    run_inference_once(image_path, height, width, inference_time, min_scale, max_scale, noise_scheduler, weight = weight_path)
