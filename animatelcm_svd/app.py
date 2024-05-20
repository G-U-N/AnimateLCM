# import spaces

import gradio as gr
# import gradio.helpers
import torch
import os
from glob import glob
from pathlib import Path
from typing import Optional

from PIL import Image
from diffusers.utils import load_image, export_to_video
from pipeline import StableVideoDiffusionPipeline

import random
from safetensors import safe_open
from animatelcm_scheduler import AnimateLCMSVDStochasticIterativeScheduler


def get_safetensors_files():
    models_dir = "./safetensors"
    safetensors_files = [
        f for f in os.listdir(models_dir) if f.endswith(".safetensors")
    ]
    return safetensors_files


def model_select(selected_file):
    print("load model weights", selected_file)
    pipe.unet.cpu()
    file_path = os.path.join("./safetensors", selected_file)
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    missing, unexpected = pipe.unet.load_state_dict(state_dict, strict=True)
    pipe.unet.cuda()
    del state_dict
    return


noise_scheduler = AnimateLCMSVDStochasticIterativeScheduler(
    num_train_timesteps=40,
    sigma_min=0.002,
    sigma_max=700.0,
    sigma_data=1.0,
    s_noise=1.0,
    rho=7,
    clip_denoised=False,
)
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    scheduler=noise_scheduler,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.to("cuda")
pipe.enable_model_cpu_offload()  # for smaller cost
model_select("AnimateLCM-SVD-xt-1.1.safetensors")
# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True) # for faster inference


max_64_bit_int = 2**63 - 1

# @spaces.GPU
def sample(
    image: Image,
    seed: Optional[int] = 42,
    randomize_seed: bool = False,
    motion_bucket_id: int = 80,
    fps_id: int = 8,
    max_guidance_scale: float = 1.2,
    min_guidance_scale: float = 1,
    width: int = 1024,
    height: int = 576,
    num_inference_steps: int = 4,
    decoding_t: int = 4,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    output_folder: str = "outputs_gradio",
):
    if image.mode == "RGBA":
        image = image.convert("RGB")

    if randomize_seed:
        seed = random.randint(0, max_64_bit_int)
    generator = torch.manual_seed(seed)

    os.makedirs(output_folder, exist_ok=True)
    base_count = len(glob(os.path.join(output_folder, "*.mp4")))
    video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")

    with torch.autocast("cuda"):
        frames = pipe(
            image,
            decode_chunk_size=decoding_t,
            generator=generator,
            motion_bucket_id=motion_bucket_id,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            min_guidance_scale=min_guidance_scale,
            max_guidance_scale=max_guidance_scale,
        ).frames[0]
    export_to_video(frames, video_path, fps=fps_id)
    torch.manual_seed(seed)

    return video_path, seed


def resize_image(image, output_size=(1024, 576)):
    # Calculate aspect ratios
    target_aspect = output_size[0] / output_size[1]  # Aspect ratio of the desired size
    image_aspect = image.width / image.height  # Aspect ratio of the original image

    # Resize then crop if the original image is larger
    if image_aspect > target_aspect:
        # Resize the image to match the target height, maintaining aspect ratio
        new_height = output_size[1]
        new_width = int(new_height * image_aspect)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        # Calculate coordinates for cropping
        left = (new_width - output_size[0]) / 2
        top = 0
        right = (new_width + output_size[0]) / 2
        bottom = output_size[1]
    else:
        # Resize the image to match the target width, maintaining aspect ratio
        new_width = output_size[0]
        new_height = int(new_width / image_aspect)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        # Calculate coordinates for cropping
        left = 0
        top = (new_height - output_size[1]) / 2
        right = output_size[0]
        bottom = (new_height + output_size[1]) / 2

    # Crop the image
    cropped_image = resized_image.crop((left, top, right, bottom))
    return cropped_image


with gr.Blocks() as demo:
    gr.Markdown(
        """
                # [AnimateLCM: Accelerating the Animation of Personalized Diffusion Models and Adapters with Decoupled Consistency Learning](https://arxiv.org/abs/2402.00769)
                Fu-Yun Wang, Zhaoyang Huang (*Corresponding Author), Xiaoyu Shi, Weikang Bian, Guanglu Song, Yu Liu, Hongsheng Li (*Corresponding Author)<br>
                
                [arXiv Report](https://arxiv.org/abs/2402.00769) | [Project Page](https://animatelcm.github.io/) | [Github](https://github.com/G-U-N/AnimateLCM) | [Civitai](https://civitai.com/models/290375/animatelcm-fast-video-generation) | [Replicate](https://replicate.com/camenduru/animate-lcm)
                
                Related Models:
                [AnimateLCM-t2v](https://huggingface.co/wangfuyun/AnimateLCM): Personalized Text-to-Video Generation
                [AnimateLCM-SVD-xt](https://huggingface.co/wangfuyun/AnimateLCM-SVD-xt): General Image-to-Video Generation
                [AnimateLCM-i2v](https://huggingface.co/wangfuyun/AnimateLCM-I2V): Personalized Image-to-Video Generation
                """
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(label="Upload your image", type="pil")
            generate_btn = gr.Button("Generate")
        video = gr.Video()
    with gr.Accordion("Advanced options", open=False):
        safetensors_dropdown = gr.Dropdown(
            label="Choose Safetensors", choices=get_safetensors_files()
        )
        seed = gr.Slider(
            label="Seed",
            value=42,
            randomize=False,
            minimum=0,
            maximum=max_64_bit_int,
            step=1,
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=False)
        motion_bucket_id = gr.Slider(
            label="Motion bucket id",
            info="Controls how much motion to add/remove from the image",
            value=80,
            minimum=1,
            maximum=255,
        )
        fps_id = gr.Slider(
            label="Frames per second",
            info="The length of your video in seconds will be 25/fps",
            value=8,
            minimum=5,
            maximum=30,
        )
        width = gr.Slider(
            label="Width of input image",
            info="It should be divisible by 64",
            value=1024,
            minimum=576,
            maximum=2048,
        )
        height = gr.Slider(
            label="Height of input image",
            info="It should be divisible by 64",
            value=576,
            minimum=320,
            maximum=1152,
        )
        max_guidance_scale = gr.Slider(
            label="Max guidance scale",
            info="classifier-free guidance strength",
            value=1.2,
            minimum=1,
            maximum=2,
        )
        min_guidance_scale = gr.Slider(
            label="Min guidance scale",
            info="classifier-free guidance strength",
            value=1,
            minimum=1,
            maximum=1.5,
        )
        num_inference_steps = gr.Slider(
            label="Num inference steps",
            info="steps for inference",
            value=4,
            minimum=1,
            maximum=20,
            step=1,
        )

    image.upload(fn=resize_image, inputs=image, outputs=image, queue=False)
    generate_btn.click(
        fn=sample,
        inputs=[
            image,
            seed,
            randomize_seed,
            motion_bucket_id,
            fps_id,
            max_guidance_scale,
            min_guidance_scale,
            width,
            height,
            num_inference_steps,
        ],
        outputs=[video, seed],
        api_name="video",
    )
    safetensors_dropdown.change(fn=model_select, inputs=safetensors_dropdown)

    gr.Examples(
        examples=[
            ["test_imgs/ai-generated-8496135_1280.jpg"],
            ["test_imgs/dog-7396912_1280.jpg"],
            ["test_imgs/ship-7833921_1280.jpg"],
            ["test_imgs/girl-4898696_1280.jpg"],
            ["test_imgs/power-station-6579092_1280.jpg"]
        ],
        inputs=[image],
        outputs=[video, seed],
        fn=sample,
        cache_examples=True,
    )

if __name__ == "__main__":
    demo.queue(max_size=20, api_open=False)
    demo.launch(share=True, show_api=False)
