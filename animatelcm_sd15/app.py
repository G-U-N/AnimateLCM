
import os
import json
import torch
import random

import gradio as gr
from glob import glob
from omegaconf import OmegaConf
from datetime import datetime
from safetensors import safe_open

from diffusers import AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

from animatelcm.scheduler.lcm_scheduler import LCMScheduler
from animatelcm.models.unet import UNet3DConditionModel
from animatelcm.pipelines.pipeline_animation import AnimationPipeline
from animatelcm.utils.util import save_videos_grid
from animatelcm.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatelcm.utils.convert_lora_safetensor_to_diffusers import convert_lora
from animatelcm.utils.lcm_utils import convert_lcm_lora
import copy

sample_idx = 0
scheduler_dict = {
    "LCM": LCMScheduler,
}

css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""


class AnimateController:
    def __init__(self):

        # config dirs
        self.basedir = os.getcwd()
        self.stable_diffusion_dir = os.path.join(
            self.basedir, "models", "StableDiffusion")
        self.motion_module_dir = os.path.join(
            self.basedir, "models", "Motion_Module")
        self.personalized_model_dir = os.path.join(
            self.basedir, "models", "Personalized")
        self.savedir = os.path.join(
            self.basedir, "samples", datetime.now().strftime("Gradio-%Y-%m-%dT%H-%M-%S"))
        self.savedir_sample = os.path.join(self.savedir, "sample")
        self.lcm_lora_path = "models/LCM_LoRA/AnimateLCM_sd15_t2v_lora.safetensors"
        os.makedirs(self.savedir, exist_ok=True)

        self.stable_diffusion_list = []
        self.motion_module_list = []
        self.personalized_model_list = []

        self.refresh_stable_diffusion()
        self.refresh_motion_module()
        self.refresh_personalized_model()

        # config models
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.pipeline = None
        self.lora_model_state_dict = {}

        self.inference_config = OmegaConf.load("configs/inference-t2v.yaml")

    def refresh_stable_diffusion(self):
        self.stable_diffusion_list = glob(
            os.path.join(self.stable_diffusion_dir, "*/"))

    def refresh_motion_module(self):
        motion_module_list = glob(os.path.join(
            self.motion_module_dir, "*.ckpt"))
        self.motion_module_list = [
            os.path.basename(p) for p in motion_module_list]

    def refresh_personalized_model(self):
        personalized_model_list = glob(os.path.join(
            self.personalized_model_dir, "*.safetensors"))
        self.personalized_model_list = [
            os.path.basename(p) for p in personalized_model_list]

    def update_stable_diffusion(self, stable_diffusion_dropdown):
        stable_diffusion_dropdown = os.path.join(self.stable_diffusion_dir,stable_diffusion_dropdown)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            stable_diffusion_dropdown, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            stable_diffusion_dropdown, subfolder="text_encoder").cuda()
        self.vae = AutoencoderKL.from_pretrained(
            stable_diffusion_dropdown, subfolder="vae").cuda()
        self.unet = UNet3DConditionModel.from_pretrained_2d(
            stable_diffusion_dropdown, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(self.inference_config.unet_additional_kwargs)).cuda()
        return gr.Dropdown.update()

    def update_motion_module(self, motion_module_dropdown):
        if self.unet is None:
            gr.Info(f"Please select a pretrained model path.")
            return gr.Dropdown.update(value=None)
        else:
            motion_module_dropdown = os.path.join(
                self.motion_module_dir, motion_module_dropdown)
            motion_module_state_dict = torch.load(
                motion_module_dropdown, map_location="cpu")
            missing, unexpected = self.unet.load_state_dict(
                motion_module_state_dict, strict=False)
            assert len(unexpected) == 0
            return gr.Dropdown.update()

    def update_base_model(self, base_model_dropdown):
        if self.unet is None:
            gr.Info(f"Please select a pretrained model path.")
            return gr.Dropdown.update(value=None)
        else:
            base_model_dropdown = os.path.join(
                self.personalized_model_dir, base_model_dropdown)
            base_model_state_dict = {}
            with safe_open(base_model_dropdown, framework="pt", device="cpu") as f:
                for key in f.keys():
                    base_model_state_dict[key] = f.get_tensor(key)

            converted_vae_checkpoint = convert_ldm_vae_checkpoint(
                base_model_state_dict, self.vae.config)
            self.vae.load_state_dict(converted_vae_checkpoint)

            converted_unet_checkpoint = convert_ldm_unet_checkpoint(
                base_model_state_dict, self.unet.config)
            self.unet.load_state_dict(converted_unet_checkpoint, strict=False)

            # self.text_encoder = convert_ldm_clip_checkpoint(base_model_state_dict)
            return gr.Dropdown.update()

    def update_lora_model(self, lora_model_dropdown):
        lora_model_dropdown = os.path.join(
            self.personalized_model_dir, lora_model_dropdown)
        self.lora_model_state_dict = {}
        if lora_model_dropdown == "none":
            pass
        else:
            with safe_open(lora_model_dropdown, framework="pt", device="cpu") as f:
                for key in f.keys():
                    self.lora_model_state_dict[key] = f.get_tensor(key)
        return gr.Dropdown.update()

    def animate(
        self,
        lora_alpha_slider,
        spatial_lora_slider,
        prompt_textbox,
        negative_prompt_textbox,
        sampler_dropdown,
        sample_step_slider,
        width_slider,
        length_slider,
        height_slider,
        cfg_scale_slider,
        seed_textbox
    ):

        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()

        pipeline = AnimationPipeline(
            vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer, unet=self.unet,
            scheduler=scheduler_dict[sampler_dropdown](
                **OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs))
        ).to("cuda")

        if self.lora_model_state_dict != {}:
            pipeline = convert_lora(
                pipeline, self.lora_model_state_dict, alpha=lora_alpha_slider)

        pipeline.unet = convert_lcm_lora(copy.deepcopy(
            self.unet), self.lcm_lora_path, spatial_lora_slider)

        pipeline.to("cuda")

        if seed_textbox != -1 and seed_textbox != "":
            torch.manual_seed(int(seed_textbox))
        else:
            torch.seed()
        seed = torch.initial_seed()

        sample = pipeline(
            prompt_textbox,
            negative_prompt=negative_prompt_textbox,
            num_inference_steps=sample_step_slider,
            guidance_scale=cfg_scale_slider,
            width=width_slider,
            height=height_slider,
            video_length=length_slider,
        ).videos

        save_sample_path = os.path.join(
            self.savedir_sample, f"{sample_idx}.mp4")
        save_videos_grid(sample, save_sample_path)

        sample_config = {
            "prompt": prompt_textbox,
            "n_prompt": negative_prompt_textbox,
            "sampler": sampler_dropdown,
            "num_inference_steps": sample_step_slider,
            "guidance_scale": cfg_scale_slider,
            "width": width_slider,
            "height": height_slider,
            "video_length": length_slider,
            "seed": seed
        }
        json_str = json.dumps(sample_config, indent=4)
        with open(os.path.join(self.savedir, "logs.json"), "a") as f:
            f.write(json_str)
            f.write("\n\n")
        return gr.Video.update(value=save_sample_path)


controller = AnimateController()

controller.update_stable_diffusion("stable-diffusion-v1-5")
controller.update_motion_module("AnimateLCM_sd15_t2v.ckpt")
controller.update_base_model("realistic2.safetensors")


def ui():
    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # [AnimateLCM: Accelerating the Animation of Personalized Diffusion Models and Adapters with Decoupled Consistency Learning](https://arxiv.org/abs/2402.00769)
            Fu-Yun Wang, Zhaoyang Huang (*Corresponding Author), Xiaoyu Shi, Weikang Bian, Guanglu Song, Yu Liu, Hongsheng Li (*Corresponding Author)<br>
            [arXiv Report](https://arxiv.org/abs/2402.00769) | [Project Page](https://animatelcm.github.io/) | [Github](https://github.com/G-U-N/AnimateLCM) | [Civitai](https://civitai.com/models/290375/animatelcm-fast-video-generation) | [Replicate](https://replicate.com/camenduru/animate-lcm)
            """
            
            '''
            Important Notes: 
            1. The generation speed is around 1~2 seconds. There is delay in the space.
            2. Increase the sampling step and cfg if you want more fancy videos.
            '''
        )
        with gr.Column(variant="panel"):
            with gr.Row():

                base_model_dropdown = gr.Dropdown(
                    label="Select base Dreambooth model (required)",
                    choices=controller.personalized_model_list,
                    interactive=True,
                    value="realistic2.safetensors"
                )
                
                motion_module_dropdown = gr.Dropdown(
                    label="Select motion modules",
                    choices=controller.motion_module_list,
                    interactive=True,
                    value="sd15_t2v_beta_motion.ckpt"
                )
                base_model_dropdown.change(fn=controller.update_base_model, inputs=[
                                           base_model_dropdown], outputs=[base_model_dropdown])
                
                motion_module_dropdown.change(fn=controller.update_motion_module, inputs=[motion_module_dropdown],outputs=[motion_module_dropdown])

                lora_model_dropdown = gr.Dropdown(
                    label="Select LoRA model (optional)",
                    choices=["none"],
                    value="none",
                    interactive=True,
                )
                lora_model_dropdown.change(fn=controller.update_lora_model, inputs=[
                                           lora_model_dropdown], outputs=[lora_model_dropdown])

                lora_alpha_slider = gr.Slider(
                    label="LoRA alpha", value=0.8, minimum=0, maximum=2, interactive=True)
                spatial_lora_slider = gr.Slider(
                    label="LCM LoRA alpha", value=0.8, minimum=0.0, maximum=1.0, interactive=True)

                personalized_refresh_button = gr.Button(
                    value="\U0001F503", elem_classes="toolbutton")

                def update_personalized_model():
                    controller.refresh_personalized_model()
                    return [
                        gr.Dropdown.update(
                            choices=controller.personalized_model_list),
                        gr.Dropdown.update(
                            choices=["none"] + controller.personalized_model_list)
                    ]
                personalized_refresh_button.click(fn=update_personalized_model, inputs=[], outputs=[
                                                  base_model_dropdown, lora_model_dropdown])

        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 2. Configs for AnimateLCM.
                """
            )

            prompt_textbox = gr.Textbox(label="Prompt", lines=2, value="a boy holding a rabbit")
            negative_prompt_textbox = gr.Textbox(
                label="Negative prompt", lines=2, value="bad quality")

            with gr.Row().style(equal_height=False):
                with gr.Column():
                    with gr.Row():
                        sampler_dropdown = gr.Dropdown(label="Sampling method", choices=list(
                            scheduler_dict.keys()), value=list(scheduler_dict.keys())[0])
                        sample_step_slider = gr.Slider(
                            label="Sampling steps", value=6, minimum=1, maximum=25, step=1)

                    width_slider = gr.Slider(
                        label="Width",            value=512, minimum=256, maximum=1024, step=64)
                    height_slider = gr.Slider(
                        label="Height",           value=512, minimum=256, maximum=1024, step=64)
                    length_slider = gr.Slider(
                        label="Animation length", value=16,  minimum=12,   maximum=20,   step=1)
                    cfg_scale_slider = gr.Slider(
                        label="CFG Scale",        value=1.5, minimum=1,   maximum=2)

                    with gr.Row():
                        seed_textbox = gr.Textbox(label="Seed", value=-1)
                        seed_button = gr.Button(
                            value="\U0001F3B2", elem_classes="toolbutton")
                        seed_button.click(fn=lambda: gr.Textbox.update(
                            value=random.randint(1, 1e8)), inputs=[], outputs=[seed_textbox])

                    generate_button = gr.Button(
                        value="Generate", variant='primary')

                result_video = gr.Video(
                    label="Generated Animation", interactive=False)
            
            
            generate_button.click(
                fn=controller.animate,
                inputs=[
                    lora_alpha_slider,
                    spatial_lora_slider,
                    prompt_textbox,
                    negative_prompt_textbox,
                    sampler_dropdown,
                    sample_step_slider,
                    width_slider,
                    length_slider,
                    height_slider,
                    cfg_scale_slider,
                    seed_textbox,
                ],
                outputs=[result_video]
            )
            examples = [
                [0.8, 0.8, "a boy is holding a rabbit", "bad quality", "LCM", 8, 512, 16, 512, 1.5, 1234],
                [0.8, 0.8, "1girl smiling", "bad quality", "LCM", 4, 512, 16, 512, 1.5, 1233],
                [0.8, 0.8, "1girl,face,white background,", "bad quality", "LCM", 6, 512, 16, 512, 1.5, 1234],
                [0.8, 0.8, "clouds in the sky, best quality", "bad quality", "LCM", 4, 512, 16, 512, 1.5, 1234],
                
                
            ]
            gr.Examples(
                examples = examples,
                inputs=[
                    lora_alpha_slider,
                    spatial_lora_slider,
                    prompt_textbox,
                    negative_prompt_textbox,
                    sampler_dropdown,
                    sample_step_slider,
                    width_slider,
                    length_slider,
                    height_slider,
                    cfg_scale_slider,
                    seed_textbox,
                ],
                outputs=[result_video],
                fn=controller.animate,
                cache_examples=True,
            )

    return demo


if __name__ == "__main__":
    demo = ui()
    # gr.close_all()
    demo.queue(concurrency_count=3, max_size=20)
    demo.launch(share=True, server_name="127.0.0.1")
