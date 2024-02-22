# AnimateLCM

[AnimateLCM: Accelerating the Animation of Personalized Diffusion Models and Adapters with Decoupled Consistency Learning](https://arxiv.org/abs/2402.00769)



https://github.com/G-U-N/AnimateLCM/assets/60997859/b5e5c928-6cf0-47b8-a2db-4340d49a7c31



Thank you all for your attention. For more details, please refer to our [Project Page](https://animatelcm.github.io/) and [Hugging Face Demo 🤗](https://huggingface.co/spaces/wangfuyun/AnimateLCM).


Use the diffusers to test the beta version AnimateLCM text-to-video models.

```python
import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)
pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="sd15_lora_beta.safetensors", adapter_name="lcm-lora")
pipe.set_adapters(["lcm-lora"], [0.8])

pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

output = pipe(
    prompt="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution",
    negative_prompt="bad quality, worse quality, low resolution",
    num_frames=16,
    guidance_scale=2.0,
    num_inference_steps=6,
    generator=torch.Generator("cpu").manual_seed(0),
)
frames = output.frames[0]
export_to_gif(frames, "animatelcm.gif")

```


🎉 Check the advanced developpment of community: [ComfyUI-AnimateLCM](https://github.com/dezi-ai/ComfyUI-AnimateLCM) and [ComfyUI-Reddit](https://www.reddit.com/r/comfyui/comments/1ajjp9v/animatelcm_support_just_dropped/).

🎉 Awesomw Workflow for AnimateLCM: [Tutorial Video](https://youtu.be/HxlZHsd6xAk).

More code and weights will be released.

## Reference
```bib
@artical{wang2024animatelcm,
      title={AnimateLCM: Accelerating the Animation of Personalized Diffusion Models and Adapters with Decoupled Consistency Learning}, 
      author={Fu-Yun Wang and Zhaoyang Huang and Xiaoyu Shi and Weikang Bian and Guanglu Song and Yu Liu and Hongsheng Li},
      year={2024},
      eprint={2402.00769},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
