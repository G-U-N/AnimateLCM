##  AnimateLCM SD15


### Enviroment

```
conda create -n animatelcm python=3.9
conda activate animatelcm
pip install -r requirements.txt
```

### Models

1. stable diffusion
```
cd models/StableDiffusion/
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```
2. motion_module
```
cd ..
cd Motion_Module
wget -c https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v.ckpt
wget -c https://huggingface.co/wangfuyun/AnimateLCM-I2V/resolve/main/AnimateLCM_sd15_i2v.ckpt
```

3. spatial_lora
```
cd ..
cd LCM_LoRA
wget -c https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v_lora.safetensors
wget -c https://huggingface.co/wangfuyun/AnimateLCM-I2V/resolve/main/AnimateLCM_sd15_i2v_lora.safetensors
```

4. personalized models 

You can either download from the civitai page or apply this [civitai downloader](https://github.com/ashleykleynhans/civitai-downloader). Then put your downloaded models on the Personalized folder


### Inference

```
python app.py

python app-i2v.py
```

### Batch Inference

```
python batch_inference.py --config=./configs/batch_inference_example.yaml
```