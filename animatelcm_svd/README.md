## AnimateLCM SVD

### Enviroment 

You can directly using the `environment.yaml`
```
conda env create -f enviroment.yaml
conda activate animatelcm_svd
```
or through the requirements.txt

```
conda create -n animatelcm_svd python=3.9
conda activate animatelcm_svd
pip install -r requirements.txt
```

### Models


You can download the models through `wget`

```
cd safetensors
wget -c https://huggingface.co/wangfuyun/AnimateLCM-SVD-xt/resolve/main/AnimateLCM-SVD-xt-1.1.safetensors
wget -c https://huggingface.co/wangfuyun/AnimateLCM-SVD-xt/resolve/main/AnimateLCM-SVD-xt.safetensors
cd ..
```

### Runing

Simply running 
```
python app.py
```
