from transformers import CLIPTextModel, CLIPTokenizer

def load_clip():
    clip_model = CLIPModel.from_pretrained("CLIP-ViT-H-14-laion2B-s32B-b79K")
    clip_processor = CLIPProcessor.from_pretrained("CLIP-ViT-H-14-laion2B-s32B-b79K")
    return clip_model, clip_processor

def get_clip_score(image_pil,text, clip_model, clip_processor):
    inputs = clip_processor(text=text, images=image_pil, return_tensors="pt", padding=True)
    if torch.cuda.is_available():
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    return logits_per_image
