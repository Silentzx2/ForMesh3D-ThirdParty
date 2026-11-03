import sys
from PIL import Image
from app.utils import rgba_to_rgb, simple_remove
from app.custom_models.utils import load_pipeline
from scripts.utils import rotate_normals_torch
from scripts.all_typing import *

import torch
training_config = "app/custom_models/image2normal.yaml"
checkpoint_path = "ckpt/image2normal/unet_state_dict.pth"
trainer, pipeline = load_pipeline(training_config, checkpoint_path)
if torch.cuda.is_available() and hasattr(pipeline, "enable_model_cpu_offload"):
    try:
        pipeline.enable_model_cpu_offload()
    except Exception as e:
        print(f"Warning: could not enable model CPU offload for normal_prediction: {e}")
if hasattr(pipeline, "enable_attention_slicing"):
    try:
        pipeline.enable_attention_slicing()
    except Exception:
        pass

def predict_normals(image: List[Image.Image], guidance_scale=2., do_rotate=True, num_inference_steps=30, **kwargs):
    if torch.cuda.is_available():
        try:
            pipeline.to("cuda")
        except Exception:
            pass
    img_list = image if isinstance(image, list) else [image]
    img_list = [rgba_to_rgb(i) if i.mode == 'RGBA' else i for i in img_list]
    ret = []
    for img in img_list:
        out = trainer.pipeline_forward(
            pipeline=pipeline,
            image=img,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, 
            **kwargs
        ).images
        ret.extend(out)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    try:
        pipeline.to("cpu")
    except Exception:
        pass
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    images = simple_remove(ret)
    if do_rotate and len(images) > 1:
        images = rotate_normals_torch(images, return_types='pil')
    return images