import sys
import torch
try:
    import gradio as gr
except ImportError:
    gr = None
from PIL import Image
import numpy as np
from rembg import remove
from app.utils import change_rgba_bg, rgba_to_rgb
from app.custom_models.utils import load_pipeline
from scripts.all_typing import *
from scripts.utils import session, simple_preprocess

training_config = "app/custom_models/image2mvimage.yaml"
checkpoint_path = "ckpt/img2mvimg/unet_state_dict.pth"
trainer, pipeline = load_pipeline(training_config, checkpoint_path)
if torch.cuda.is_available() and hasattr(pipeline, "enable_model_cpu_offload"):
    try:
        pipeline.enable_model_cpu_offload()
    except Exception as e:
        print(f"Warning: could not enable model CPU offload for mvimg_prediction: {e}")
if hasattr(pipeline, "enable_attention_slicing"):
    try:
        pipeline.enable_attention_slicing()
    except Exception:
        pass

def predict(img_list: List[Image.Image], guidance_scale=2., **kwargs):
    if isinstance(img_list, Image.Image):
        img_list = [img_list]
    img_list = [rgba_to_rgb(i) if i.mode == 'RGBA' else i for i in img_list]
    ret = []
    for img in img_list:
        images = trainer.pipeline_forward(
            pipeline=pipeline,
            image=img,
            guidance_scale=guidance_scale, 
            **kwargs
        ).images
        ret.extend(images)
    return ret


def run_mvprediction(input_image: Image.Image, remove_bg=True, guidance_scale=1.5, seed=1145):
    if input_image.mode == 'RGB' or np.array(input_image)[..., -1].mean() == 255.:
        # still do remove using rembg, since simple_preprocess requires RGBA image
        print("RGB image not RGBA! still remove bg!")
        remove_bg = True

    if remove_bg:
        input_image = remove(input_image, session=session)

    # make front_pil RGBA with white bg
    input_image = change_rgba_bg(input_image, "white")
    single_image = simple_preprocess(input_image)

    generator = torch.Generator(device="cuda").manual_seed(int(seed)) if seed >= 0 else None

    rgb_pils = predict(
        single_image,
        generator=generator,
        guidance_scale=guidance_scale,
        width=256,
        height=256,
        num_inference_steps=30,
    )

    try:
        pipeline.to("cpu")
    except Exception:
        pass
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rgb_pils, single_image
