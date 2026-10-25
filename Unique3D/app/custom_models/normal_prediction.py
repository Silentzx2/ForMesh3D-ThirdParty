import sys
import torch
from pathlib import Path
from PIL import Image
from app.utils import rgba_to_rgb, simple_remove
from app.custom_models.utils import load_pipeline
from scripts.utils import rotate_normals_torch
from scripts.all_typing import *

trainer = None
pipeline = None


def get_pipeline():
    global trainer, pipeline
    if trainer is not None and pipeline is not None:
        return trainer, pipeline

    root = Path(__file__).resolve().parent.parent.parent
    training_config = str(root / "app" / "custom_models" / "image2normal.yaml")

    ckpt_candidates = [
        root / "ckpt" / "image2normal" / "unet_state_dict.pth",
        root / "ckpt" / "image2normal" / "unet" / "diffusion_pytorch_model.safetensors",
        root.parent.parent / "pretrained" / "Unique3D" / "ckpt" / "image2normal" / "unet" / "diffusion_pytorch_model.safetensors",
        root.parent.parent / "pretrained" / "Unique3D" / "ckpt" / "image2normal" / "unet_state_dict.pth",
    ]
    checkpoint_path = None
    for cand in ckpt_candidates:
        if cand.exists():
            checkpoint_path = str(cand)
            break
    if checkpoint_path is None:
        checkpoint_path = str(root / "ckpt" / "image2normal" / "unet_state_dict.pth")

    trainer, pipeline = load_pipeline(training_config, checkpoint_path)
    return trainer, pipeline


def predict_normals(image: List[Image.Image], guidance_scale=2., do_rotate=True, num_inference_steps=30, **kwargs):
    tr, pipe = get_pipeline()
    if torch.cuda.is_available():
        if hasattr(pipe, "device") and pipe.device.type != "cuda":
            pipe.to("cuda")
        if hasattr(pipe, "enable_attention_slicing"):
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
        if hasattr(pipe, "enable_vae_slicing"):
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass

    img_list = image if isinstance(image, list) else [image]
    img_list = [rgba_to_rgb(i) if i.mode == 'RGBA' else i for i in img_list]
    images = tr.pipeline_forward(
        pipeline=pipe,
        image=img_list,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale, 
        **kwargs
    ).images
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    images = simple_remove(images)
    if do_rotate and len(images) > 1:
        images = rotate_normals_torch(images, return_types='pil')
    return images