import torch
import os

import numpy as np
from hashlib import md5
def hash_img(img):
    return md5(np.array(img).tobytes()).hexdigest()
def hash_any(obj):
    return md5(str(obj).encode()).hexdigest()

def refine_lr_with_sd(pil_image_list, concept_img_list, control_image_list, prompt_list, pipe=None, strength=0.35, neg_prompt_list="", output_size=(512, 512), controlnet_conditioning_scale=1.):
    if pipe is None:
        return pil_image_list
    with torch.no_grad():
        images = pipe(
            image=pil_image_list,
            ip_adapter_image=concept_img_list,
            prompt=prompt_list,
            neg_prompt=neg_prompt_list,
            num_inference_steps=50,
            strength=strength,
            height=output_size[0],
            width=output_size[1],
            control_image=control_image_list,
            guidance_scale=5.0,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=torch.manual_seed(233),
        ).images
    return images

SR_cache = None

def run_sr_fast(source_pils, scale=4):
    from PIL import Image
    import numpy as np
    global SR_cache
    upsampler = SR_cache

    if upsampler is None:
        try:
            from scripts.upsampler import RealESRGANer
            onnx_path = "ckpt/realesrgan-x4.onnx"
            from pathlib import Path
            root = Path(__file__).resolve().parent.parent
            candidates = [
                Path("ckpt/realesrgan-x4.onnx"),
                root / "ckpt" / "realesrgan-x4.onnx",
                root.parent.parent / "pretrained" / "Unique3D" / "ckpt" / "realesrgan-x4.onnx",
                root.parent.parent / "pretrained" / "Unique3D" / "realesrgan-x4.onnx",
                root.parent.parent / "pretrained" / "realesrgan-x4.onnx",
            ]
            if os.environ.get("PRETRAINED_DIR"):
                p_dir = Path(os.environ["PRETRAINED_DIR"])
                candidates.extend([
                    p_dir / "Unique3D" / "ckpt" / "realesrgan-x4.onnx",
                    p_dir / "Unique3D" / "realesrgan-x4.onnx",
                    p_dir / "realesrgan-x4.onnx",
                ])
            for cand in candidates:
                if cand.is_file():
                    onnx_path = str(cand.resolve())
                    break

            upsampler = RealESRGANer(
                scale=scale,
                onnx_path=onnx_path,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=torch.cuda.is_available(),
                gpu_id=0 if torch.cuda.is_available() else None,
            )
            SR_cache = upsampler
        except Exception:
            upsampler = None

    ret_pils = []
    for idx, img_pils in enumerate(source_pils):
        np_in = isinstance(img_pils, np.ndarray)
        assert isinstance(img_pils, (Image.Image, np.ndarray))
        if upsampler is not None:
            try:
                img = np.array(img_pils)
                output, _ = upsampler.enhance(img, outscale=scale)
                if np_in:
                    ret_pils.append(output)
                else:
                    ret_pils.append(Image.fromarray(output))
                continue
            except Exception:
                pass
        # Fallback to high quality Lanczos upscaling if RealESRGANer is unavailable or fails
        pil_img = Image.fromarray(img_pils) if np_in else img_pils
        w, h = pil_img.size
        upscaled = pil_img.resize((int(w * scale), int(h * scale)), resample=Image.LANCZOS)
        ret_pils.append(np.array(upscaled) if np_in else upscaled)

    return ret_pils
