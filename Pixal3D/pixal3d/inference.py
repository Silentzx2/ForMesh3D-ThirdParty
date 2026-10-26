# Proxy module to expose inference functions under pixal3d.inference namespace
import sys
from pathlib import Path

_parent_dir = Path(__file__).resolve().parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

try:
    from inference import *
except Exception:
    pass

# Fallbacks in case root inference.py cannot be fully imported
if "distance_from_fov" not in globals():
    import math

    def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
        try:
            import torch
            focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
            f_pixels = focal_length * resolution / 32.0
            return float(f_pixels.item())
        except Exception:
            focal_length = 16.0 / math.tan(camera_angle_x / 2.0)
            return float(focal_length * resolution / 32.0)

    def distance_from_fov(camera_angle_x, grid_point=None, target_point=None, mesh_scale=1.0, image_resolution=512):
        try:
            import torch
            rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
            if grid_point is None:
                grid_point = torch.tensor([-1.0, 0.0, 0.0])
            if target_point is None:
                target_point = torch.tensor([0, image_resolution - 1])
            gp = grid_point.to(torch.float32) @ rotation_matrix.T
            gp = gp / mesh_scale / 2
            xw, yw, zw = gp[0].item(), gp[1].item(), gp[2].item()
            xt, yt = float(target_point[0].item()), float(target_point[1].item())
            f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
            x_ndc = xt - image_resolution / 2.0
            y_ndc = -(yt - image_resolution / 2.0)
            distance_x = f_pixels * xw / x_ndc - yw
            return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}
        except Exception:
            focal_length = 16.0 / math.tan(camera_angle_x / 2.0)
            f_pixels = focal_length * image_resolution / 32.0
            distance_x = 1.0 / (2.0 * math.tan(camera_angle_x / 2.0))
            return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}

if "load_moge_model" not in globals():
    def load_moge_model(device="cuda", model_name="Ruicheng/moge-2-vitl"):
        from moge.model.v2 import MoGeModel
        moge_model = MoGeModel.from_pretrained(model_name)
        moge_model = moge_model.to(device)
        moge_model.eval()
        return moge_model

if "get_camera_params_wild_moge" not in globals():
    def get_camera_params_wild_moge(image_path, moge_model, device="cuda", mesh_scale=1.0, extend_pixel=0, image_resolution=512):
        import math
        import numpy as np
        from PIL import Image
        pil_image = Image.open(image_path).convert("RGB")
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        import torch
        image_tensor = torch.tensor(image_np, dtype=torch.float32, device=device).permute(2, 0, 1)
        output = moge_model.infer(image_tensor)
        fov_x = output["fov_x"]
        if isinstance(fov_x, torch.Tensor):
            camera_angle_x = float(fov_x.item())
        else:
            camera_angle_x = float(fov_x)
        camera_angle_x = math.radians(camera_angle_x)
        grid_point = torch.tensor([-1.0, 0.0, 0.0], device=device)
        target_point = torch.tensor([0, image_resolution - 1], device=device)
        dist_res = distance_from_fov(camera_angle_x, grid_point.cpu(), target_point.cpu(), mesh_scale, image_resolution)
        return {
            "camera_angle_x": camera_angle_x,
            "distance": dist_res["distance_from_x"],
            "mesh_scale": mesh_scale,
        }
