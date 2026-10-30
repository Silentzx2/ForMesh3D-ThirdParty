import torch
import numpy as np
from PIL import Image
import pymeshlab
import pymeshlab as ml
from pymeshlab import PercentageValue
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from rembg import new_session, remove
import torch
import torch.nn.functional as F
from typing import List, Tuple
from PIL import Image
import trimesh

providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': 8 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'HEURISTIC',
    })
]

session = new_session(providers=providers)

NEG_PROMPT="sketch, sculpture, hand drawing, outline, single color, NSFW, lowres, bad anatomy,bad hands, text, error, missing fingers, yellow sleeves, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry,(worst quality:1.4),(low quality:1.4)"

def load_mesh_with_trimesh(file_name, file_type=None):
    import trimesh
    mesh: trimesh.Trimesh = trimesh.load(file_name, file_type=file_type)
    if isinstance(mesh, trimesh.Scene):
        assert len(mesh.geometry) > 0
        # save to obj first and load again to avoid offset issue
        from io import BytesIO
        with BytesIO() as f:
            mesh.export(f, file_type="obj")
            f.seek(0)
            mesh = trimesh.load(f, file_type="obj")
        if isinstance(mesh, trimesh.Scene):
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in mesh.geometry.values()))
    assert isinstance(mesh, trimesh.Trimesh)

    vertices = torch.from_numpy(mesh.vertices).T
    faces = torch.from_numpy(mesh.faces).T
    colors = None
    if mesh.visual is not None:
        if hasattr(mesh.visual, 'vertex_colors'):
            colors = torch.from_numpy(mesh.visual.vertex_colors)[..., :3].T / 255.
    if colors is None:
        # print("Warning: no vertex color found in mesh! Filling it with gray.")
        colors = torch.ones_like(vertices) * 0.5
    return vertices, faces, colors

def meshlab_mesh_to_py3dmesh(mesh: pymeshlab.Mesh) -> Meshes:
    verts = torch.from_numpy(mesh.vertex_matrix()).float()
    faces = torch.from_numpy(mesh.face_matrix()).long()
    colors = torch.from_numpy(mesh.vertex_color_matrix()[..., :3]).float()
    textures = TexturesVertex(verts_features=[colors])
    return Meshes(verts=[verts], faces=[faces], textures=textures)


def py3dmesh_to_meshlab_mesh(meshes: Meshes) -> pymeshlab.Mesh:
    colors_in = F.pad(meshes.textures.verts_features_packed().cpu().float(), [0,1], value=1).numpy().astype(np.float64)
    m1 = pymeshlab.Mesh(
        vertex_matrix=meshes.verts_packed().cpu().float().numpy().astype(np.float64),
        face_matrix=meshes.faces_packed().cpu().long().numpy().astype(np.int32),
        v_normals_matrix=meshes.verts_normals_packed().cpu().float().numpy().astype(np.float64),
        v_color_matrix=colors_in)
    return m1


def to_pyml_mesh(vertices,faces):
    m1 = pymeshlab.Mesh(
        vertex_matrix=vertices.cpu().float().numpy().astype(np.float64),
        face_matrix=faces.cpu().long().numpy().astype(np.int32),
    )
    return m1


def to_py3d_mesh(vertices, faces, normals=None):
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer.mesh.textures import TexturesVertex
    mesh = Meshes(verts=[vertices], faces=[faces], textures=None)
    if normals is None:
        normals = mesh.verts_normals_packed()
    # set normals as vertext colors
    mesh.textures = TexturesVertex(verts_features=[normals / 2 + 0.5])
    return mesh


def from_py3d_mesh(mesh):
    return mesh.verts_list()[0], mesh.faces_list()[0], mesh.textures.verts_features_packed()

def rotate_normalmap_by_angle(normal_map: np.ndarray, angle: float):
    """
    rotate along y-axis
    normal_map: np.array, shape=(H, W, 3) in [-1, 1]
    angle: float, in degree
    """
    angle = angle / 180 * np.pi
    R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    return np.dot(normal_map.reshape(-1, 3), R.T).reshape(normal_map.shape)

# from view coord to front view world coord
def rotate_normals(normal_pils, return_types='np', rotate_direction=1) -> np.ndarray:  # [0, 255]
    n_views = len(normal_pils)
    ret = []
    for idx, rgba_normal in enumerate(normal_pils):
        # rotate normal
        normal_np = np.array(rgba_normal)[:, :, :3] / 255           # in [-1, 1]
        alpha_np = np.array(rgba_normal)[:, :, 3] / 255             # in [0, 1]
        normal_np = normal_np * 2 - 1
        normal_np = rotate_normalmap_by_angle(normal_np, rotate_direction * idx * (360 / n_views))
        normal_np = (normal_np + 1) / 2
        normal_np = normal_np * alpha_np[..., None]                 # make bg black
        rgba_normal_np = np.concatenate([normal_np * 255, alpha_np[:, :, None] * 255] , axis=-1)
        if return_types == 'np':
            ret.append(rgba_normal_np)
        elif return_types == 'pil':
            ret.append(Image.fromarray(rgba_normal_np.astype(np.uint8)))
        else:
            raise ValueError(f"return_types should be 'np' or 'pil', but got {return_types}")
    return ret


def rotate_normalmap_by_angle_torch(normal_map, angle):
    """
    rotate along y-axis
    normal_map: torch.Tensor, shape=(H, W, 3) in [-1, 1], device='cuda'
    angle: float, in degree
    """
    angle = torch.tensor(angle / 180 * np.pi).to(normal_map)
    R = torch.tensor([[torch.cos(angle), 0, torch.sin(angle)], 
                      [0, 1, 0], 
                      [-torch.sin(angle), 0, torch.cos(angle)]]).to(normal_map)
    return torch.matmul(normal_map.view(-1, 3), R.T).view(normal_map.shape)

def do_rotate(rgba_normal, angle):
    rgba_normal = torch.from_numpy(rgba_normal).float().cuda() / 255
    rotated_normal_tensor = rotate_normalmap_by_angle_torch(rgba_normal[..., :3] * 2 - 1, angle)
    rotated_normal_tensor = (rotated_normal_tensor + 1) / 2
    rotated_normal_tensor = rotated_normal_tensor * rgba_normal[:, :, [3]]    # make bg black
    rgba_normal_np = torch.cat([rotated_normal_tensor * 255, rgba_normal[:, :, [3]] * 255], dim=-1).cpu().numpy()
    return rgba_normal_np

def rotate_normals_torch(normal_pils, return_types='np', rotate_direction=1):
    n_views = len(normal_pils)
    ret = []
    for idx, rgba_normal in enumerate(normal_pils):
        # rotate normal
        angle = rotate_direction * idx * (360 / n_views)
        rgba_normal_np = do_rotate(np.array(rgba_normal), angle)
        if return_types == 'np':
            ret.append(rgba_normal_np)
        elif return_types == 'pil':
            ret.append(Image.fromarray(rgba_normal_np.astype(np.uint8)))
        else:
            raise ValueError(f"return_types should be 'np' or 'pil', but got {return_types}")
    return ret

def change_bkgd(img_pils, new_bkgd=(0., 0., 0.)):
    ret = []
    new_bkgd = np.array(new_bkgd).reshape(1, 1, 3)
    for rgba_img in img_pils:
        img_np = np.array(rgba_img)[:, :, :3] / 255
        alpha_np = np.array(rgba_img)[:, :, 3] / 255
        ori_bkgd = img_np[:1, :1]
        # color = ori_color * alpha + bkgd * (1-alpha)
        # ori_color = (color - bkgd * (1-alpha)) / alpha
        alpha_np_clamp = np.clip(alpha_np, 1e-6, 1) # avoid divide by zero
        ori_img_np = (img_np - ori_bkgd * (1 - alpha_np[..., None])) / alpha_np_clamp[..., None]
        img_np = np.where(alpha_np[..., None] > 0.05, ori_img_np * alpha_np[..., None] + new_bkgd * (1 - alpha_np[..., None]), new_bkgd)
        rgba_img_np = np.concatenate([img_np * 255, alpha_np[..., None] * 255], axis=-1)
        ret.append(Image.fromarray(rgba_img_np.astype(np.uint8)))
    return ret

def change_bkgd_to_normal(normal_pils) -> List[Image.Image]:
    n_views = len(normal_pils)
    ret = []
    for idx, rgba_normal in enumerate(normal_pils):
        # calcuate background normal
        target_bkgd = rotate_normalmap_by_angle(np.array([[[0., 0., 1.]]]), idx * (360 / n_views))
        normal_np = np.array(rgba_normal)[:, :, :3] / 255           # in [-1, 1]
        alpha_np = np.array(rgba_normal)[:, :, 3] / 255             # in [0, 1]
        normal_np = normal_np * 2 - 1
        old_bkgd = normal_np[:1,:1]
        normal_np[alpha_np > 0.05] = (normal_np[alpha_np > 0.05] - old_bkgd * (1 - alpha_np[alpha_np > 0.05][..., None])) / alpha_np[alpha_np > 0.05][..., None]
        normal_np = normal_np * alpha_np[..., None] + target_bkgd * (1 - alpha_np[..., None])
        normal_np = (normal_np + 1) / 2
        rgba_normal_np = np.concatenate([normal_np * 255, alpha_np[..., None] * 255] , axis=-1)
        ret.append(Image.fromarray(rgba_normal_np.astype(np.uint8)))
    return ret


def fix_vert_color_glb(mesh_path):
    from pygltflib import GLTF2, Material, PbrMetallicRoughness
    obj1 = GLTF2().load(mesh_path)
    obj1.meshes[0].primitives[0].material = 0
    obj1.materials.append(Material(
        pbrMetallicRoughness = PbrMetallicRoughness(
            baseColorFactor = [1.0, 1.0, 1.0, 1.0],
            metallicFactor = 0.05,
            roughnessFactor = 0.40,
        ),
        emissiveFactor = [0.0, 0.0, 0.0],
        doubleSided = True,
    ))
    obj1.save(mesh_path)


def srgb_to_linear(c_srgb):
    c_linear = np.where(c_srgb <= 0.04045, c_srgb / 12.92, ((c_srgb + 0.055) / 1.055) ** 2.4)
    return c_linear.clip(0, 1.)


def save_py3dmesh_with_trimesh_fast(meshes: Meshes, save_glb_path, apply_sRGB_to_LinearRGB=True, include_vertex_color=False):
    # convert from pytorch3d meshes to trimesh mesh
    vertices = meshes.verts_packed().cpu().float().numpy()
    triangles = meshes.faces_packed().cpu().long().numpy()
    normals = None
    try:
        norm_t = meshes.verts_normals_packed()
        if norm_t is not None and norm_t.numel() > 0:
            norm_np = norm_t.cpu().float().numpy()
            if norm_np.shape[0] == vertices.shape[0]:
                normals = norm_np
    except Exception:
        normals = None

    if save_glb_path.endswith(".glb"):
        # rotate 180 along +Y
        vertices[:, [0, 2]] = -vertices[:, [0, 2]]
        if normals is not None:
            normals[:, [0, 2]] = -normals[:, [0, 2]]

    t_kwargs = {"process": False}
    if normals is not None and normals.shape[0] == vertices.shape[0]:
        t_kwargs["vertex_normals"] = normals

    np_color = None
    if include_vertex_color and getattr(meshes, "textures", None) is not None:
        try:
            np_color = meshes.textures.verts_features_packed().cpu().float().numpy()
            if apply_sRGB_to_LinearRGB and np_color is not None:
                np_color = srgb_to_linear(np_color)
            if np_color is not None and np_color.shape[0] == vertices.shape[0] and np_color.shape[1] == 3:
                t_kwargs["vertex_colors"] = np_color
        except Exception:
            np_color = None

    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, **t_kwargs)
    mesh.remove_unreferenced_vertices()

    # Automatically purge floating disconnected debris and shards
    try:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            total_f = len(mesh.faces)
            significant = [c for c in components if len(c.faces) >= max(30, int(total_f * 0.05))]
            if significant:
                mesh = trimesh.util.concatenate(significant)
            else:
                mesh = max(components, key=lambda c: len(c.faces))
    except Exception:
        pass

    # save mesh
    mesh.export(save_glb_path)
    if save_glb_path.endswith(".glb") and np_color is not None:
        try:
            fix_vert_color_glb(save_glb_path)
        except Exception:
            pass
    print(f"saving to {save_glb_path}")


def save_glb_and_video(save_mesh_prefix: str, meshes: Meshes, with_timestamp=True, dist=3.5, azim_offset=180, resolution=512, fov_in_degrees=1 / 1.15, cam_type="ortho", view_padding=60, export_video=True, include_vertex_color=False) -> Tuple[str, str]:
    import time
    if '.' in save_mesh_prefix:
        save_mesh_prefix = ".".join(save_mesh_prefix.split('.')[:-1])
    if with_timestamp:
        save_mesh_prefix = save_mesh_prefix + f"_{int(time.time())}"
    ret_mesh = save_mesh_prefix + ".glb"
    # optimized version
    save_py3dmesh_with_trimesh_fast(meshes, ret_mesh, include_vertex_color=include_vertex_color)
    return ret_mesh, None


def simple_clean_mesh(pyml_mesh: ml.Mesh, apply_smooth=True, stepsmoothnum=1, apply_sub_divide=False, sub_divide_threshold=0.25):
    ms = ml.MeshSet()
    ms.add_mesh(pyml_mesh, "cube_mesh")
    
    # 1. Non-manifold repair and degenerate geometry cleanup
    try:
        ms.apply_filter("meshing_remove_duplicate_faces")
        ms.apply_filter("meshing_remove_null_faces")
        ms.apply_filter("meshing_repair_non_manifold_vertices")
        ms.apply_filter("meshing_repair_non_manifold_edges", method='Remove Faces')
    except Exception:
        pass

    # 2. Prune small disconnected floating fragments
    try:
        ms.apply_filter("meshing_remove_connected_component_by_diameter", mincomponentdiag=PercentageValue(10))
    except Exception:
        pass

    # 3. Controlled Laplacian smoothing to remove high-frequency gradient noise / spikes
    if apply_smooth and stepsmoothnum > 0:
        try:
            ms.apply_filter("apply_coord_laplacian_smoothing", stepsmoothnum=stepsmoothnum, cotangentweight=True)
        except Exception:
            pass

    if apply_sub_divide:
        try:
            ms.apply_filter("meshing_surface_subdivision_loop", iterations=1, threshold=PercentageValue(sub_divide_threshold))
        except Exception:
            pass

    try:
        ms.apply_filter("meshing_remove_unreferenced_vertices")
    except Exception:
        pass

    return meshlab_mesh_to_py3dmesh(ms.current_mesh())


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def simple_preprocess(input_image, rembg_session=session, background_color=255):
    RES = 2048
    input_image.thumbnail([RES, RES], Image.Resampling.LANCZOS)
    if input_image.mode != 'RGBA' or np.asarray(input_image)[:, :, -1].mean() > 245.0:
        image_rem = input_image.convert('RGBA')
        try:
            input_image = remove(image_rem, alpha_matting=False, session=rembg_session)
        except Exception:
            input_image = image_rem

    arr = np.asarray(input_image)
    alpha = np.asarray(input_image)[:, :, -1]
    x_nonzero = np.nonzero((alpha > 60).sum(axis=1))
    y_nonzero = np.nonzero((alpha > 60).sum(axis=0))
    if len(x_nonzero[0]) > 0 and len(y_nonzero[0]) > 0:
        x_min = int(x_nonzero[0].min())
        y_min = int(y_nonzero[0].min())
        x_max = int(x_nonzero[0].max())
        y_max = int(y_nonzero[0].max())
        arr = arr[x_min: x_max, y_min: y_max]
        input_image = Image.fromarray(arr)
    # Add a 10% breathing margin around the subject before expanding to square.
    # This prevents the subject's top/bottom/sides from touching the canvas border,
    # which otherwise creates severe boundary distortion walls in orthographic 3D reconstruction.
    w, h = input_image.size
    max_side = max(w, h)
    margin_px = max(int(max_side * 0.10), 16)
    padded_w = w + margin_px * 2
    padded_h = h + margin_px * 2
    padded_img = Image.new("RGBA", (padded_w, padded_h), (background_color, background_color, background_color, 0))
    padded_img.paste(input_image, (margin_px, margin_px))
    input_image = expand2square(padded_img, (background_color, background_color, background_color, 0))
    return input_image

def init_target(img_pils, new_bkgd=(0., 0., 0.), device="cuda"):
    # Convert the background color to a PyTorch tensor
    new_bkgd = torch.tensor(new_bkgd, dtype=torch.float32).view(1, 1, 3).to(device)
    
    # Convert all images to PyTorch tensors and process them
    imgs = torch.stack([torch.from_numpy(np.array(img, dtype=np.float32)) for img in img_pils]).to(device) / 255
    img_nps = imgs[..., :3]
    alpha_nps = imgs[..., 3]
    ori_bkgds = img_nps[:, :1, :1]
    
    # Avoid divide by zero and calculate the original image
    alpha_nps_clamp = torch.clamp(alpha_nps, 1e-6, 1)
    ori_img_nps = (img_nps - ori_bkgds * (1 - alpha_nps.unsqueeze(-1))) / alpha_nps_clamp.unsqueeze(-1)
    ori_img_nps = torch.clamp(ori_img_nps, 0, 1)
    img_nps = torch.where(alpha_nps.unsqueeze(-1) > 0.05, ori_img_nps * alpha_nps.unsqueeze(-1) + new_bkgd * (1 - alpha_nps.unsqueeze(-1)), new_bkgd)

    rgba_img_np = torch.cat([img_nps, alpha_nps.unsqueeze(-1)], dim=-1)
    return rgba_img_np