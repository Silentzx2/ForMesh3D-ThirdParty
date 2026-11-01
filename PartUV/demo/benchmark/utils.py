import trimesh
import os
from tqdm import tqdm
import numpy as np

import seaborn as sns
import colorsys
import random
import argparse

try:    
    from uv_distortion import *
except:
    from pack.uv_distortion import *

import re
def parse_pipeline_log(log_path):
    """
    Parse a single pipeline log file and return relevant statistics:
      - number_of_parts
      - max_distortion
      - vertex_count
      - face_count
      - uv_count
      - total_components
      - pipeline_time
      - num_uv_calls (count of "[GET_UV] Method unwrap_aligning_BB" occurrences)
    """
    number_of_parts = 0
    max_distortion = 0.0
    vertex_count = None
    face_count = None
    uv_count = None
    total_components = None
    pipeline_time = None
    overlap_count = None
    num_uv_calls = 0  # new counter for UV calls
    
    # Regular expressions for matching
    part_regex = re.compile(r'Part\s+(\d+)\s+has\s+\d+\s+charts\s+and\s+distortion\s+([\d.]+)')
    v_regex = re.compile(r'# of V:\s+(\d+)')
    f_regex = re.compile(r'# of F:\s+(\d+)')
    uv_regex = re.compile(r'# of UV:\s+(\d+)')
    comp_regex = re.compile(r'Total components:\s+(\d+)')
    pipeline_regex = re.compile(r'Total time spent in pipeline:\s*([\d.]+)\s*seconds\.')
    overlap_regex = re.compile(r'Total overlapping triangles:\s+(\d+)')
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # "Part X has ... distortion Y"
            match_part = part_regex.search(line)
            if match_part:
                part_index = int(match_part.group(1))
                distortion = float(match_part.group(2))
                max_distortion = max(max_distortion, distortion)
                number_of_parts = max(number_of_parts, part_index + 1)
                
            # # of V
            match_v = v_regex.search(line)
            if match_v:
                vertex_count = int(match_v.group(1))
            
            # # of F
            match_f = f_regex.search(line)
            if match_f:
                face_count = int(match_f.group(1))
            
            # # of UV
            match_uv = uv_regex.search(line)
            if match_uv:
                uv_count = int(match_uv.group(1))
            
            # Total components
            match_comp = comp_regex.search(line)
            if match_comp:
                total_components = int(match_comp.group(1))
                
            # Pipeline time
            match_pipeline = pipeline_regex.search(line)
            if match_pipeline:
                pipeline_time = float(match_pipeline.group(1))
                
            # Overlapping triangles
            match_overlap = overlap_regex.search(line)
            if match_overlap:
                overlap_count = float(match_overlap.group(1))
            
            # Count occurrences of "[GET_UV] Method unwrap_aligning_BB"
            if "[GET_UV] Method unwrap_aligning_BB" in line:
                num_uv_calls += 1
                
    return {
        "filename": os.path.basename(log_path),
        "path": log_path,
        "number_of_parts": number_of_parts,
        "max_distortion": max_distortion,
        "vertex_count": vertex_count,
        "face_count": face_count,
        "uv_count": uv_count,
        "total_components": total_components,
        "pipeline_time": pipeline_time,
        "overlap_count": overlap_count,
        "num_uv_calls": num_uv_calls
    }


# ==================== Packing section (Blender OBJ/UV packing) ====================
# This section contains utilities for working with Blender via bpy:
# - import_obj: Import an OBJ and rename it in the Blender scene.
# - bpy_pack: Load, UV-pack, and export an OBJ file (overwriting or to a new path).
# ================================================================================
import bpy
import os

def import_obj(filepath, new_name="my_object"):
    # Load the OBJ file
    bpy.ops.wm.obj_import(filepath=filepath)
    
    # The newly imported object will be selected automatically,
    # so we can retrieve it with:
    obj = bpy.context.selected_objects[0]
    
    # Now rename the object
    obj.name = new_name
    return obj

#     in_path = "/home/wzn/workspace/partuv/00aee5c2fef743d69421bb642d446a5b/final_components.obj"

# INSERT_YOUR_CODE
# Remove all objects in the scene
def bpy_pack(in_path, out_path=None):

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


    in_path = in_path    
    obj = import_obj(in_path, os.path.basename(in_path))


    bpy.ops.object.select_all(action='SELECT')
        
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    # Select all UVs and pack islands
    bpy.ops.uv.select_all(action='SELECT')
    # bpy.ops.uv.average_islands_scale()

    bpy.ops.uv.pack_islands(margin=0.001)
    # save the obj with packed uv


    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.wm.obj_export(filepath=out_path if out_path is not None else in_path, export_selected_objects=True, export_materials=False,export_normals=False)
    
    return out_path



# ==================== Saving Visualization  (Color mapping) ====================
# This section contains utilities for saving the visualization of the UV mapping.
# ================================================================================



def load_color_indice_file(color_indices_path):
    with open(color_indices_path, 'r') as f:
        color_indices = [int(line.strip()) for line in f.readlines()]
    return color_indices
    
    
    
def get_2D_color_palette(indices):
    
    
    pallette_20 =    [(0.8862745098039215, 0.6470588235294118, 0.7137254901960784), (0.8823529411764706, 0.6392156862745098, 0.48627450980392156), (0.8862745098039215, 0.6784313725490196, 0.1450980392156863), (0.8705882352941177, 0.7490196078431373, 0.08235294117647059), (0.8470588235294118, 0.8, 0.06666666666666667), (0.788235294117647, 0.8274509803921568, 0.07058823529411765), (0.6313725490196078, 0.8431372549019608, 0.06274509803921569), (0.058823529411764705, 0.8588235294117647, 0.13725490196078433), (0.07058823529411765, 0.8549019607843137, 0.6431372549019608), (0.0784313725490196, 0.8470588235294118, 0.7647058823529411), (0.08235294117647059, 0.8431372549019608, 0.8235294117647058), (0.09411764705882353, 0.8352941176470589, 0.8549019607843137), (0.15294117647058825, 0.8, 0.8627450980392157), (0.21176470588235294, 0.7764705882352941, 0.8823529411764706), (0.6274509803921569, 0.7490196078431373, 0.8862745098039215), (0.7803921568627451, 0.7333333333333333, 0.8823529411764706), (0.8392156862745098, 0.6705882352941176, 0.8862745098039215), (0.8862745098039215, 0.5372549019607843, 0.8823529411764706), (0.8823529411764706, 0.592156862745098, 0.8392156862745098), (0.8862745098039215, 0.6196078431372549, 0.788235294117647), (0.07058823529411765, 0.20392156862745098, 0.33725490196078434)]    
    
    pallette_12 = [(0.8862745098039215, 0.6470588235294118, 0.7137254901960784), (0.8823529411764706, 0.6549019607843137, 0.20784313725490197), (0.8666666666666667, 0.7686274509803922, 0.07058823529411765), (0.788235294117647, 0.8274509803921568, 0.07058823529411765), (0.2980392156862745, 0.8549019607843137, 0.058823529411764705), (0.07058823529411765, 0.8509803921568627, 0.6980392156862745), (0.08235294117647059, 0.8431372549019608, 0.8235294117647058), (0.12941176470588237, 0.8117647058823529, 0.8588235294117647), (0.30196078431372547, 0.7607843137254902, 0.8862745098039215), (0.7803921568627451, 0.7333333333333333, 0.8823529411764706), (0.8705882352941177, 0.592156862745098, 0.8862745098039215), (0.8823529411764706, 0.6039215686274509, 0.8235294117647058)]
    
    
    pallette_14 = [(0.8666666666666667, 0.5254901960784314, 0.6235294117647059),
 (0.8235294117647058, 0.4823529411764706, 0.43529411764705883),
 (0.8627450980392157, 0.5647058823529412, 0.2235294117647059),
 (0.7686274509803922, 0.6745098039215687, 0.2196078431372549),
 (0.6431372549019608, 0.7137254901960784, 0.2196078431372549),
 (0.36470588235294116, 0.7450980392156863, 0.2196078431372549),
 (0.23137254901960785, 0.7372549019607844, 0.6823529411764706),
 (0.25882352941176473, 0.7176470588235294, 0.8196078431372549),
 (0.5176470588235295, 0.6823529411764706, 0.8627450980392157),
 (0.6627450980392157, 0.6549019607843137, 0.8627450980392157),
 (0.7450980392156863, 0.615686274509804, 0.8627450980392157),
 (0.8627450980392157, 0.4745098039215686, 0.803921568627451),
 (0.6392156862745098, 0.403921568627451, 0.34509803921568627),
 (0.5882352941176471, 0.5882352941176471, 0.5882352941176471)]
    
    
    return [pallette_14[i%12] for i in indices]
    
    
def get_2D_color_map(original_colors):
    
    
    pallette =   {(1.0, 0.5499793313219214, 0.6307793038399977): (0.8862745098039215,
  0.6470588235294118,
  0.7137254901960784),
 (0.9999999999999999,
  0.543120828850038,
  0.4083961577982457): (0.8823529411764706, 0.6392156862745098, 0.48627450980392156),
 (1.0, 0.5913914943779714, 0.16280786054841467): (0.8862745098039215,
  0.6784313725490196,
  0.1450980392156863),
 (0.948113267973574,
  0.6823027162769595,
  0.10554090554854145): (0.8705882352941177, 0.7490196078431373, 0.08235294117647059),
 (0.8822110117651106,
  0.7745168956010082,
  0.07970976551805631): (0.8470588235294118, 0.8, 0.06666666666666667),
 (0.7472248582278954,
  0.829324813539755,
  0.08956416311086668): (0.788235294117647,
  0.8274509803921568,
  0.07058823529411765),
 (0.5350868275048658,
  0.868133858324082,
  0.08099578200257751): (0.6313725490196078, 0.8431372549019608, 0.06274509803921569),
 (0.07260128684192202,
  0.9119109289592552,
  0.15517485477124432): (0.058823529411764705, 0.8588235294117647, 0.13725490196078433),
 (0.08675528936859234,
  0.8944944897563128,
  0.5461391148418169): (0.07058823529411765, 0.8549019607843137, 0.6431372549019608),
 (0.0957222759589974,
  0.8830687029479654,
  0.7060021324861115): (0.0784313725490196, 0.8470588235294118, 0.7647058823529411),
 (0.10310497547995956,
  0.8734288755242989,
  0.817466830046268): (0.08235294117647059, 0.8431372549019608, 0.8235294117647058),
 (0.1100185540235451,
  0.8468095300521435,
  0.9007395072037617): (0.09411764705882353, 0.8352941176470589, 0.8549019607843137),
 (0.16675901657411374,
  0.7737765310095684,
  0.925550753641895): (0.15294117647058825, 0.8, 0.8627450980392157),
 (0.2130690644518859,
  0.7242393941234814,
  0.9999999999999999): (0.21176470588235294,
  0.7764705882352941,
  0.8823529411764706),
 (0.531009756103958, 0.6856953736406072, 1.0): (0.6274509803921569,
  0.7490196078431373,
  0.8862745098039215),
 (0.736119882742555,
  0.6588364633404517,
  0.9999999999999999): (0.7803921568627451, 0.7333333333333333, 0.8823529411764706),
 (0.8619900993438493, 0.5784871904319366, 1.0): (0.8392156862745098,
  0.6705882352941176,
  0.8862745098039215),
 (1.0, 0.4484206443763319, 0.9903296003132859): (0.8862745098039215,
  0.5372549019607843,
  0.8823529411764706),
 (0.9999999999999999,
  0.49590934944682485,
  0.8558643756894744): (0.8823529411764706, 0.592156862745098, 0.8392156862745098),
 (1.0, 0.5250775929310856, 0.7496985249423401): (0.8862745098039215,
  0.6196078431372549,
  0.788235294117647)}    
    return [pallette[color] for color in original_colors]
      
    
def generate_unique_colors(num_ids, palette='husl', saturation_scale=1.5, lightness_scale=1.1, random_shuffle=True, load_color_indices=None):
    # if num_ids <= 12:
    #     step = 12 / num_ids
    #     indices = [int(round(i * step)) % 12 for i in range(num_ids)]
    # else:
    #     indices = [i % 12 for i in range(num_ids)]
    # pallette_colors = sns.color_palette(palette, 12)
    
    
    color_1_indices = [0, 1, 2, 5, 8, 10, 14, 19,21, 22, 23, 27]
    color_1 = sns.color_palette('husl', 30)
    pallette_colors = [color_1[i] for i in color_1_indices]
    random_shuffle = False
    # load_color_indices = None
    
    if load_color_indices is not None and  type(load_color_indices) == list:
            indices = load_color_indices
    elif load_color_indices is not None and type(load_color_indices) == str and os.path.exists(load_color_indices):
        with open(load_color_indices, 'r') as f:
            indices = [int(line.strip()) for line in f.readlines()]
    else:
        if num_ids <= 12:
            # without replacement   
            indices = random.sample(range(12), num_ids)
        else:
            # with replacement
            indices = [random.randint(0, 11) for i in range(num_ids)]
            
        if load_color_indices is not None:

            with open(load_color_indices, 'w') as f:
                for i in indices:
                    f.write(f"{i}\n")
            
    pallette_colors = [pallette_colors[i] for i in indices]
    
    adjusted_colors = []

    for r, g, b in pallette_colors:
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        s = max(0.0, min(1.0, s * saturation_scale))    # Reduce saturation
        l = max(0.0, min(1.0, l * lightness_scale))     # Increase lightness
        r_new, g_new, b_new = colorsys.hls_to_rgb(h, l, s)
        adjusted_colors.append((r_new, g_new, b_new))
    


    return adjusted_colors, indices

def color_mesh_components(components, color_list=None):
    """
    Colors each component of a mesh with a different color.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        The input mesh to be colored
    color_list : list of RGBA colors, optional
        List of colors to use. If None, random colors will be generated.
        Colors should be in RGBA format with values between 0 and 1.
    
    Returns:
    --------
    trimesh.Trimesh
        A new mesh with colored components
    """

    if type(components) is trimesh.Trimesh:
        components = components.split(only_watertight=False)
    
    # If no colors provided, generate random colors
    if color_list is None:
        # Generate random colors (excluding alpha)
        color_list = np.random.random((len(components), 4))
        # Set alpha to 1.0 for all colors
        color_list[:, 3] = 1.0
    
    # Make sure we have enough colors
    if len(color_list) < len(components):
        raise ValueError(f"Not enough colors provided. Need {len(components)} colors but got {len(color_list)}")
    
    # Create a list to store colored meshes
    colored_components = []
    face_colors = []
    
    # Color each component
    for idx, component in enumerate(components):
        # Create a color array for all faces in this component
        component_color = np.tile(list(color_list[idx]) + [1], (len(component.faces), 1))
        component.visual.face_colors = component_color
        colored_components.append(component)
        face_colors.append(component_color)
    
    # Combine all components back into a single mesh
    colored_mesh = trimesh.util.concatenate(colored_components)
    
    # Set the combined face colors
    colored_mesh.visual.face_colors = np.vstack(face_colors)
    return colored_mesh

def convert_facecolor_to_vertexcolor(mesh_with_facecolors: trimesh.Trimesh) -> trimesh.Trimesh:
    # Get data
    faces = mesh_with_facecolors.faces
    vertices = mesh_with_facecolors.vertices
    face_colors = mesh_with_facecolors.visual.face_colors  # (F, 4)

    # Expand each face into its own 3 unique vertices
    new_vertices = vertices[faces].reshape(-1, 3)            # (F*3, 3)
    new_faces = np.arange(len(new_vertices)).reshape(-1, 3)  # (F, 3)
    new_colors = np.repeat(face_colors, 3, axis=0)           # (F*3, 4)

    # Create a new mesh with per-vertex color
    mesh_vertex_colored = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, vertex_colors=new_colors, process=False)
    return mesh_vertex_colored


# uv_mesh = trimesh.load(os.path.join(mesh_path, "uv_all.obj"))
# uv_mesh.visual.uv = normalize_uv_charts([uv_mesh.visual.uv])[0]
# uv_mesh.export(os.path.join(mesh_path, "uv_all.obj"))

def convert_to_vertex_color(mesh_path, output_path="mesh_color2", random_shuffle=True, load_color_indices=None, part_list=[]):
    # ── load, forcing a Scene so we always get a container ───────────
    scene = trimesh.load(
        os.path.join(mesh_path, "uv_all.obj"),
        split_object=True,       # start a new mesh at every “o” line
        group_material=False,    # don’t break further on materials
        force="scene"            # guarantees a Scene even for one part
    )

    # geometry is an OrderedDict:  {object_name : Trimesh}
    # print(scene.geometry.keys())               # → dict_keys(['part_0.obj', 'part_1.obj', …])

    # if len(part_list) > 0:
    #     scene.geometry = {k: scene.geometry[f"part_{k}.obj"] for k in part_list}

    # iterate if you also need the faces / vertices of each part
    colors, indices = generate_unique_colors(len(scene.geometry), random_shuffle=random_shuffle, load_color_indices=load_color_indices)
    all_uv = []
    new_uv_color = []
    uv_mesh_components = []
    for i, (name, geom) in enumerate(scene.geometry.items()):
        # print(f"{name}: {len(geom.faces)} faces, {len(geom.vertices)} vertices")
        all_uv.append(geom.visual.uv)
        new_uv_color.append(colors[i])
        uv_mesh_components.append(geom)



    mesh_colored = color_mesh_components(uv_mesh_components, new_uv_color)
    
    new_uv_color = get_2D_color_palette(indices)
    

        
        
    # save_uv_layout_with_packing(uv_mesh_components, all_uv, os.path.join(mesh_path, f"{output_path}_uv.png"), mode=new_uv_color,
    #                         image_size=4096, save_image=True)
    
    save_uv_layout_with_packing(uv_mesh_components, all_uv, os.path.join(mesh_path, "all_uv.png"), mode=new_uv_color,
                                image_size=4096, save_image=True)
    
    save_uv_layout_with_packing(uv_mesh_components, all_uv, os.path.join(mesh_path, "distortion.png"), mode='area',
                                image_size=4096, save_image=True)
    # calculate_uv_utilization(uv_mesh)
    mesh_colored = trimesh.Trimesh(vertices=mesh_colored.vertices, faces=mesh_colored.faces, 
                                    face_colors=mesh_colored.visual.face_colors)
    mesh_colored.export(os.path.join(mesh_path, f"{output_path}.ply"))
    reload = trimesh.load(os.path.join(mesh_path, f"{output_path}.ply"))
    mesh_vertex_colored = convert_facecolor_to_vertexcolor(mesh_colored)
    mesh_vertex_colored.export(os.path.join(mesh_path, f"{output_path}.obj"))

    print(f"Exported {os.path.join(mesh_path, f'{output_path}.obj')}")

if __name__ == "__main__":  
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_path', type=str, default="/ariesdv0/zhaoning/workspace/IUV/lscm/libigl-example-project/mesh_processing/output_meshes/POT-mini2/POT-mini/00aee5c2fef743d69421bb642d446a5b/", help='Path to the mesh file')
    args = parser.parse_args()
        
    mesh_path = args.mesh_path
    
    
    indices_path = os.path.join(mesh_path, "color_indices.txt")
    convert_to_vertex_color(mesh_path, output_path="mesh_color2", random_shuffle=not os.path.exists(indices_path), load_color_indices=indices_path)