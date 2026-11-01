import blenderproc as bproc

# (rest of your pipeline script...)
bproc.init()
# install_addon.py
import bpy
import os
import   sys, re, pathlib
import argparse
import time
import json
from collections import deque
from typing import List

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--num_atlas", type=int, required=True)
args = parser.parse_args()

directory = args.input_dir  
# Path to your ZIP file
# zip_path = os.path.expanduser('/ariesdv0/zhaoning/workspace/IUV/lscm/libigl-example-project/mesh_processing/uvpackmaster3-addon-3.3.6.zip')
# bpy.ops.preferences.addon_install(filepath=zip_path, overwrite=True)

module_name = 'uvpackmaster3'
bpy.ops.preferences.addon_enable(module=module_name)

"""
Load part_*.obj files, create a UVPM3 grouping scheme, put every part's islands
into the group whose index equals the numeric suffix, and finally JOIN them
into one object (optional).

Usage
-----
blender -b -P assign_groups_from_parts.py --  /absolute/path/to/folder
"""
import bpy, sys, re, pathlib
import os


def import_obj(filepath, new_name=None):
    # Load the OBJ file
    bpy.ops.wm.obj_import(filepath=filepath)
    
    # The newly imported object will be selected automatically,
    # so we can retrieve it with:
    obj = bpy.context.selected_objects[0]
    
    if not new_name:
        new_name = os.path.basename(filepath)
    # Now rename the object
    obj.name = new_name
    return obj


def blender_pack_uv(directory):
    if os.path.exists(os.path.join(directory, "final_components.obj")):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        obj = import_obj(os.path.join(directory, "final_components.obj"))
    
        bpy.context.view_layer.objects.active = obj

        # Go to EDIT mode to work with UVs
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')

        # Select all UVs and pack islands
        bpy.ops.uv.select_all(action='SELECT')

        bpy.ops.uv.average_islands_scale()

        bpy.ops.uv.pack_islands(margin=0.001, scale=False)

        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.wm.obj_export(filepath=os.path.join(directory, f"final_packed.obj"),
                        export_materials=False,export_normals=False)
        return os.path.join(directory, f"final_packed.obj")
    else:
        raise ValueError("final_components.obj not found in the directory")


with open(os.path.join(directory, "hierarchy.json"), "r") as f:
    loaded_tree = json.load(f)
parsed_tree = {}
for k, v in loaded_tree.items():
    parsed_tree[int(k)] = v
print("\nLoaded (and parsed) tree from 'my_tree.json':")
print(parsed_tree)



def is_leaf(node):                                           # “leaf” = no children
    return "left" not in node and "right" not in node

def find_root(tree_dict):
    all_nodes = set(tree_dict.keys())
    children = set()
    for node_id, node_data in tree_dict.items():
        if "left" in node_data:
            children.add(node_data["left"])
        if "right" in node_data:
            children.add(node_data["right"])
    roots = all_nodes - children
    return max(roots, key=lambda x: tree_dict[x]['faces'])



def closest_n_nodes(parsed_tree, root_id, N):
    """
    Return the N nodes closest to `root`, using BFS.
    At each depth, expand the node with the largest `.face` value first.
    The returned list preserves the queue order after the final expansion.
    """
    q = deque([(root_id, -parsed_tree[root_id]['faces'])])               # (node, depth)
    leaves = []
    while len(q) < N and q:              # keep expanding until frontier ≥ n
        # breakpoint()
        node, depth = q.pop()
        if node not in parsed_tree:
            print(f"WARNING: {node} is not in tree??")
            continue
        if is_leaf(parsed_tree[node]):
            leaves.append(node)
            N -= 1
            
        else:
            for child in [parsed_tree[node]['left'], parsed_tree[node]['right']]:
                q.append((child, -parsed_tree[child]['faces']))
        if len(q) >= N:              # frontier big enough – stop early
            break

    return [node for node, _ in list(q)[:N]]+leaves




def covering_nodes(t, root_id: int, N: int) -> list[int]:
    """
    Return N independent nodes whose sub-trees cover every leaf,
    choosing nodes as close to the root as possible.
    """
    # depth & leaf count ------------------------------------------------------
    depth = {}
    leaves = 0
    stack = [(root_id, 0)]
    while stack:
        nid, d = stack.pop()
        depth[nid] = d
        if nid not in t:
            print(f"WARNING: {nid} is not in tree??")
            continue
        node = t[nid]
        if is_leaf(node):
            leaves += 1
        else:
            if "left" in node:
                stack.append((node["left"], d + 1))
            if "right" in node:
                stack.append((node["right"], d + 1))

    if not 1 <= N <= leaves:
        raise ValueError(f"N must be between 1 and #leaves ({leaves})")

    # iterative split ---------------------------------------------------------
    cover = [root_id]                                      # current partition
    while len(cover) < N:
        cover.sort(key=depth.__getitem__)                  # shallowest first
        # find earliest internal node
        for i, nid in enumerate(cover):
            if not is_leaf(t[nid]):
                break
        else:                                              # should never happen
            raise RuntimeError("Cannot expand further – ran out of internal nodes")

        nid = cover.pop(i)                                 # replace with children
        node = t[nid]
        if "left" in node:
            cover.append(node["left"])
        if "right" in node:
            cover.append(node["right"])

    cover.sort(key=depth.__getitem__)
    return cover
# ---------------------------------------------------------------------------

root_id = find_root(parsed_tree)
print("Root:", root_id)

N = args.num_atlas                                         # ← set desired number
nodes = closest_n_nodes(parsed_tree, root_id, N)
# breakpoint()
print(f"\n{N} covering nodes (ids, shallowest first):")
print(nodes)
print([parsed_tree[n]['faces'] for n in nodes])

def get_tree_part( tree, node_id):
    descendants = []
    stack = [node_id]
    while stack:
        current = stack.pop()
        if current not in tree:
            print(f"WARNING: {current} is not in tree??")
            continue
        if "part" in tree[current]:
            descendants.append(tree[current]["part"])
            
        elif current in tree:  # Check if it's a non-leaf node
            left, right = tree[current]['left'], tree[current]['right']
            # descendants.extend([left, right])
            stack.extend([left, right])
        else:
            print(f"WARNING: {current} is not in tree??")
    return descendants

part_to_atlas = {}
parts_in_group = []
for i, node_id in enumerate(nodes):
    curr_parts = get_tree_part(parsed_tree, node_id)
    parts_in_group += [curr_parts]
    for part in curr_parts:
        part_to_atlas[part] = i
    print(curr_parts)

# breakpoint()


# Clean up the scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

imported_objects = []

# Import and process each part
uv_offset = 0.0
uv_offset_step = 1.2  # how much to move each part in UV space to avoid overlap


bpy.context.scene.uvpm3_props.group_method = '4'

bpy.ops.uvpackmaster3.new_grouping_scheme(access_desc_id="default")
bpy.context.scene.uvpm3_props.grouping_schemes[0].name = "Scheme"

time_start = time.time()
for i, part_indices in enumerate(parts_in_group):
    
    bpy.ops.object.select_all(action='DESELECT')
    # bpy.ops.object.delete(use_global=False)
    print(f"Packing group {i} with parts {part_indices}")
    imported_objs = []
    for part_index in part_indices:
        
        #  directly loading part files....
        filepath = os.path.join(directory, "individual_parts", f"part_{part_index}.obj")
        
        obj = import_obj(filepath)
        imported_objs.append(obj)
        
    for obj in imported_objs:
        obj.select_set(True)
        

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    bpy.ops.uv.select_all(action='SELECT')

    
    bpy.ops.uvpackmaster3.new_group_info(access_desc_id="default")
    bpy.context.scene.uvpm3_props.grouping_schemes[0].active_group_idx = i
    bpy.ops.uvpackmaster3.set_island_manual_group(access_desc_id="default")
    
    bpy.ops.object.mode_set(mode='OBJECT')
    # bpy.ops.wm.obj_export(filepath=os.path.join(directory, "individual_parts", f"group_{i}_packed.obj"),
    #                       export_materials=False,
    #                       export_normals=False,
    #                       export_selected_objects=True)
    
# Select all objects for packing
bpy.ops.object.select_all(action='SELECT')

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')

# Select all UVs and pack islands
bpy.ops.uv.select_all(action='SELECT')
bpy.ops.uv.average_islands_scale()

bpy.context.scene.uvpm3_props.rotation_enable = True
bpy.context.scene.uvpm3_props.heuristic_enable = True
bpy.context.scene.uvpm3_props.heuristic_search_time = 3
bpy.context.scene.uvpm3_props.rotation_step = 16
# set the group method to manual scheme 
bpy.context.scene.uvpm3_props.default_main_props.group_method = '4'

# bpy.ops.uvpackmaster3.pack(mode_id='pack.groups_together')
bpy.ops.uvpackmaster3.pack(mode_id='pack.groups_to_tiles')



bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='SELECT')

bpy.ops.wm.obj_export(filepath=os.path.join(directory, f"final_packed.obj"),
                        export_materials=False,export_normals=False)

time_end = time.time()
print(f"UVP time taken: {time_end - time_start:.2f} seconds")