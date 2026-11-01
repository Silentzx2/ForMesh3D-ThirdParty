#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal benchmarking runner for PartUV unwrapping across a dataset.

Pipeline (unchanged logic):
1) PF preprocess -> (V, F, tree_dict)
2) partuv.pipeline_numpy(V, F, tree_dict, configPath, threshold=1.25, pack_final_mesh=False)
3) save_results(pack_path, final_parts, individual_parts), then flatten 'individual_parts' into pack_path
4) pack_mesh(pack_path)
5) Write timings and gather {final_packed.obj, all_uv.png, distortion.png, logs} into final_output/<mesh_name>/

Only essential flags are kept.
"""

import argparse
import glob
import os
import shutil
import sys
import time
import contextlib
import json

import torch

import partuv
from partuv.preprocess_utils.partfield_official.run_PF import PFInferenceModel
from partuv.preprocess import preprocess, save_results
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from pack.pack import pack_mesh
from pack.eval_charts import evaluate_mesh

from utils import convert_to_vertex_color, bpy_pack
import struct
from typing import Dict, Tuple, Optional
import trimesh
# ----------------------------- CLI -----------------------------

def parse_arguments():
    p = argparse.ArgumentParser("Benchmark PartUV unwrapping over a dataset")
    p.add_argument("--input_folder", "-i",
                   default="/ariesdv0/zhaoning/workspace/IUV/IUV/mesh_processing/random_obj",
                   help="Folder containing meshes (.glb/.obj/.ply/.off/.stl)")
    p.add_argument("--output_folder", "-o",
                   default="/ariesdv0/zhaoning/workspace/IUV/IUV/mesh_processing/random_obj_output",
                   help="Working root. Per-mesh subfolders created here.")
    p.add_argument("--final_output_folder", "-fo",
                   default="final_output",
                   help="Folder (under output_folder) to collect final artifacts")
    p.add_argument("--config_file", "-cf",
                   default="config/config.yaml",
                   help="Config copied to final_output/config.yaml and used by the pipeline")
    p.add_argument("--one_mesh", "-one", default=None,
                   help="Only run meshes whose path contains this substring")
    
    p.add_argument("--use_preprocessed_binary", "-ub", action="store_true",
                   help="Use preprocessed binary from PartField")
    
    p.add_argument("--threads", type=int, default=8,
                   help="OMP/MKL thread cap (preserved hardcoded default=8)")
    
    p.add_argument("--pack_method", type=str, default="blender",
                   choices=["uvpackmaster", "blender"],
                   help="Pack method")
    p.add_argument("--force_clean", "-f", action="store_true",
                help="Start by deleting final_output folder")

    return p.parse_args()


# ----------------------------- Helpers -----------------------------

def discover_meshes(input_folder):
    exts = [".glb", ".obj", ".ply", ".off", ".stl"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_folder, f"*{ext}")))
    return files


def mirror_into_workdir(mesh_paths, output_root, include_binary_file=False):
    """
    For each source mesh, create workdir: <output_root>/output/<mesh_name>/
    and copy the mesh into it (preserving original filename).
    Returns a list of copied mesh full paths.
    """
    copied = []
    work_root = os.path.join(output_root, "output")
    os.makedirs(work_root, exist_ok=True)
    for src in mesh_paths:
        mesh_name = os.path.splitext(os.path.basename(src))[0]
        mesh_dir = os.path.join(work_root, mesh_name)
        os.makedirs(mesh_dir, exist_ok=True)
        dst = os.path.join(mesh_dir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy(src, dst)
            
        if include_binary_file:
            binary_file_path = os.path.join(os.path.dirname(src), f"{mesh_name}.bin")
            if os.path.exists(binary_file_path):
                shutil.copy(binary_file_path, os.path.join(mesh_dir, f"{mesh_name}.bin"))
                
        copied.append(dst)
    return copied




def load_mesh(mesh_path, pf_model, out_dir):
    """Preprocess to (V, F, tree_dict) exactly as before."""
    mesh, tree_filename, tree_dict, preprocess_times = preprocess(mesh_path, pf_model, out_dir,save_tree_file=False, save_processed_mesh=False, sample_on_faces=10, sample_batch_size=100_000, merge_vertices_epsilon=1e-7)
    return mesh.vertices, mesh.faces, tree_dict, preprocess_times

def load_binary_tree(path: str, null_value: int = -1) -> Tuple[Dict[int, Dict[str, int]], Optional[int]]:
    """Load the tree saved by your writer. Returns (tree_dict, root_id).
    `null_value` is the sentinel used for 'no child' (default -1)."""
    with open(path, "rb") as f:
        header = f.read(4)
        if len(header) != 4:
            raise ValueError("Corrupt file: missing/short header")
        (n_nodes,) = struct.unpack("<i", header)
        payload = f.read()
    if len(payload) != 12 * n_nodes:
        raise ValueError("Corrupt file: size does not match header count")

    tree: Dict[int, Dict[str, int]] = {}
    children = set()
    for node_id, left, right in struct.iter_unpack("<iii", payload):
        tree[node_id] = {"left": left, "right": right}
        if left != null_value:  children.add(left)
        if right != null_value: children.add(right)

    roots = set(tree.keys()) - children
    root_id = next(iter(roots)) if len(roots) == 1 else None  # None if ambiguous
    return tree, root_id


def run_pipeline_and_pack(V, F, tree_dict, cfg_path, pack_path, pack_method):
    """
    Run partuv.pipeline_numpy with threshold=1.25, pack_final_mesh=False,
    then save_results + flatten 'individual_parts' and pack.
    """
    os.makedirs(pack_path, exist_ok=True)
    pipeline_times = {}

    # # Respect thread caps (same variables the original set for subprocess)
    # os.environ['OMP_NUM_THREADS'] = str(threads)
    # os.environ['MKL_NUM_THREADS'] = str(threads)
    
    print(f"Running pipeline to unwrap {pack_path}")

    start = time.time()
    final_parts, individual_parts = partuv.pipeline_numpy(
        V=V,
        F=F,
        tree_dict=tree_dict,
        configPath=cfg_path,
        threshold=1.25,
        pack_final_mesh=False
    )
        
    elapsed = time.time() - start
    
    pipeline_times["pipeline"] = elapsed  * 1000.0
    
    print(f"Total time spent in pipeline: {elapsed:.3f} seconds.")

    time_save_start = time.time()
    save_results(pack_path, final_parts, individual_parts)
    time_save_end = time.time()
    pipeline_times["save"] = (time_save_end - time_save_start) * 1000.0

    # Flatten 'individual_parts' up one level (preserved behavior)
    ind_dir = os.path.join(pack_path, "individual_parts")
    if os.path.isdir(ind_dir):
        for f in os.listdir(ind_dir):
            shutil.move(os.path.join(ind_dir, f), os.path.join(pack_path, f))


    print(f"Packing mesh: {pack_path}")

    metrics_before = evaluate_mesh(os.path.join(pack_path, "final_components.obj"))
    print(metrics_before)
    
    time_blender = pack_mesh(pack_path, uvpackmaster=False, save_visuals=True)
    
    pipeline_times["pack_blender"] = time_blender["pack_ms"]
    pipeline_times["save_visuals_blender"] = time_blender["save_visuals_ms"]

    # time_uvpackmaster = pack_mesh(pack_path, uvpackmaster=True, save_visuals=True)
    # pipeline_times["pack_uvpackmaster"] = time_uvpackmaster["pack_ms"]
    # pipeline_times["save_visuals_uvpackmaster"] = time_uvpackmaster["save_visuals_ms"]
    
    metrics = evaluate_mesh(os.path.join(pack_path, "final_packed.obj"))
    print(metrics)
    
    
    indices_path = os.path.join(pack_path, "color_indices.txt")
    
    # time_color_distortion_end = time.time()
    # pipeline_times["color_distortion"] = (time_color_distortion_end - time_color_distortion_start) * 1000.0
    
    return pipeline_times

def gather_artifacts(work_dir, final_dir, mesh_stem, pack_path, pipeline_log_path, module_times):
    """
    Copy minimal benchmarking artifacts:
      - final_packed.obj, all_uv.png, distortion.png
      - pipeline log and pf_time.log
    """
    os.makedirs(final_dir, exist_ok=True)

    # Save the module_times dictionary to module_times.json in final_dir
    if module_times != {}:
        module_times_path = os.path.join(final_dir, "module_times.json")
        with open(module_times_path, "w") as f:
            json.dump(module_times, f, indent=2)
    else:
        print(f"[{mesh_stem}] No module times found, skipping module_times.json")

    # Artifacts to collect
    wanted = ["final_packed.obj", "final_packed_uv.png", "distortion.png", "final_packed_uv_blender.png", "distortion_blender.png", "final_packed_uv_uvpackmaster.png", "distortion_uvpackmaster.png"]
    for name in wanted:
        src = os.path.join(pack_path, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final_dir, name))

    # Logs
    if os.path.exists(pipeline_log_path):
        shutil.copy(
            pipeline_log_path,
            os.path.join(final_dir, f"{mesh_stem}_pipeline.log")
        )




def unwrap_one(mesh_path, final_root, cfg_path, pf_model, threads, pack_method, use_preprocessed_binary):
    """
    Full per-mesh run in the workdir that contains mesh_path.
    """
    work_dir = os.path.dirname(mesh_path)
    mesh_stem = os.path.splitext(os.path.basename(mesh_path))[0]
    pack_path = os.path.join(work_dir, "output")
    log_dir = os.path.join(work_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    pipeline_log = os.path.join(log_dir, f"{mesh_stem}_pipeline.log")
    final_dir = os.path.join(final_root, mesh_stem)
    
    module_times = {}

    # PF preprocess timing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    t0 = time.time()
    if use_preprocessed_binary:
        binary_file_path = os.path.join(work_dir,  f"{mesh_stem}.bin")
        if not os.path.exists(binary_file_path):
            print(f"Binary file not found: {binary_file_path}")
            V, F, tree_dict, preprocess_times = load_mesh(mesh_path, pf_model, work_dir)
    
        else:
            preprocess_times = { "fix": 0, "export": 0, "pf": 0}
            time_load_start = time.time()
            tree_dict, _ = load_binary_tree(binary_file_path)
            mesh = trimesh.load(mesh_path)
            V, F = mesh.vertices, mesh.faces
            assert len(tree_dict) == F.shape[0] -1 
            time_load_end = time.time()
            preprocess_times["load"] = time_load_end - time_load_start
    
    else:
        V, F, tree_dict, preprocess_times = load_mesh(mesh_path, pf_model, work_dir)
        
        
    t1 = time.time()
    pf_ms = (t1 - t0) * 1000.0
    module_times["preprocess"] = preprocess_times


    # Clean previous run pack_path
    shutil.rmtree(pack_path, ignore_errors=True)

    pipeline_times = run_pipeline_and_pack(V, F, tree_dict, cfg_path, pack_path, pack_method)

    module_times.update(pipeline_times)
    # Gather outputs
    gather_artifacts(work_dir, final_dir, mesh_stem, pack_path, pipeline_log,  module_times)
    print(f"[{mesh_stem}] done.")


# ----------------------------- Main -----------------------------

def main():
    args = parse_arguments()

    # Final output root
    final_root = os.path.join(args.output_folder, args.final_output_folder)
    # if args.force_clean:
    #     shutil.rmtree(final_root, ignore_errors=True)
    os.makedirs(final_root, exist_ok=True)

    # Mirror dataset into working structure
    src_meshes = discover_meshes(args.input_folder)
    work_meshes = mirror_into_workdir(src_meshes, args.output_folder, include_binary_file=args.use_preprocessed_binary)

    # Copy config for provenance and use
    cfg_dst = os.path.join(final_root, "config.yaml")
    shutil.copy(args.config_file, cfg_dst)

    # One PF model reused across meshes
    pf_model = PFInferenceModel(device="cuda", checkpoint_path="model_objaverse.ckpt")

    # Run
    for src, work in zip(src_meshes, work_meshes):
        if args.one_mesh and (args.one_mesh not in work):
            continue

        skipped = False
        # Skip if already unwrapped in workdir (preserves your time-saving behavior)
        if not args.force_clean and os.path.exists(os.path.join(os.path.dirname(work), "output", "final_packed.obj")):
            print(f"[skip] Already unwrapped: {work}")
            # Still ensure artifacts are in final_output (idempotent)
            mesh_stem = os.path.splitext(os.path.basename(work))[0]
            final_dir = os.path.join(final_root, mesh_stem)
            skipped = True

            # if not os.path.exists(os.path.join(final_dir, "final_packed.obj")) and os.path.exists(os.path.join(os.path.dirname(work), "module_times.json")):
            check_list = ["data_final.json", "distortion_blender.png", "final_packed.obj", "final_packed_uv_blender.png", "module_times.json"]

            
            if not all(os.path.exists(os.path.join(final_dir, f)) for f in check_list):
                
                if not os.path.exists(os.path.join(final_dir, "module_times.json")):
                    print(f"[{mesh_stem}] No module times found, skipping re-gathering artifacts and running pipeline again")
                    skipped = False
                else:
                    print(f"[{mesh_stem}] Missing artifacts, re-gathering artifacts and running pipeline again")
                    module_times=json.load(open(os.path.join(final_dir, "module_times.json"))) 
                        # Re-gather if final folder is missing
                    gather_artifacts(
                        work_dir=os.path.dirname(work),
                        final_dir=final_dir,
                        mesh_stem=mesh_stem,
                        pack_path=os.path.join(os.path.dirname(work), "output"),
                        pipeline_log_path=os.path.join(os.path.dirname(work), "log", f"{mesh_stem}_pipeline.log"),
                        module_times=module_times
                    )

            
        if skipped:
            continue

        unwrap_one(
            mesh_path=work,
            final_root=final_root,
            cfg_path=cfg_dst,
            pf_model=pf_model,
            threads=args.threads,
            pack_method=args.pack_method,
            use_preprocessed_binary=args.use_preprocessed_binary
        )

    print("All meshes processed.")


if __name__ == "__main__":
    main()
