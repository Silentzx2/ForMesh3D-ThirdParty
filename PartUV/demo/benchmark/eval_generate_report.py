#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import fnmatch
import base64
import json, re
import argparse
from pathlib import Path

from typing import Union
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
from tqdm import tqdm

from utils import parse_pipeline_log
from pack.eval_charts import (
    evaluate_mesh,
    count_zero_area_face_mesh,
)

# -------------------------- Utils --------------------------


def encode_image_to_base64(image_path: str):
    if not os.path.exists(image_path):
        return None
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def _flatten_dict(d, parent_key=''):
    items = []
    if not isinstance(d, dict):
        return items
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key))
        else:
            items.append((new_key, v))
    return items

def _format_module_times(module_times: dict) -> str:
    if not isinstance(module_times, dict) or not module_times:
        return "N/A"
    parts = []
    for k, v in _flatten_dict(module_times):
        if isinstance(v, (int, float)):
            parts.append(f"{k}: {v:.3f}")
        else:
            parts.append(f"{k}: {v}")
    return "<br/>".join(parts)

def update_stats(stats, method, data_dict):
    # Accumulate scalar metrics
    stats['time'                  ][method] += data_dict['time']
    stats['charts'                ][method] += data_dict['num_charts']
    stats['seam_length'           ][method] += data_dict['seam_length']
    stats['area_distortion_nuvo'  ][method] += data_dict['area_distortion_nuvo']

    # Efficiency (ignore >1 as invalid same as before)
    if data_dict['efficiency'] <= 1:
        stats['efficiency'][method] += data_dict['efficiency']
    else:
        stats['invalid_efficiency_count'][method] += 1

    # Angular metrics (skip NaNs)
    if (not np.isnan(data_dict['angular_distortion_nuvo'])
        and not np.isnan(data_dict['angular_distortion_mean'])):
        stats['angular_distortion_mean'][method] += data_dict['angular_distortion_mean']
        stats['angular_distortion_nuvo'][method] += data_dict['angular_distortion_nuvo']
        stats['angular_distortion_tile95_data'][method].append(data_dict['angular_distortion_mean'])
    else:
        stats['angular_nan_count'][method] += 1

    # Distortion metrics with inf-handling
    if data_dict['max_distortion'] != float('inf'):
        stats['max_distortion'][method] += data_dict['max_distortion']
        stats['distortion_tile95_data'][method].append(data_dict['max_distortion'])

    if data_dict['overall_distortion'] != float('inf'):
        stats['total_distortion'][method] += data_dict['overall_distortion']
    else:
        stats['inf_count'][method] += 1

    if (data_dict['total_distortion_0_1'] != float('inf')
        and not np.isnan(data_dict['total_distortion_0_1'])):
        stats['total_distortion_0_1'][method] += data_dict['total_distortion_0_1']
    else:
        stats['inf_count_0_1'][method] += 1

    # Other counters
    stats['overlap_count'][method] += 1 if data_dict['overlap_count'] > 0 else 0
    stats['zero_area_count'][method] += 1 if data_dict['zero_area_count'] > 0 else 0
    stats['distortion_tile95'][method] += data_dict['distortion_tile95']
    stats['angular_distortion_tile95'][method] += data_dict['angular_distortion_tile95']

    if data_dict['L2_invalid'] or data_dict['L2_mesh'] in (float('inf'),) or np.isnan(data_dict['L2_mesh']):
        stats['L2_invalid'][method] += 1
    else:
        stats['L2_mesh'][method] += data_dict['L2_mesh']

    if data_dict['Linf_invalid'] or data_dict['Linf_mesh'] in (float('inf'),) or np.isnan(data_dict['Linf_mesh']):
        stats['Linf_invalid'][method] += 1
    else:
        stats['Linf_mesh'][method] += data_dict['Linf_mesh']

    stats['num_invalid'][method] += data_dict['num_invalid']

    # Arrays
    stats['median_num_charts'   ][method].append(data_dict['num_charts'])
    stats['median_seam_length'  ][method].append(data_dict['seam_length'])
    
    stats['has_zero_area_face'][method] = data_dict['has_zero_area_face']

def _add_stats(target, delta):
    for metric in delta:
        for method in delta[metric]:
            if isinstance(delta[metric][method], list):
                target[metric][method].extend(delta[metric][method])
            else:
                target[metric][method] += delta[metric][method]

# -------------------------- Worker --------------------------

def _process_one(InputArgs):
    folder, METHODS, METRICS, NEW_METRICS, re_eval, args = InputArgs

    stats = {m: {meth: 0 for meth in METHODS} for m in METRICS}
    for new_metric in NEW_METRICS:
        stats[new_metric] = {m: [] for m in METHODS}

    html_content = []
    overlap_inc = 0
    folder_name = os.path.basename(folder)

    # Find pipeline log
    log_file = None
    for lf in glob.glob(os.path.join(folder, '*pipeline.log')):
        log_file = lf
        break



    log_stats = {}
    if log_file and os.path.isfile(log_file):
        log_stats = parse_pipeline_log(log_file)

    module_times = {}
    if os.path.exists(os.path.join(folder, 'module_times.json')):
        module_times = json.load(open(os.path.join(folder, 'module_times.json')))
    else:
        print(f"WARNING: No module_times.json found in {folder}!!!!!!!!!!")
    

    #   fix the preprocess time to ms
    if 'preprocess' in module_times:
        module_times['preprocess']['load'] *= 1000.0
        module_times['preprocess']['fix'] *= 1000.0
        module_times['preprocess']['export'] *= 1000.0
        module_times['preprocess']['pf'] *= 1000.0

    
    module_times_str = _format_module_times(module_times)
        

    # Images (per-method): final_packed_uv_{method}.png, distortion_{method}.png
    uv_paths   = sorted(glob.glob(os.path.join(folder, 'final_packed_uv_*.png')))
    dist_paths = sorted(glob.glob(os.path.join(folder, 'distortion_*.png')))

    # Load or re-eval ours
    if args.load_metrics and os.path.exists(os.path.join(folder, 'data_final.json')):
        data_file = json.load(open(os.path.join(folder, 'data_final.json')))
        ours_data = data_file.get('ours', None)
        re_eval = False if ours_data is not None else True
    else:
        re_eval = True

    if re_eval:
        uv_obj = os.path.join(folder, 'final_packed.obj')
        if not os.path.exists(uv_obj):
            # nothing to evaluate, skip this folder entirely
            return [], {m: {meth: 0 for meth in ['ours']} for m in METRICS}, 0
        ours_data = evaluate_mesh(uv_obj)

    if ours_data:
            
        preprocess_time_ms = sum(module_times['preprocess'].values())
        pipeline_time_ms = module_times.get('pipeline', -1) or -1
        if preprocess_time_ms <= 0 :
            print(f"[{folder_name}] preprocess time is negative: {preprocess_time_ms}")
        ours_data['time'] = (pipeline_time_ms if pipeline_time_ms is not None else -1) + (preprocess_time_ms if preprocess_time_ms >= 0 else 0.0)
        ours_data['time'] /= 1000.0

        # Extra checks
        zc, face_idx = count_zero_area_face_mesh(os.path.join(folder, 'final_packed.obj'), return_zero_face_idx=True)
        ours_data['zero_area_count'] = zc
        if zc > 0:
            print(f"[{folder_name}] zero area faces: {face_idx}")

        # Store per-mesh metrics for re-load later
        with open(os.path.join(folder, 'data_final.json'), 'w') as f:
            json.dump({'ours': ours_data}, f, indent=4)

        
        update_stats(stats, 'ours', ours_data)

    # Pull a few values from the log for table display
    number_of_parts  = log_stats.get('number_of_parts', 'N/A')
    max_distortion   = ours_data.get('max_distortion', 'N/A')
    vertex_count     = ours_data.get('vertex_count', 'N/A')
    face_count       = ours_data.get('face_count', 'N/A')
    total_charts     = ours_data.get('num_charts', 'N/A')
    run_time         = ours_data.get('time', 'N/A')
    overlap_count    = ours_data.get('overlap_count', 'N/A')

    if overlap_count not in (None, 'N/A'):
        try:
            if int(overlap_count) > 0:
                overlap_inc += 1
        except:
            pass

    # Build HTML row (ours only)
    info = ""
    if ours_data:
        metric_str = " ".join([f"{k}: {v:.3f}" if isinstance(v, (float, int)) else f"{k}: {v}"
                               for k, v in reversed(ours_data.items())])
        info = f"<h4>Ours: {metric_str}</h4>"

    # Build HTML blocks showing all available images per type (N/A if none)
    if uv_paths:
        uv_blocks = [info] if info else []
        for p in uv_paths:
            b64 = encode_image_to_base64(p)
            if not b64:
                continue
            method = os.path.basename(p).replace('final_packed_uv_', '').replace('.png', '')
            uv_blocks.append(
                f"<div style='display:inline-block;margin:4px;vertical-align:top;'>"
                f"<div style='font-size:12px;text-align:center'>{method}</div>"
                f"<img src='data:image/png;base64,{b64}' width='400'/>"
                f"</div>"
            )
        all_uv_html = ''.join(uv_blocks) if uv_blocks else "N/A"
    else:
        all_uv_html = "N/A"

    if dist_paths:
        dist_blocks = []
        for p in dist_paths:
            b64 = encode_image_to_base64(p)
            if not b64:
                continue
            method = os.path.basename(p).replace('distortion_', '').replace('.png', '')
            dist_blocks.append(
                f"<div style='display:inline-block;margin:4px;vertical-align:top;'>"
                f"<div style='font-size:12px;text-align:center'>{method}</div>"
                f"<img src='data:image/png;base64,{b64}' width='400'/>"
                f"</div>"
            )
        distortion_html = ''.join(dist_blocks) if dist_blocks else "N/A"
    else:
        distortion_html = "N/A"

    html_content.append("<tr>")
    html_content.append(f"<td>{folder_name}</td>")
    html_content.append(f"<td>{number_of_parts}</td>")
    html_content.append(f"<td>{max_distortion}</td>")
    html_content.append(f"<td>{vertex_count}</td>")
    html_content.append(f"<td>{face_count}</td>")
    html_content.append(f"<td>{total_charts}</td>")
    html_content.append(f"<td>{run_time}</td>")
    html_content.append(f"<td>{overlap_count}</td>")
    html_content.append(f"<td>{module_times_str}</td>")
    html_content.append(f"<td>{all_uv_html}</td>")
    html_content.append(f"<td>{distortion_html}</td>")
    html_content.append("</tr>")

    return html_content, stats, overlap_inc

# -------------------------- Report --------------------------

def generate_html_report(data_dir='data', output_html='report.html', re_eval=False, args=None):
    # Collect folders that have minimal artifacts from the benchmarking pipeline
    all_subfolders = []
    for root, dirs, files in os.walk(data_dir):
        has_uv   = any(fnmatch.fnmatch(f, 'final_packed_uv_*.png') for f in files)
        has_dist = any(fnmatch.fnmatch(f, 'distortion_*.png') for f in files)
        if has_uv or has_dist or ('final_packed.obj' in files):
            all_subfolders.append(root)

    if args.one:
        all_subfolders = [p for p in all_subfolders if args.one in p]
        print(f"Processing one mesh filter: {args.one}")

    html_content = []
    html_content.append("<html>")
    html_content.append("<head><meta charset='utf-8'><title>PartUV Benchmark Report</title></head>")
    html_content.append("<body>")
    html_content.append("<h1>PartUV Benchmark Report</h1>")

    # Table header (ours only)
    html_content.append("<table border='1' style='border-collapse: collapse; width:100%;'>")
    html_content.append(
        "<tr>"
        "<th>Mesh ID</th>"
        "<th>Number of Parts</th>"
        "<th>Max Distortion</th>"
        "<th>Vertex Count</th>"
        "<th>Face Count</th>"
        "<th>Total Charts</th>"
        "<th>Pipeline Time (s)</th>"
        "<th>Overlap Count</th>"
        "<th>Module Times (ms) </th>"
        "<th>final_packed_uv_* (all)</th>"
        "<th>distortion_* (all)</th>"
        "</tr>"
    )

    METHODS = ['ours']
    METRICS = [
        'time', 'charts', 'total_distortion', 'total_distortion_0_1',
        'overlap_count', 'seam_length', 'efficiency', 'angular_distortion_mean',
        'angular_distortion_nuvo', 'area_distortion_nuvo', 'max_distortion',
        'zero_area_count', 'distortion_tile95', 'angular_distortion_tile95',
        'L2_mesh', 'Linf_mesh', 'L2_invalid', 'Linf_invalid', 'num_invalid',
        'angular_nan_count', 'inf_count', 'inf_count_0_1', 'timeout_count',
        'invalid_efficiency_count', 'has_zero_area_face'
    ]
    NEW_METRICS = ['median_num_charts', 'median_seam_length',
                   'distortion_tile95_data', 'angular_distortion_tile95_data']

    stats = {metric: {m: 0 for m in METHODS} for metric in METRICS}
    for new_metric in NEW_METRICS:
        stats[new_metric] = {m: [] for m in METHODS}

    total_in_report = 0
    overlap_file_count = 0

    # Use a modest parallelism by default
    max_workers = max(1, (mp.cpu_count() // 4) or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        worker_args = [
            (folder, METHODS, METRICS, NEW_METRICS, re_eval, args)
            for folder in sorted(all_subfolders)
        ]
        futures = {pool.submit(_process_one, wa): wa[0] for wa in worker_args}

        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc="Processing",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"):
            # try:
            html_row, stats_delta, overlap_inc = fut.result()
            # except Exception as e:
            #     print(f"[ERROR] {futures[fut]} → {e}")
            #     continue


            html_content += html_row
            _add_stats(stats, stats_delta)
            overlap_file_count += overlap_inc
            total_in_report += 1

    # Close table
    html_content.append("</table>")

    # ---- Summary (ours only) ----
    html_content.append("<h2>Summary Statistics (Ours)</h2>")

    denom_time = max(total_in_report - stats['timeout_count']['ours'], 1)
    denom_total = max(total_in_report, 1)

    avg = {
        'time ↓': (stats['time']['ours'] / denom_time, denom_time),
        'charts ↓': (stats['charts']['ours'] / denom_total, denom_total),
        'median_num_charts ↓': (np.median(stats['median_num_charts']['ours']) if stats['median_num_charts']['ours'] else 0, denom_total),
        'median_seam_length ↓': (np.median(stats['median_seam_length']['ours']) if stats['median_seam_length']['ours'] else 0, denom_total),
        'total_distortion ↓': (stats['total_distortion']['ours'] / max(denom_total - stats['inf_count']['ours'], 1),
                               max(denom_total - stats['inf_count']['ours'], 1)),
        'total_distortion_0_1 ↓': (stats['total_distortion_0_1']['ours'] / max(denom_total - stats['inf_count_0_1']['ours'], 1),
                                   max(denom_total - stats['inf_count_0_1']['ours'], 1)),
        'overlap_count ↓': (stats['overlap_count']['ours'], denom_total),
        'seam_length ↓': (stats['seam_length']['ours'] / denom_total, denom_total),
        'efficiency ↑': (stats['efficiency']['ours'] / max(denom_total - stats['invalid_efficiency_count']['ours'], 1),
                         max(denom_total - stats['invalid_efficiency_count']['ours'], 1)),
        'angular_distortion_mean ↑': (stats['angular_distortion_mean']['ours'] / denom_total, denom_total),
        'angular_distortion_nuvo ↑': (stats['angular_distortion_nuvo']['ours'] / max(denom_total - stats['angular_nan_count']['ours'], 1),
                                      max(denom_total - stats['angular_nan_count']['ours'], 1)),
        'area_distortion_nuvo ↑': (stats['area_distortion_nuvo']['ours'] / denom_total, denom_total),
        'max_distortion ↓': (stats['max_distortion']['ours'] / max(denom_total - stats['inf_count']['ours'], 1),
                             max(denom_total - stats['inf_count']['ours'], 1)),
        'zero_area_count ↓': (stats['zero_area_count']['ours'], denom_total),
    }

    # Save metrics JSON next to HTML (using args.json_dir)
    json_output_path = os.path.join(args.json_dir, os.path.splitext(os.path.basename(output_html))[0] + "_metrics.json")
    print(f"Saving metrics to {json_output_path}")
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    with open(json_output_path, 'w', encoding='utf-8') as jf:
        json.dump({'ours': avg}, jf, indent=4)

    
    # Show summary in HTML
    html_content.append("<h3>Aggregates</h3>")
    for k, (val, valid_n) in avg.items():
        html_content.append(f"<p>{k}: {val:.5f} (valid: {valid_n})</p>")

    html_content.append("</body></html>")

    # Write HTML
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_content))

# -------------------------- CLI --------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PartUV-only HTML report.")
    parser.add_argument("--data_dir", type=str,
                        required=True,
                        help="Directory containing per-mesh final outputs.")
    parser.add_argument("--output_html", type=str,
                        default="partuv_benchmark_report.html",
                        help="Output HTML filename (will be placed under html/).")
    parser.add_argument("--force_delete_html", "-f", action="store_true",
                        help="Delete the existing HTML before generating.")
    parser.add_argument("--load_metrics", "-lm", action="store_true",
                        help="Load metrics from per-mesh data_final.json if present.")
    parser.add_argument("--json_dir", "-j", type=str, default=None,
                        help="Directory to write aggregated metrics JSON.")
    parser.add_argument("--one", "-one", default=None, help="Substring filter for a single mesh id.")

    args = parser.parse_args()

    if args.json_dir is None:
        args.json_dir = os.path.join(os.path.dirname(args.data_dir), "json/")
    os.makedirs(args.json_dir, exist_ok=True)
    output_html = os.path.join(os.path.dirname(args.data_dir), "html/", args.output_html)
    os.makedirs(os.path.dirname(output_html), exist_ok=True)

    if args.force_delete_html and os.path.exists(output_html):
        os.remove(output_html)

    if os.path.exists(output_html):
        print(f"File {output_html} already exists. Remove it or use -f to overwrite.")
    else:
        generate_html_report(
            data_dir=args.data_dir,
            output_html=output_html,
            re_eval=False,
            args=args
        )
        print(f"Report written to {output_html}")
