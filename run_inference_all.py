"""
run_inference_all.py - High-Throughput Batch GPU Inference on Full nuScenes Mini

Purpose:
    Runs batch 3D LiDAR object detection using BEVFusion across all 404 keyframes
    of the nuScenes Mini dataset to enable full-sequence video generation.

Key Mechanics:
    1. Automatic Windows CUDA DLL loader (v11.8, v12.4, v11.3) via os.add_dll_directory.
    2. Smart resume: scans demo/outputs/preds/ to skip already inferred frames.
    3. Self-healing fallback: catches 'box_mode_3d' and postprocessing exceptions,
       directly extracting predictions via model.test_step() without crashing.
    4. Fault tolerance: per-frame exception handling prevents single corrupted frames
       from stopping the 404-frame batch run.
    5. Real-time ETA and FPS progress reporting during inference.
    6. Outputs prediction JSON files to demo/outputs/preds/<frame_name>.json.
"""

import os
import glob
import json
import sys
import time
import numpy as np
import torch

# 1. Windows CUDA DLL Support (from mmdet3d_inference2)
if os.name == 'nt':
    for cuda_ver in ['v11.8', 'v12.4', 'v11.3']:
        cuda_bin = rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{cuda_ver}\bin"
        if os.path.exists(cuda_bin):
            try:
                os.add_dll_directory(cuda_bin)
            except Exception:
                pass

base_dir = os.path.dirname(os.path.abspath(__file__))

# Add mmdetection3d to sys.path
mmdet3d_path = os.path.join(base_dir, "external", "mmdetection3d")
if os.path.exists(mmdet3d_path):
    sys.path.insert(0, mmdet3d_path)
sys.path.insert(0, base_dir)

try:
    from mmdet3d.apis import LidarDet3DInferencer
    from mmengine.config import Config
except ImportError as e:
    print(f"Error importing mmdet3d: {e}")
    sys.exit(1)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Directories
demo_dir = os.path.join(base_dir, "demo")
output_dir = os.path.join(demo_dir, "outputs")
preds_dir = os.path.join(output_dir, "preds")
os.makedirs(preds_dir, exist_ok=True)

# Find config
config_candidates = [
    os.path.join(base_dir, "external", "mmdetection3d", "projects", "BEVFusion", "configs", "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"),
    os.path.join(base_dir, "custom_changes", "configs", "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py")
]
config_file = next((c for c in config_candidates if os.path.exists(c)), None)

# Find checkpoint
checkpoint_candidates = [
    os.path.join(base_dir, "checkpoints", "bevfusion_lidar", "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933_fixed.pth"),
    os.path.join(base_dir, "work_dirs", "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d", "epoch_10.pth"),
    os.path.join(base_dir, "checkpoints", "bevfusion_lidar", "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth")
]
checkpoint_file = next((cp for cp in checkpoint_candidates if os.path.exists(cp)), None)

if not config_file or not checkpoint_file:
    print("Error: Config or Checkpoint file not found!")
    sys.exit(1)

# Find ALL LiDAR files from nuScenes mini
data_roots = [
    os.path.join(base_dir, "data", "nuscenes", "samples", "LIDAR_TOP"),
    os.path.join(base_dir, "data", "v1.0-mini", "samples", "LIDAR_TOP"),
    os.path.join(base_dir, "data", "samples", "LIDAR_TOP"),
    "/content/data/nuscenes/samples/LIDAR_TOP"
]
lidar_folder = next((p for p in data_roots if os.path.exists(p)), None)

if not lidar_folder:
    print("Error: nuScenes samples/LIDAR_TOP directory not found!")
    sys.exit(1)

all_lidar_files = sorted(glob.glob(os.path.join(lidar_folder, "*.bin")))
total_files = len(all_lidar_files)
print(f"Found {total_files} LiDAR files to infer with BEVFusion.")

# Check which ones are already processed to allow resume
pending_files = []
for f in all_lidar_files:
    json_name = os.path.basename(f).replace('.bin', '.json')
    if not os.path.exists(os.path.join(preds_dir, json_name)):
        pending_files.append(f)

print(f"Already processed: {total_files - len(pending_files)} files.")
print(f"Remaining to process: {len(pending_files)} files.")

if len(pending_files) == 0:
    print("All frames have already been processed!")
    sys.exit(0)

# Initialize BEVFusion Inferencer
print(f"\nLoading BEVFusion model onto {device}...")
cfg = Config.fromfile(config_file)
inferencer = LidarDet3DInferencer(model=cfg, weights=checkpoint_file, device=device)

def run_single_inference_with_fallback(pcd_path, out_dir, score_thr=0.3):
    """
    Infers a single LiDAR point cloud file with the robust fallback 
    from mmdet3d_inference2 to prevent crashes on box_mode_3d/postprocessing bugs.
    """
    inputs = dict(points=pcd_path)
    json_filename = os.path.basename(pcd_path).replace('.bin', '.json')
    json_out_path = os.path.join(out_dir, "preds", json_filename)

    try:
        # Standard inferencer call
        inferencer(inputs=inputs, out_dir=out_dir, show=False, pred_score_thr=score_thr)
    except (AttributeError, KeyError) as e:
        if 'box_mode_3d' in str(e) or 'box_type_3d' in str(e) or 'predictions' in str(e):
            # Bypass fallback from mmdet3d_inference2: extract directly from model
            preprocessed = list(inferencer.preprocess(inputs, 0))
            if len(preprocessed) > 0:
                data_sample = preprocessed[0]
                with torch.no_grad():
                    preds = inferencer.model.test_step([data_sample])
                if preds and hasattr(preds[0], 'pred_instances_3d'):
                    inst = preds[0].pred_instances_3d
                    if score_thr > 0:
                        inst = inst[inst.scores_3d >= score_thr]
                    
                    bboxes = inst.bboxes_3d.cpu().numpy() if hasattr(inst.bboxes_3d, 'cpu') else inst.bboxes_3d
                    scores = inst.scores_3d.cpu().numpy() if hasattr(inst.scores_3d, 'cpu') else inst.scores_3d
                    labels = inst.labels_3d.cpu().numpy() if hasattr(inst.labels_3d, 'cpu') else inst.labels_3d

                    pred_dict = {
                        'bboxes_3d': bboxes.tolist() if isinstance(bboxes, np.ndarray) else bboxes,
                        'scores_3d': scores.tolist() if isinstance(scores, np.ndarray) else scores,
                        'labels_3d': labels.tolist() if isinstance(labels, np.ndarray) else labels,
                    }
                    with open(json_out_path, 'w') as jf:
                        json.dump(pred_dict, jf)
        else:
            raise e

print(f"Starting GPU inference on {len(pending_files)} files (Enhanced Version 2 Engine)...")
start_time = time.time()

for idx, pcd_file in enumerate(pending_files, start=1):
    try:
        run_single_inference_with_fallback(pcd_file, output_dir, score_thr=0.3)
    except Exception as err:
        print(f"  Warning: Skipped frame {os.path.basename(pcd_file)} due to error: {err}")

    if idx % 25 == 0 or idx == len(pending_files):
        elapsed = time.time() - start_time
        speed = idx / elapsed
        remaining = (len(pending_files) - idx) / speed if speed > 0 else 0
        print(f"  Inferred {idx:03d}/{len(pending_files)} ({idx/len(pending_files)*100:.1f}%) - Speed: {speed:.1f} fps - Est. remaining: {remaining:.0f}s")

total_time = time.time() - start_time
print("=" * 70)
print(f"DONE! Inferred {len(pending_files)} frames in {total_time:.1f}s.")
print(f"All predictions saved to: {preds_dir}")
print("=" * 70)
