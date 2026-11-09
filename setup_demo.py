"""
setup_demo.py - Quick Automated 10-Frame Demo Inference Pipeline

Purpose:
    Executes automated 3D LiDAR object detection using the pre-trained BEVFusion model
    on the 10 demo sample point clouds located in the demo/ directory.

Key Mechanics:
    1. Cross-platform path resolution using relative base directory (works on Windows, Linux, Colab).
    2. Dynamic device selection: auto-detects CUDA GPU (cuda:0) and falls back to CPU.
    3. Auto-locates BEVFusion model config and fixed checkpoint (.pth).
    4. Automatically recovers the 10 demo .bin samples or fetches them from dataset samples.
    5. Employs the self-healing fallback mechanism from mmdet3d_inference2 to prevent
       post-processing and box_mode_3d crashes.
    6. Exports 3D bounding boxes, scores, and labels into demo/outputs/preds/*.json.
"""

import os
import shutil
import glob
import sys
import torch

# Base directory of the repository
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
    print("Please make sure mmdetection3d is installed or run:")
    print("  cd external/mmdetection3d && python setup.py develop")
    sys.exit(1)

# Device selection: auto-detect CUDA, fallback to CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Directories
demo_dir = os.path.join(base_dir, "demo")
output_dir = os.path.join(demo_dir, "outputs")
os.makedirs(demo_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Find config file
config_candidates = [
    os.path.join(base_dir, "external", "mmdetection3d", "projects", "BEVFusion", "configs", "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"),
    os.path.join(base_dir, "custom_changes", "configs", "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py")
]
config_file = next((c for c in config_candidates if os.path.exists(c)), None)
if not config_file:
    print(f"Error: BEVFusion config file not found in candidates:\n{config_candidates}")
    sys.exit(1)

# Find checkpoint file
checkpoint_candidates = [
    os.path.join(base_dir, "checkpoints", "bevfusion_lidar", "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933_fixed.pth"),
    os.path.join(base_dir, "work_dirs", "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d", "epoch_10.pth"),
    os.path.join(base_dir, "checkpoints", "bevfusion_lidar", "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth")
]
checkpoint_file = next((cp for cp in checkpoint_candidates if os.path.exists(cp)), None)
if not checkpoint_file:
    print("=" * 60)
    print("Error: Checkpoint file not found!")
    print("Please download and fix the BEVFusion checkpoint first:")
    print("  mkdir -p checkpoints/bevfusion_lidar")
    print("  wget -O checkpoints/bevfusion_lidar/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth")
    print("  python fix_bevfusion_checkpoint.py --input checkpoints/bevfusion_lidar/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth --output checkpoints/bevfusion_lidar/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933_fixed.pth")
    print("=" * 60)
    sys.exit(1)

# Check sample files: prioritize demo directory, fallback to dataset samples
demo_files = sorted(glob.glob(os.path.join(demo_dir, "*.bin")))
if len(demo_files) >= 1:
    sample_files = demo_files[:10]
    print(f"Using {len(sample_files)} sample files found in {demo_dir}")
else:
    # Look for source directory
    source_candidates = [
        os.path.join(base_dir, "data", "v1.0-mini", "samples", "LIDAR_TOP"),
        os.path.join(base_dir, "data", "nuscenes", "samples", "LIDAR_TOP"),
        os.path.join(base_dir, "data", "nuscenes_demo", "lidar")
    ]
    source_dir = next((s for s in source_candidates if os.path.exists(s) and len(glob.glob(os.path.join(s, "*.bin"))) > 0), None)
    if not source_dir:
        print(f"Error: No .bin LiDAR files found in {demo_dir} or dataset folders.")
        sys.exit(1)

    all_files = sorted(glob.glob(os.path.join(source_dir, "*.bin")))
    sample_files = []
    for src in all_files[:10]:
        filename = os.path.basename(src)
        dst = os.path.join(demo_dir, filename)
        shutil.copy2(src, dst)
        sample_files.append(dst)
        print(f"Copied {filename} to {demo_dir}")

# Run Inference
print(f"Loading config from {config_file}")
print(f"Loading weights from {checkpoint_file}")
cfg = Config.fromfile(config_file)
inferencer = LidarDet3DInferencer(model=cfg, weights=checkpoint_file, device=device)

def run_single_inference_with_fallback(pcd_path, out_dir, score_thr=0.3):
    """Fallback engine from mmdet3d_inference2 to prevent box_mode_3d crashes."""
    inputs = dict(points=pcd_path)
    json_filename = os.path.basename(pcd_path).replace('.bin', '.json')
    json_out_path = os.path.join(out_dir, "preds", json_filename)
    try:
        inferencer(inputs=inputs, out_dir=out_dir, show=False, pred_score_thr=score_thr)
    except (AttributeError, KeyError) as e:
        if 'box_mode_3d' in str(e) or 'box_type_3d' in str(e) or 'predictions' in str(e):
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
                    os.makedirs(os.path.join(out_dir, "preds"), exist_ok=True)
                    with open(json_out_path, 'w') as jf:
                        json.dump(pred_dict, jf)
        else:
            raise e

print("Starting Inference...")
for pcd_file in sample_files:
    print(f"Processing {os.path.basename(pcd_file)}...")
    run_single_inference_with_fallback(pcd_file, output_dir, score_thr=0.3)

print(f"Done. Results saved to {output_dir}")
