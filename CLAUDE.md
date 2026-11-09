# CLAUDE.md - BEVFusion 3D LiDAR Object Detection

## Project Overview
This repository implements, evaluates, and visualizes 3D Object Detection using **BEVFusion** (LiDAR-only architecture) on the **nuScenes** autonomous driving benchmark. Built upon OpenMMLab's `mmdetection3d`, it provides automated GPU batch inference, custom CUDA operators (`bev_pool`, `voxel_layer`), sparse convolution checkpoint converters, and synchronized 360-degree multi-camera + BEV radar visualization pipelines.

---

## Environment & Requirements

- **OS**: Ubuntu 22.04 LTS (Jammy Jellyfish) / Windows 10/11
- **Python**: 3.10.22 (Pixi / Conda)
- **PyTorch**: 2.1.2 with CUDA 11.8 (`torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2`)
- **Key Libraries**:
  - `numpy==1.26.4` (pinned for binary compatibility with compiled ops)
  - `mmcv==2.1.0` (compiled with CUDA 11.8)
  - `mmengine==0.10.7`
  - `mmdet3d>=1.4.0` (dev-1.x branch)
  - `nuscenes-devkit`, `open3d>=0.17.0`, `opencv-python`, `matplotlib`, `pillow`
- **BEVFusion CUDA Ops Compilation**:
  ```bash
  cd external/mmdetection3d/projects/BEVFusion && python setup.py develop && cd ../../..
  ```

---

## Upstream Patching Workflow (`custom_changes/`)

Because `external/mmdetection3d/` is git-ignored, all critical patches are maintained in `custom_changes/`. Always apply these patches before building `mmdet3d`:

```bash
# Linux / macOS / Colab
cp -r custom_changes/* external/mmdetection3d/

# Windows PowerShell
Copy-Item -Recurse -Force custom_changes\* external\mmdetection3d\
```

Detailed patch breakdown is documented in `custom_changes/README.md`:
- `mmdet3d/__init__.py`: Disables strict MMCV version assert to allow custom cu118 wheels.
- `mmdet3d/apis/inferencers/base_3d_inferencer.py`: Adds `.cpu()` tensor transfer and automated prediction naming.
- `mmdet3d/evaluation/metrics/nuscenes_metric.py`: Dynamically filters ground truth for partial dataset evaluation.
- `projects/BEVFusion/setup.py`: Injects NVCC flags (`-allow-unsupported-compiler`, `sm_75` for Tesla T4).
- `configs/`: Sets local checkpoint path, disables petrel S3 storage, enforces LiDAR-only, tunes learning rate.

---

## Essential Commands & Execution Workflows

### 1. Pretrained Weights & Checkpoint Fix
BEVFusion weights require transposing `spconv 1.x` `[out, D, H, W, in]` to `spconv 2.x` `[D, H, W, in, out]`:
```bash
# 1. Download official weights
mkdir -p checkpoints/bevfusion_lidar
wget -O checkpoints/bevfusion_lidar/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth \
  https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth

# 2. Fix tensor layout
python fix_bevfusion_checkpoint.py \
  --input checkpoints/bevfusion_lidar/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth \
  --output checkpoints/bevfusion_lidar/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933_fixed.pth
```

### 2. Quick Demo (10 Sample Frames)
```bash
# Run inference on sample LiDAR files in demo/
python setup_demo.py

# Render top-down BEV images to demo/outputs/vis/
python visualize_demo.py
```

### 3. nuScenes Mini Full 404-Frame Pipeline
```bash
# 1. Download & unpack nuScenes Mini
mkdir -p data/nuscenes
wget -c https://www.nuscenes.org/data/v1.0-mini.tgz -P data/
tar -xzf data/v1.0-mini.tgz -C data/nuscenes

# 2. Run GPU batch inference on all 404 keyframes (~60 seconds on Tesla T4)
python run_inference_all.py

# 3. Generate Full HD 1080p synchronized 6-camera + BEV video
python create_full_nuscenes_video.py

# 4. Generate 360-degree surround vision + BEV animated GIF
python visualize_nuscenes_mini_gif.py
```

---

## Asset & Notebook Storage Guidelines

- **Demo Media (`assets/`)**: All public showcase images, videos, posters, and animated GIFs are stored in `assets/`:
  - `assets/BEVFusion_Demo_Full.mp4`: 1080p Full HD video (404 frames @ 20 FPS).
  - `assets/demo_preview.gif`: Animated dashboard preview.
  - `assets/bev_visualization_sample.png`: High-resolution BEV detection plot.
  - `assets/demo_dashboard_poster.png`: 360-degree surround frame.
- **Development Notebooks (`notebooks/`)**: The development notebook (`notebooks/BEVFusion_Python3_10.ipynb`) is preserved on disk for development and experimentation, but is excluded from Git via `.gitignore` pending review.

---

## Coordinate Reference Frame (Ego-Centric RFU)

The nuScenes dataset adheres to the RFU coordinate convention:
- **X-axis**: Points to the **right** of the vehicle (lateral).
- **Y-axis**: Points **forward** along the direction of travel (longitudinal).
- **Z-axis**: Points **up** perpendicular to the ground.

When rendering top-down BEV radar displays:
- The ego vehicle is pinned at origin `(cx, cy)` pointing UP.
- Lateral coordinates map to horizontal pixels: `u = cx + X * scale`.
- Longitudinal coordinates map to vertical pixels: `v = cy - Y * scale` (forward Y extends upward on screen).

---

## Model Architecture: BEVFusion (LiDAR-Only)

1. **Voxelization**: Quantizes LiDAR point clouds $(x, y, z, \text{intensity}, \text{timestamp})$ into $0.075\text{m} \times 0.075\text{m} \times 0.2\text{m}$ voxels over a $108\text{m} \times 108\text{m} \times 8\text{m}$ area ($1440 \times 1440 \times 41$ grid).
2. **BEVFusionSparseEncoder**: 4-stage 3D sparse convolutional neural network (`spconv 2.x`) mapping 5D point features to 256 sparse channels.
3. **SECOND 2D Backbone**: Applies 2D convolutions over flattened BEV feature maps with multi-scale downsampling.
4. **SECONDFPN Neck**: Upsamples and concatenates features into a unified 512-channel BEV spatial representation.
5. **TransFusionHead**: Transformer decoder utilizing 200 learnable object queries with cross-attention to predict 3D bounding boxes for 10 nuScenes classes.

---

## Development & Code Style Conventions

- Follow PEP 8 guidelines across all Python scripts.
- Avoid committing raw datasets or model checkpoint binaries (`.pth`, `.bin`, `.tar`).
- Preserve self-healing fallback handlers in inferencer scripts to prevent runtime postprocessing crashes.
- Maintain zero emojis in all technical documentation files.
