# 3D LiDAR-Based Bird's-Eye View Object Detection via BEVFusion

[![Python 3.10](https://img.shields.io/badge/Python-3.10.22-blue.svg)](https://www.python.org/)
[![PyTorch 2.1.2](https://img.shields.io/badge/PyTorch-2.1.2%2Bcu118-red.svg)](https://pytorch.org/)
[![MMCV 2.1.0](https://img.shields.io/badge/MMCV-2.1.0-green.svg)](https://github.com/open-mmlab/mmcv)
[![CUDA 11.8](https://img.shields.io/badge/CUDA-11.8-brightgreen.svg)](https://developer.nvidia.com/cuda-toolkit)
[![nuScenes](https://img.shields.io/badge/Dataset-nuScenes-orange.svg)](https://www.nuscenes.org/)
[![Colab](https://img.shields.io/badge/Colab-Verified%20Notebook-yellow.svg)](notebooks/BEVFusion_Python3_10.ipynb)

A dedicated deep learning framework for **3D LiDAR Object Detection** using **BEVFusion** (Bird's-Eye View Representation) on the **nuScenes** autonomous driving benchmark. This repository provides end-to-end support for custom CUDA sparse voxelization, model checkpoint conversion (`spconv 1.x` to `2.x`), automated GPU batch inference, and synchronized 360-degree surround camera + BEV radar visualization pipelines.

---

## Table of Contents
1. [Demo Showcase](#demo-showcase)
2. [Model Architecture](#model-architecture)
3. [Verified Environment Specifications](#verified-environment-specifications)
4. [Google Colab Notebook](#google-colab-notebook)
5. [Quick Start (Local & Colab)](#quick-start-local--colab)
6. [nuScenes Mini Full 404-Frame Pipeline](#nuscenes-mini-full-404-frame-pipeline)
7. [Repository Structure](#repository-structure)
8. [Root Directory Scripts Reference](#root-directory-scripts-reference)
9. [Custom Changes & Upstream Patching](#custom-changes--upstream-patching)
10. [Troubleshooting & Technical Insights](#troubleshooting--technical-insights)
11. [License & Acknowledgments](#license--acknowledgments)

---

## Demo Showcase

### 1. Synchronized 360-Degree Surround Vision & 3D BEV Radar (Full 404 Frames)
The dashboard combines **all 6 surround cameras** (Front, Front-Left, Front-Right, Back, Back-Left, Back-Right) synchronized with the **Bird's-Eye View (BEV) 3D LiDAR detection map** in real time:

<p align="center">
  <img src="assets/demo_preview.gif" width="100%" alt="BEVFusion 360-Degree Multi-Camera & BEV Radar Demo Preview" />
</p>

> **Full Video (1080p, 404 frames @ 20 FPS):**  
> Download or play the full high-resolution demo video directly:  
> [**Download Full HD Video (BEVFusion_Demo_Full.mp4)**](assets/BEVFusion_Demo_Full.mp4)

<details>
<summary><b>Click to expand embedded HTML5 Video Player</b></summary>

<video src="assets/BEVFusion_Demo_Full.mp4" controls="controls" width="100%">
  Your browser does not support the video tag.
</video>
</details>

---

### 2. High-Resolution Bird's-Eye View (BEV) Detection
Detailed top-down projection showing depth-colored point cloud (Viridis colormap), ego vehicle marker, and oriented 3D bounding boxes with class labels and confidence scores:

<p align="center">
  <img src="assets/bev_visualization_sample.png" width="85%" alt="BEV 3D Object Detection Result" />
</p>

---

## Model Architecture

The implemented architecture is the **BEVFusion (LiDAR-Only)** pipeline, optimized for high-accuracy 3D oriented bounding box prediction on point clouds:

```
[Raw LiDAR Points (x, y, z, r, t)] 
       |
       v
 1. Voxelization (0.075m x 0.075m x 0.2m, 1440x1440x41 grid)
       |
       v
 2. BEVFusionSparseEncoder (4-stage 3D Sparse Convolution via spconv 2.x)
       |
       v
 3. SECOND 2D Backbone (Multi-scale BEV spatial convolutions: 128 & 256 channels)
       |
       v
 4. SECONDFPN Neck (Upsampling and concatenation to unified 512-channel BEV map)
       |
       v
 5. TransFusionHead (Transformer decoder with 200 learnable queries + Cross-Attention)
       |
       v
[3D Bounding Boxes (x, y, z, dx, dy, dz, yaw, score, class)]
```

### Detected Object Classes (nuScenes 10-Class Benchmark):
`car`, `truck`, `construction_vehicle`, `bus`, `trailer`, `barrier`, `motorcycle`, `bicycle`, `pedestrian`, `traffic_cone`.

---

## Verified Environment Specifications

| Component | Standard / Colab Specification |
| :--- | :--- |
| **Operating System** | **Ubuntu 22.04.4 LTS (Jammy Jellyfish)** / Windows 10/11 |
| **Python** | **3.10.22** (via Pixi / Conda) |
| **Python Executable**| `/content/.pixi/envs/default/bin/python` |
| **CUDA Toolkit** | **11.8 (nvcc V11.8.89)** |
| **GPU Hardware** | **NVIDIA Tesla T4 (15 GB VRAM)** |
| **PyTorch** | `torch==2.1.2+cu118`, `torchvision==0.16.2+cu118` |
| **MMCV** | `mmcv==2.1.0` (Prebuilt with CUDA 11.8 & GCC 9.3) |
| **MMEngine** | `mmengine==0.10.7` |
| **MMDetection3D** | `mmdet3d==1.4.0` (dev-1.x branch) |
| **Sparse Conv** | `spconv-cu118` |
| **Numerical Libraries** | `numpy==1.26.4`, `pandas==2.2.2`, `scipy==1.13.1` |

---

## Google Colab Notebook

The complete end-to-end workflow has been structured into **10 verified sections** inside the Jupyter notebook:
[**`notebooks/BEVFusion_Python3_10.ipynb`**](notebooks/BEVFusion_Python3_10.ipynb)

* Contains all verified outputs (log outputs, GPU benchmarks, rendered BEV figures, and embedded players).
* Ready to run directly on Google Colab with GPU T4 acceleration.

---

## Quick Start (Local & Colab)

### Step 1: Environment & Dependency Installation

```bash
# 1. Install PyTorch with CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# 2. Pin NumPy and install project dependencies
pip install numpy==1.26.4 pandas==2.2.2
pip install -r requirements.txt

# 3. Install MMEngine and MMCV 2.1.0 (CUDA 11.8)
pip install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
```

### Step 2: Setup MMDetection3D & Compile BEVFusion CUDA Ops

```bash
# 1. Clone upstream mmdetection3d (dev-1.x branch)
mkdir -p external
git clone -b dev-1.x https://github.com/open-mmlab/mmdetection3d.git external/mmdetection3d

# 2. Apply custom patches from custom_changes/
cp -r custom_changes/* external/mmdetection3d/

# 3. Install mmdet3d in development mode
cd external/mmdetection3d && python setup.py develop && cd ../..

# 4. Compile BEVFusion custom CUDA operators (bev_pool, voxel_layer)
cd external/mmdetection3d/projects/BEVFusion && python setup.py develop && cd ../../..
```

### Step 3: Download Pretrained Weights & Fix Checkpoint

The official OpenMMLab checkpoint was saved with `spconv 1.x` weight layout. We convert it to `spconv 2.x` layout before inference:

```bash
# 1. Download official weights
mkdir -p checkpoints/bevfusion_lidar
wget -O checkpoints/bevfusion_lidar/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth \
  https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth

# 2. Transpose tensor dimensions
python fix_bevfusion_checkpoint.py \
  --input checkpoints/bevfusion_lidar/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth \
  --output checkpoints/bevfusion_lidar/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933_fixed.pth
```

### Step 4: Run Quick 10-Frame Demo Inference & Visualize

```bash
# Run BEVFusion on 10 sample LiDAR frames (outputs saved to demo/outputs/preds/)
python setup_demo.py

# Render top-down BEV 2D visualizations (outputs saved to demo/outputs/vis/)
python visualize_demo.py
```

---

## nuScenes Mini Full 404-Frame Pipeline

To process continuous sequences and generate complete multi-camera + BEV radar animations:

### 1. Download & Unpack nuScenes Mini (~1.7 GB)
```bash
mkdir -p data/nuscenes
wget -c https://www.nuscenes.org/data/v1.0-mini.tgz -P data/
tar -xzf data/v1.0-mini.tgz -C data/nuscenes
```

### 2. High-Throughput GPU Inference on All 404 Frames (~60 seconds on Tesla T4)
```bash
python run_inference_all.py
```
*Processes all 404 LiDAR frames, exports predictions to `demo/outputs/preds/` with automatic fault-tolerance and resume capability.*

### 3. Generate Full HD (1080p) Synchronized Dashboard Video
```bash
python create_full_nuscenes_video.py
```
*Renders a 1920x1080 video with all 6 surround cameras on the left and dynamic 3D BEV LiDAR radar map on the right, saved at `demo/outputs/full_video/nuscenes_full_360_bev.mp4`.*

### 4. Generate 360-Degree Surround + BEV Animated GIF
```bash
python visualize_nuscenes_mini_gif.py
```
*Outputs animated sequence to `demo/outputs/nuscenes_mini_gif/nuscenes_360_surround_bev_animation.gif`.*

---

## Repository Structure

```
├── assets/                            # Demo media files for showcase and documentation
│   ├── BEVFusion_Demo_Full.mp4        # Full HD 1080p demo video (404 frames @ 20 FPS)
│   ├── demo_preview.gif               # Animated dashboard preview GIF
│   ├── demo_dashboard_poster.png      # 360-degree surround + BEV dashboard frame
│   └── bev_visualization_sample.png   # High-resolution BEV visualization output
├── notebooks/
│   └── BEVFusion_Python3_10.ipynb     # Self-contained Google Colab notebook with 51 verified outputs
├── custom_changes/                    # Upstream overrides ensuring exact reproducibility
│   ├── configs/                       # BEVFusion nuScenes 3D detection config
│   ├── projects/BEVFusion/setup.py    # Custom CUDA compilation script
│   └── mmdet3d/                       # Evaluation metric & inferencer patches
├── data/
│   ├── nuscenes/                      # nuScenes mini dataset (samples: CAM_*, LIDAR_TOP)
│   └── nuscenes_demo/                 # Standalone 6-camera demo sample
├── demo/                              # 10 default sample LiDAR frames (.bin)
├── fix_bevfusion_checkpoint.py        # spconv 1.x -> spconv 2.x tensor weight layout fixer
├── bev_inferencer.py                  # Core CLI 3D detection inference engine (alias: mmdet3d_inference2.py)
├── setup_demo.py                      # Cross-platform 10-sample inference pipeline
├── run_inference_all.py               # GPU batch inference runner for all 404 frames
├── visualize_demo.py                  # Top-down BEV rendering pipeline
├── visualize_nuscenes_mini_gif.py     # 360-degree surround vision + BEV animated GIF generator
├── create_full_nuscenes_video.py      # Full HD 1080p synchronized dashboard video creator
├── requirements.txt                   # Pinned Python package dependencies
├── AGENT.md                           # Core agent development directives (<200 words)
├── CLAUDE.md                          # Comprehensive developer workflow reference
└── README.md                          # Project documentation
```

---

## Root Directory Scripts Reference

The root directory contains dedicated, standalone scripts designed for specific execution, conversion, and visualization phases:

| Script | Purpose & Key Functionality | Primary Outputs |
| :--- | :--- | :--- |
| `setup_demo.py` | Automated quick-demo pipeline for 10 sample LiDAR frames. Features relative path resolution, auto-detection of GPU/CPU, and self-healing fallback against inferencer postprocessing errors. | `demo/outputs/preds/*.json` |
| `run_inference_all.py` | High-throughput GPU inference engine for all 404 keyframes of nuScenes Mini. Incorporates Windows CUDA DLL loaders, resume capability (skips already computed frames), and per-frame fault tolerance. | `demo/outputs/preds/<frame>.json` (404 files) |
| `visualize_demo.py` | Renders individual 2D top-down BEV figures with Viridis elevation-colored point clouds and rotated bounding box polygons with class labels and confidence scores. Matplotlib Agg backend prevents headless display errors. | `demo/outputs/vis/*.png` |
| `create_full_nuscenes_video.py` | High-performance video creation engine using pure OpenCV drawing and NumPy array slicing (renders at 20-30 FPS). Synchronizes all 6 surround cameras with the BEV radar map in RFU coordinates and encodes via ffmpeg H.264. | `demo/outputs/full_video/nuscenes_full_360_bev.mp4` |
| `visualize_nuscenes_mini_gif.py` | Multi-panel dashboard generator arranging 6 surround cameras (3 front, 3 back) alongside a large BEV detection panel. Automatically compiles multi-frame sequences into an animated GIF. | `demo/outputs/nuscenes_mini_gif/*.gif` |
| `fix_bevfusion_checkpoint.py` | Fixes tensor layout mismatch between `spconv 1.x` (`[out_ch, D, H, W, in_ch]`) and `spconv 2.x` (`[D, H, W, in_ch, out_ch]`) via `weight.permute(1, 2, 3, 4, 0).contiguous()`. | `*_fixed.pth` |
| `bev_inferencer.py` | Comprehensive command-line diagnostic tool supporting multi-modality (LiDAR, monocular, multi-modal), Open3D 3D point cloud/bounding box artifact exports (`.ply`), and 2D camera projections. | Predictions, `.ply` models, 2D projections |

---

## Custom Changes & Upstream Patching

The `custom_changes/` directory mirrors modifications applied to `external/mmdetection3d/` to resolve critical upstream bugs, compiler incompatibilities, and dataset evaluation constraints:

### 1. `custom_changes/mmdet3d/` (Core Framework Patches)

* **`mmdet3d/__init__.py` (Bypass Strict MMCV Version Assert):**
  * *Problem*: The upstream code asserts `mmcv_version < '2.2.0'`. Prebuilt wheels with custom CUDA build tags (such as `2.1.0+cu118`) fail this string comparison, raising an `AssertionError: MMCV is used but incompatible` that halts imports.
  * *Solution*: Disabled the strict version assertion to allow verified MMCV 2.1.0 builds to import seamlessly.

* **`mmdet3d/apis/inferencers/base_3d_inferencer.py` (GPU Memory & File Naming Fix):**
  * *Problem*: Attempting to convert GPU bounding box tensors directly via `.numpy().tolist()` without `.cpu()` causes a PyTorch runtime error. Furthermore, default postprocessing lacks automatic output naming based on raw LiDAR sweep filenames.
  * *Solution*: Added explicit `.cpu()` tensor transfers (`pred_instances_3d.bboxes_3d.tensor.cpu().tolist()`) and automated filename extraction to ensure prediction results are saved cleanly to `osp.join(pred_out_dir, 'preds', lidar_path + '.json')`.

* **`mmdet3d/evaluation/metrics/nuscenes_metric.py` (Partial Dataset Evaluation Patch):**
  * *Problem*: The official `NuScenesEval` module strictly enforces `assert set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens)`, assuming the entire multi-gigabyte dataset is being evaluated. When evaluating subsets (such as nuScenes Mini or selected sample frames), this assertion crashes the evaluation process.
  * *Solution*: Monkeypatched `NuScenesEval.__init__` to dynamically filter ground truth annotations, loading only the tokens that correspond to the samples actually predicted (`pred_tokens`). This enables smooth mAP and NDS metric computation on subset evaluations.

### 2. `custom_changes/projects/BEVFusion/` (CUDA Compilation Fixes)

* **`projects/BEVFusion/setup.py` (NVCC Compiler Compatibility):**
  * *Problem*: When compiling BEVFusion custom CUDA kernels (`bev_pool`, `voxel_layer`) on modern Linux distributions (Ubuntu 22.04 with GCC 11/12/14) or Windows (MSVC), NVCC 11.8 fails with `fatal error: unsupported compiler`.
  * *Solution*: Injected compilation flags into `extra_compile_args['nvcc']`:
    * `-allow-unsupported-compiler` and `-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` to override compiler version checks.
    * Explicit GPU architecture targets: `sm_70`, `sm_75` (NVIDIA Tesla T4 on Colab), `sm_80` (A100), and `sm_86` (RTX 30 series).

### 3. `custom_changes/configs/` (BEVFusion Model Tuning)

* **`bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py`:**
  * *Local Checkpoint Loading*: Configured `load_from = 'checkpoints/bevfusion_lidar/..._fixed.pth'` to eliminate external HTTP download dependencies during inference.
  * *Offline Storage*: Disabled the internal `petrel` S3 storage backend by setting `backend_args = None`, ensuring point clouds are read from local disk paths.
  * *Modality Isolation*: Enforced pure LiDAR processing (`use_lidar=True, use_camera=False`).
  * *Optimized Learning Rate*: Reduced learning rate to `lr = 0.0001` for stable transfer learning and fine-tuning without gradient divergence.

---

## Troubleshooting & Technical Insights

### 1. Checkpoint Size Mismatch (spconv 1.x vs spconv 2.x)
* **Problem**: Official checkpoint throws `RuntimeError: size mismatch for pts_middle_encoder...`
* **Cause**: `spconv 1.x` uses layout `[out_ch, D, H, W, in_ch]`, while modern `spconv 2.x` expects `[D, H, W, in_ch, out_ch]`.
* **Fix**: Run `python fix_bevfusion_checkpoint.py` which executes `weight.permute(1, 2, 3, 4, 0).contiguous()`.

### 2. Missing external/ Directory
* **Problem**: Cloning the repository does not include `external/mmdetection3d/`.
* **Cause**: `external/` is git-ignored due to size.
* **Fix**: Clone `mmdetection3d` (`-b dev-1.x`) into `external/mmdetection3d` and copy all patches from `custom_changes/`.

### 3. Linux Dynamic Linker Conflict (CXXABI_1.3.15 not found)
* **Problem**: In Colab/Pixi environments, `libstdc++.so.6` version mismatch occurs.
* **Fix**: Link the Pixi environment's newer library into the system path:
  ```bash
  cp /content/.pixi/envs/default/lib/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/libstdc++.so.6
  ```

### 4. Coordinate System Alignment (Ego-Centric RFU)
* **Convention**: nuScenes uses the RFU convention:
  * **X**: Right (Lateral)
  * **Y**: Forward (Longitudinal direction of travel)
  * **Z**: Up
* **BEV Projection**: The vehicle is pinned at the origin $(cx, cy)$ with heading straight UP ($v = cy - Y \times \text{scale}$). The open road ahead extends upward, and parked vehicles lie along the left ($X < 0$) and right ($X > 0$) curbs.

---

## License & Acknowledgments
Built upon [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) and [BEVFusion](https://github.com/mit-han-lab/bevfusion). Uses the [nuScenes](https://www.nuscenes.org/) dataset.
