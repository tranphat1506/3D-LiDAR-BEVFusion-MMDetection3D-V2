# Custom Changes and Upstream Patch Documentation

## Overview

This directory contains all custom modifications, bug fixes, compiler overrides, and configuration files applied to OpenMMLab's upstream `mmdetection3d` repository (specifically the `dev-1.x` branch).

Because the upstream `external/mmdetection3d/` repository is excluded from Git tracking via `.gitignore` due to size and third-party dependencies, this `custom_changes/` folder acts as the permanent, version-controlled source of truth. Whenever setting up the project on a new system or Google Colab, the contents of this folder must be copied over `external/mmdetection3d/` before building the libraries.

---

## How to Apply Patches

After cloning the upstream `mmdetection3d` repository into `external/mmdetection3d/`:

### Linux / Google Colab / macOS
```bash
cp -r custom_changes/* external/mmdetection3d/
```

### Windows (PowerShell)
```powershell
Copy-Item -Recurse -Force custom_changes\* external\mmdetection3d\
```

---

## Detailed Breakdown of Custom Modifications

### 1. `mmdet3d/` - Core Framework Patches

#### `mmdet3d/__init__.py`
* **Upstream Problem:** The original code enforces a strict string assertion: `assert mmcv_version < '2.2.0'`. Prebuilt MMCV wheels built with custom CUDA tags (e.g., `2.1.0+cu118`) fail this string comparison, raising an `AssertionError: MMCV is used but incompatible` that halts execution upon import.
* **Modification:** Commented out the MMCV version assertion check (lines 21-24).
* **Technical Purpose:** Allows verified MMCV 2.1.0 builds to import and execute without triggering artificial assertion failures.

#### `mmdet3d/apis/inferencers/base_3d_inferencer.py`
* **Upstream Problem:** During JSON result serialization in `pred2dict()`, the bounding box tensor `bboxes_3d` resides in GPU memory. Direct conversion via `.numpy().tolist()` raises a PyTorch runtime error (`Tensor on device cuda:0 is not on cpu`). Additionally, default inferencers lack automated naming based on the source LiDAR sweep filename.
* **Modification:**
  * Added explicit `.cpu()` transfer: `'bboxes_3d': pred_instances_3d.bboxes_3d.tensor.cpu().tolist()`.
  * Implemented automated file naming: extracts the LiDAR basename and saves predictions to `osp.join(pred_out_dir, 'preds', lidar_path + '.json')`.
* **Technical Purpose:** Ensures memory-safe JSON prediction export and proper filename correlation for subsequent visualization pipelines.

#### `mmdet3d/evaluation/metrics/nuscenes_metric.py`
* **Upstream Problem:** The official nuScenes evaluation toolkit (`NuScenesEval`) asserts that predicted sample tokens match ground truth sample tokens: `assert set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens)`. This requires inference to run on the entire multi-gigabyte dataset. Running evaluation on partial datasets (e.g., nuScenes Mini 404 frames, or 10-frame demo samples) causes this assertion to crash the process.
* **Modification:** Dynamically monkeypatches `NuScenesEval.__init__` (lines 278-304) to load ground truth annotations exclusively for the sample tokens that were actually predicted (`pred_tokens`).
* **Technical Purpose:** Enables accurate, crash-free calculation of mean Average Precision (mAP) and nuScenes Detection Score (NDS) on partial datasets and demo subsets.

---

### 2. `projects/BEVFusion/` - CUDA Compilation Overrides

#### `projects/BEVFusion/setup.py`
* **Upstream Problem:** When compiling custom CUDA extensions (`bev_pool`, `voxel_layer`) using NVCC 11.8 on modern Linux distributions (such as Ubuntu 22.04 with GCC 11/12/14) or modern Windows MSVC compilers, NVCC halts with `fatal error: unsupported compiler`.
* **Modification:** Injected compiler compatibility flags and explicit GPU compute architectures into `extra_compile_args['nvcc']` (lines 22-32):
  * `-allow-unsupported-compiler`: Overrides NVCC compiler version checks.
  * `-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`: Prevents STL header mismatch errors on newer GCC toolchains.
  * `-gencode=arch=compute_75,code=sm_75`: Targets NVIDIA Turing architectures (specifically NVIDIA Tesla T4 on Google Colab and GTX 16-series).
  * `-gencode=arch=compute_70,code=sm_70`, `-gencode=arch=compute_80,code=sm_80`, `-gencode=arch=compute_86,code=sm_86`: Targets Volta, Ampere (A100), and RTX 30-series GPUs.
* **Technical Purpose:** Ensures flawless compilation of custom BEVFusion CUDA kernels across both legacy and modern host compilers.

---

### 3. `configs/` - BEVFusion Model Tuning

#### `configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py`
* **Upstream Problem:**
  1. Default config files specify remote storage paths (`backend='petrel'`, pointing to internal Shanghai AI Lab S3 servers), causing connection timeouts in external environments.
  2. Does not specify a local checkpoint path, attempting to redownload base weights on every invocation.
  3. Uses a learning rate suited for multi-GPU training from scratch rather than fine-tuning.
* **Modification:**
  * `load_from`: Explicitly set to `checkpoints/bevfusion_lidar/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933_fixed.pth`.
  * `backend_args = None`: Disables petrel S3 storage, ensuring data is read from local disk storage.
  * `input_modality`: Configured for pure LiDAR detection (`use_lidar=True, use_camera=False`).
  * `lr = 0.0001`: Tuned learning rate for stable transfer learning and fine-tuning without gradient divergence.
* **Technical Purpose:** Provides a self-contained, standalone configuration tailored for local inference and fine-tuning.

---

### 4. `tools/dataset_converters/` - Dataset Preprocessing Tools

#### `tools/dataset_converters/nuscenes_converter.py`
* **Function:** Ingests raw nuScenes annotations, point clouds, and camera calibrations via the NuScenes SDK.
* **Technical Purpose:** Synchronizes multi-sensor coordinate systems and aggregates 10 sweeps of temporal LiDAR data into consolidated binary `.pkl` files (`nuscenes_infos_train.pkl`, `nuscenes_infos_val.pkl`), removing the need to parse thousands of JSON files at runtime.

#### `tools/dataset_converters/update_infos_to_v2.py`
* **Function:** Migrates legacy MMDetection3D v0.x/v1.0 annotation structures to the OpenMMLab 2.0 / MMEngine standardized schema (`metainfo` and `data_list`).
* **Technical Purpose:** Eliminates schema incompatibility warnings and errors when loading older dataset info files into MMDetection3D 1.4+.

#### `tools/dataset_converters/create_gt_database.py`
* **Function:** Crops ground truth 3D point cloud objects from training scenes and saves them into a standalone instance database (`nuscenes_gt_database/`).
* **Technical Purpose:** Provides candidate 3D objects for training-time data augmentation (Copy-Paste / CutMix sampling), increasing object density in sparse scenes during fine-tuning.
