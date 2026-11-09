# Agent Instructions

## Overview
Repository dedicated to 3D LiDAR Object Detection using BEVFusion on the nuScenes autonomous driving benchmark.

## Core Directives
- Environment: Python 3.10 with PyTorch 2.1.2 (CUDA 11.8), MMCV 2.1.0, MMDetection3D 1.4.0, and NumPy 1.26.4.
- Demo Assets: Store demo images, gifs, and video showcases in assets/. Do not commit notebooks/ until reviewed.
- Upstream Patches: Apply patches from custom_changes/ to external/mmdetection3d/ before building. Refer to custom_changes/README.md for details.
- Checkpoints: Transpose pretrained weights from spconv 1.x to spconv 2.x via fix_bevfusion_checkpoint.py before inference.
- Coordinate System: nuScenes RFU standard (X: Right, Y: Forward, Z: Up). Top-down BEV maps must project forward Y upward (v = cy - Y*scale).

## Key Commands
- Quick Demo: python setup_demo.py (10-frame inference to demo/outputs/preds/).
- BEV Plot: python visualize_demo.py (renders top-down BEV plots).
- Batch Inference: python run_inference_all.py (runs GPU inference on all 404 nuScenes mini frames).
- Full Video: python create_full_nuscenes_video.py (renders 1080p synchronized 6-camera + BEV video).
- Surround GIF: python visualize_nuscenes_mini_gif.py (creates 360-degree multi-camera + BEV GIF).
