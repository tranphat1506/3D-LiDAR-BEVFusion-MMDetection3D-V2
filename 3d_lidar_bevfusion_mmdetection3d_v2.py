#!/usr/bin/env python3
"""
3d_lidar_bevfusion_mmdetection3d_v2.py

Local-machine 3D LiDAR inference entry point for BEVFusion on nuScenes.

This script is the local replacement for the original Colab notebook export. It
contains no Colab-only constructs (`!pip`, `condacolab`, `google.colab`,
`/content/...`, or IPython display calls). All dependencies are expected to be
installed once in the active Python environment.

What it does
------------
1. Resolves project paths relative to this file, so it works on Linux, Windows,
   and macOS without hardcoded absolute paths.
2. Verifies that PyTorch, MMDetection3D, the BEVFusion config, the checkpoint,
   and the requested input files are available before loading the model.
3. Loads the pure-LiDAR BEVFusion model through `LidarDet3DInferencer`.
4. Runs inference on a single point cloud or a directory of point clouds.
5. Writes one JSON file per frame to `<out-dir>/preds/<frame>.json` using the
   same schema consumed by `visualize_demo.py`:
       bboxes_3d, scores_3d, labels_3d
6. Optionally renders a top-down BEV PNG to `<out-dir>/vis/<frame>.png`.

Prerequisites
-------------
- Python 3.10 environment with torch 2.1.2+cu118, mmcv 2.1.0, mmengine, and
  mmdet3d 1.4.0 (see README.md).
- `external/mmdetection3d` cloned and patched from `custom_changes/`.
- A fixed checkpoint produced by `fix_bevfusion_checkpoint.py`.

Usage
-----
Run the 10 bundled demo frames:
    python 3d_lidar_bevfusion_mmdetection3d_v2.py --input demo --limit 10 --save-vis

Run a single LiDAR sweep:
    python 3d_lidar_bevfusion_mmdetection3d_v2.py --input path/to/frame.bin

Run a nuScenes Mini directory:
    python 3d_lidar_bevfusion_mmdetection3d_v2.py \
        --input data/nuscenes/samples/LIDAR_TOP --recursive --save-vis

Force CPU:
    python 3d_lidar_bevfusion_mmdetection3d_v2.py --input demo --device cpu

Output
------
<out-dir>/preds/<frame>.json   Prediction boxes, scores, and class indices.
<out-dir>/vis/<frame>.png      Optional BEV rendering when --save-vis is set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Project layout and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
EXTERNAL_MMDET3D = PROJECT_ROOT / "external" / "mmdetection3d"
BEVFUSION_PROJECT = EXTERNAL_MMDET3D / "projects" / "BEVFusion"

CONFIG_CANDIDATES = (
    PROJECT_ROOT
    / "custom_changes"
    / "configs"
    / "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
    BEVFUSION_PROJECT
    / "configs"
    / "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
)

CHECKPOINT_CANDIDATES = (
    PROJECT_ROOT
    / "checkpoints"
    / "bevfusion_lidar"
    / "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933_fixed.pth",
    PROJECT_ROOT
    / "checkpoints"
    / "bevfusion_lidar"
    / "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth",
)

INPUT_CANDIDATES = (
    PROJECT_ROOT / "demo",
    PROJECT_ROOT / "data" / "nuscenes_demo" / "lidar",
    PROJECT_ROOT / "data" / "nuscenes" / "samples" / "LIDAR_TOP",
    PROJECT_ROOT / "data" / "v1.0-mini" / "samples" / "LIDAR_TOP",
)

DEFAULT_OUT_DIR = PROJECT_ROOT / "demo" / "outputs"
SUPPORTED_EXTENSIONS = (".bin", ".pcd", ".ply", ".npy")

CLASSES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)

CLASS_COLORS = (
    "tab:blue",
    "tab:orange",
    "gold",
    "tab:red",
    "darkorange",
    "sienna",
    "tab:purple",
    "tab:pink",
    "tab:olive",
    "tab:cyan",
)


# ---------------------------------------------------------------------------
# Path and dependency helpers
# ---------------------------------------------------------------------------


def register_local_paths() -> None:
    """Prefer the local patched MMDetection3D checkout over site-packages."""
    for path in (PROJECT_ROOT, EXTERNAL_MMDET3D, BEVFUSION_PROJECT):
        text = str(path)
        if path.is_dir() and text not in sys.path:
            sys.path.insert(0, text)


def first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    """Return the first path in the iterable that exists."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_config(explicit: Optional[str]) -> Path:
    """Resolve the BEVFusion config path."""
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        return path

    found = first_existing(CONFIG_CANDIDATES)
    if found is None:
        raise FileNotFoundError(
            "BEVFusion config not found. Expected one of:\n  "
            + "\n  ".join(str(p) for p in CONFIG_CANDIDATES)
        )
    return found


def resolve_checkpoint(explicit: Optional[str]) -> Path:
    """Resolve the fixed BEVFusion checkpoint path."""
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        return path

    found = first_existing(CHECKPOINT_CANDIDATES)
    if found is None:
        raise FileNotFoundError(
            "BEVFusion checkpoint not found. Download and fix it first:\n"
            "  mkdir -p checkpoints/bevfusion_lidar\n"
            "  wget -O checkpoints/bevfusion_lidar/"
            "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth \\\n"
            "    https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/"
            "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth\n"
            "  python fix_bevfusion_checkpoint.py \\\n"
            "    --input checkpoints/bevfusion_lidar/"
            "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth \\\n"
            "    --output checkpoints/bevfusion_lidar/"
            "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933_fixed.pth"
        )
    return found


def collect_inputs(
    input_arg: Optional[str],
    recursive: bool,
    limit: int,
) -> List[Path]:
    """Collect LiDAR point cloud files from a file or directory."""
    if input_arg:
        root = Path(input_arg).expanduser().resolve()
    else:
        found = first_existing(INPUT_CANDIDATES)
        if found is None:
            raise FileNotFoundError(
                "No input directory found. Pass --input explicitly."
            )
        root = found

    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    if root.is_file():
        files = [root]
    else:
        pattern = "**/*" if recursive else "*"
        files = sorted(
            path
            for path in root.glob(pattern)
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    if not files:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise FileNotFoundError(
            f"No point cloud files ({supported}) found under: {root}"
        )

    if limit > 0:
        files = files[:limit]
    return files


def import_runtime() -> Tuple[Any, Any, Any, Any]:
    """Import torch, numpy, and the MMDetection3D inference API."""
    try:
        import numpy as np
        import torch
        from mmdet3d.apis import LidarDet3DInferencer
        from mmengine.config import Config
    except ImportError as exc:
        raise ImportError(
            "Failed to import inference dependencies. Activate the project "
            "Python 3.10 environment and install the OpenMMLab stack.\n"
            "Expected: torch 2.1.2+cu118, mmcv 2.1.0, mmengine, mmdet3d 1.4.0.\n"
            "See README.md, then run:\n"
            "  cd external/mmdetection3d && python setup.py develop\n"
            f"\nOriginal error: {exc}"
        ) from exc
    return np, torch, LidarDet3DInferencer, Config


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def tensor_to_list(value: Any) -> List[Any]:
    """Convert a tensor, array, or sequence into a plain Python list."""
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def extract_predictions(data_sample: Any) -> Dict[str, List[Any]]:
    """Extract 3D boxes, scores, and labels from a predicted DataSample."""
    instances = data_sample.pred_instances_3d
    return {
        "bboxes_3d": tensor_to_list(instances.bboxes_3d.tensor),
        "scores_3d": tensor_to_list(instances.scores_3d),
        "labels_3d": tensor_to_list(instances.labels_3d),
    }


def filter_predictions(
    prediction: Dict[str, List[Any]],
    score_threshold: float,
    np: Any,
) -> Dict[str, List[Any]]:
    """Apply a score threshold while preserving the JSON schema."""
    boxes = prediction.get("bboxes_3d") or []
    scores = prediction.get("scores_3d") or []
    labels = prediction.get("labels_3d") or []

    if not scores:
        return {"bboxes_3d": [], "scores_3d": [], "labels_3d": []}

    score_array = np.asarray(scores, dtype=np.float32)
    keep = score_array >= float(score_threshold)

    def pick(values: Sequence[Any]) -> List[Any]:
        array = np.asarray(values)
        return array[keep].tolist()

    return {
        "bboxes_3d": pick(boxes),
        "scores_3d": pick(scores),
        "labels_3d": pick(labels),
    }


# ---------------------------------------------------------------------------
# Inference backends
# ---------------------------------------------------------------------------


def infer_with_inferencer(
    inferencer: Any,
    point_path: Path,
    out_dir: Path,
    score_threshold: float,
    torch: Any,
) -> Dict[str, List[Any]]:
    """Run one frame through the public inferencer API."""
    result = inferencer(
        inputs={"points": str(point_path)},
        out_dir=str(out_dir),
        show=False,
        no_save_vis=True,
        no_save_pred=True,
        pred_score_thr=score_threshold,
    )
    predictions = result.get("predictions") or []
    if not predictions:
        raise RuntimeError("Inferencer returned no predictions")
    return predictions[0]


def infer_with_model(
    inferencer: Any,
    point_path: Path,
    score_threshold: float,
    torch: Any,
) -> Dict[str, List[Any]]:
    """Fallback that calls model.test_step directly.

    Some MMDetection3D builds raise postprocessing errors around `box_mode_3d`.
    This path bypasses `pred2dict` and extracts `pred_instances_3d` directly.
    """
    inputs = {"points": str(point_path)}
    batches = inferencer.preprocess([inputs], batch_size=1)
    for batch in batches:
        with torch.no_grad():
            predictions = inferencer.model.test_step(batch)
        if predictions and hasattr(predictions[0], "pred_instances_3d"):
            return extract_predictions(predictions[0])
    raise RuntimeError("Direct model inference returned no predictions")


def run_single_frame(
    inferencer: Any,
    point_path: Path,
    out_dir: Path,
    score_threshold: float,
    np: Any,
    torch: Any,
) -> Dict[str, Any]:
    """Infer one point cloud and persist a normalized JSON result."""
    try:
        prediction = infer_with_inferencer(
            inferencer, point_path, out_dir, score_threshold, torch
        )
    except Exception as exc:  # noqa: BLE001 - fallback must catch backend errors
        print(f"  [warn] Inferencer call failed ({exc}); using model.test_step")
        prediction = infer_with_model(inferencer, point_path, score_threshold, torch)

    filtered = filter_predictions(prediction, score_threshold, np)

    pred_dir = out_dir / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    json_path = pred_dir / f"{point_path.stem}.json"

    record = {
        "source": str(point_path),
        "score_threshold": float(score_threshold),
        "num_detections": len(filtered["scores_3d"]),
        **filtered,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)

    return record


# ---------------------------------------------------------------------------
# Optional BEV visualization
# ---------------------------------------------------------------------------


def load_points(point_path: Path, np: Any) -> Any:
    """Load a point cloud with automatic channel detection."""
    suffix = point_path.suffix.lower()

    if suffix == ".bin":
        raw = np.fromfile(str(point_path), dtype=np.float32)
        if raw.size % 5 == 0:
            points = raw.reshape(-1, 5)
        elif raw.size % 4 == 0:
            points = raw.reshape(-1, 4)
        elif raw.size % 3 == 0:
            points = raw.reshape(-1, 3)
        else:
            raise ValueError(f"Unsupported binary layout: {point_path}")
        return points

    if suffix == ".npy":
        return np.load(str(point_path))

    if suffix in (".pcd", ".ply"):
        try:
            import open3d as o3d

            cloud = o3d.io.read_point_cloud(str(point_path))
            return np.asarray(cloud.points)
        except ImportError as exc:
            raise ImportError(
                "Open3D is required to read .pcd/.ply files. "
                "Install it with `pip install open3d`."
            ) from exc

    raise ValueError(f"Unsupported point cloud format: {point_path}")


def render_bev(
    point_path: Path,
    record: Dict[str, Any],
    out_dir: Path,
    np: Any,
) -> None:
    """Render a top-down BEV image with oriented boxes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = load_points(point_path, np)
    if points.shape[1] < 3:
        raise ValueError(f"Point cloud has too few channels: {points.shape}")

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    mask = (x > -54) & (x < 54) & (y > -54) & (y < 54)

    fig, ax = plt.subplots(figsize=(12, 12), facecolor="white")
    ax.set_facecolor("white")
    scatter = ax.scatter(
        x[mask], y[mask], c=z[mask], s=0.5, cmap="viridis", alpha=0.35,
        vmin=-5, vmax=3,
    )
    colorbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Height (Z)")

    for box, score, label in zip(
        record["bboxes_3d"], record["scores_3d"], record["labels_3d"]
    ):
        box_x, box_y, _, dx, dy, _, yaw = box[:7]
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        corners = np.array(
            [
                [dx / 2, dy / 2],
                [dx / 2, -dy / 2],
                [-dx / 2, -dy / 2],
                [-dx / 2, dy / 2],
            ]
        ).T
        rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        rotated = rotation @ corners
        corner_x = rotated[0] + box_x
        corner_y = rotated[1] + box_y
        corner_x = np.append(corner_x, corner_x[0])
        corner_y = np.append(corner_y, corner_y[0])

        label_index = int(label)
        color = CLASS_COLORS[label_index] if 0 <= label_index < len(CLASS_COLORS) else "black"
        class_name = CLASSES[label_index] if 0 <= label_index < len(CLASSES) else str(label)
        ax.plot(corner_x, corner_y, color=color, linewidth=1.5)
        ax.text(
            box_x, box_y, f"{class_name}\n{score:.2f}", color=color, fontsize=6,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.5),
        )

    ax.set_title(
        f"{point_path.name} - {record['num_detections']} detections "
        f"(score >= {record['score_threshold']:.2f})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(-54, 54)
    ax.set_ylim(-54, 54)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)

    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(vis_dir / f"{point_path.stem}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local BEVFusion 3D LiDAR inference for nuScenes point clouds."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Point cloud file or directory. Defaults to the first available "
        "demo or nuScenes sample directory.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the BEVFusion config. Defaults to custom_changes/configs.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to the fixed .pth checkpoint. Defaults to checkpoints/bevfusion_lidar.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cuda, cuda:0, or cpu (default: auto).",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.3,
        help="Minimum detection score kept in the output JSON (default: 0.3).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of frames to process; 0 means all (default: 0).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories when --input is a directory.",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Render a top-down BEV PNG for every processed frame.",
    )
    return parser.parse_args(argv)


def resolve_device(requested: str, torch: Any) -> str:
    """Resolve the device string and fall back to CPU when CUDA is unavailable."""
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return requested


def print_environment(torch: Any, device: str) -> None:
    """Print a short runtime summary."""
    print("=" * 68)
    print("BEVFusion local inference")
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA:    {torch.version.cuda if torch.version.cuda else 'N/A'}")
    print(f"  Device:  {device}")
    if device.startswith("cuda") and torch.cuda.is_available():
        print(f"  GPU:     {torch.cuda.get_device_name(0)}")
    print("=" * 68)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    np, torch, LidarDet3DInferencer, Config = import_runtime()

    config_path = resolve_config(args.config)
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    input_files = collect_inputs(args.input, args.recursive, args.limit)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device, torch)
    print_environment(torch, device)
    print(f"  Config:      {config_path}")
    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  Input files: {len(input_files)}")
    print(f"  Output:      {out_dir}")
    print("=" * 68)

    config = Config.fromfile(str(config_path))
    inferencer = LidarDet3DInferencer(
        model=config,
        weights=str(checkpoint_path),
        device=device,
    )

    failures = 0
    for index, point_path in enumerate(input_files, start=1):
        print(f"[{index}/{len(input_files)}] {point_path.name}")
        try:
            record = run_single_frame(
                inferencer,
                point_path,
                out_dir,
                args.score_thr,
                np,
                torch,
            )
            print(f"  detections: {record['num_detections']}")
            if args.save_vis:
                render_bev(point_path, record, out_dir, np)
                print("  BEV image:  saved")
        except Exception as exc:  # noqa: BLE001 - keep batch inference resilient
            failures += 1
            print(f"  [error] {type(exc).__name__}: {exc}")

    print("=" * 68)
    print(
        f"Finished: {len(input_files) - failures}/{len(input_files)} frames "
        f"succeeded, {failures} failed."
    )
    print(f"Predictions: {out_dir / 'preds'}")
    if args.save_vis:
        print(f"Visualizations: {out_dir / 'vis'}")
    print("=" * 68)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
