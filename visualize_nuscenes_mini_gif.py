"""
visualize_nuscenes_mini_gif.py - 360-Degree Surround Camera + BEV Animated GIF Generator

Purpose:
    Generates synchronized multi-panel frames and animated GIFs combining all 6 surround
    cameras with the top-down BEV 3D detection radar map across continuous nuScenes scenes.

Key Mechanics:
    1. Millisecond-precision timestamp matching: links each LiDAR sweep with its closest
       frame across all 6 cameras (CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
       CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT).
    2. Multi-panel dashboard layout: 2 rows x 5 columns GridSpec layout (6 surround cameras
       on the left, large unified BEV LiDAR detection panel on the right).
    3. Ego vehicle orientation: displays the ego car at origin (0, 0) pointing forward.
    4. GIF compilation: stitches multi-camera dashboard frames into an animated sequence
       at 400ms per frame (~2.5 Hz natural keyframe rate) using Pillow.
    5. Saves output to demo/outputs/nuscenes_mini_gif/nuscenes_360_surround_bev_animation.gif.
"""

import os
import json
import glob
import numpy as np
import cv2
from PIL import Image

# Non-GUI backend for Colab / headless Linux
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

CLASS_COLORS = {
    0: '#1f77b4',  # car - blue
    1: '#ff7f0e',  # truck - orange
    2: '#bcbd22',  # construction_vehicle - olive-yellow
    3: '#d62728',  # bus - red
    4: '#e377c2',  # trailer - pink
    5: '#8c564b',  # barrier - brown
    6: '#9467bd',  # motorcycle - purple
    7: '#17becf',  # bicycle - cyan
    8: '#2ca02c',  # pedestrian - green
    9: '#ffbb78'   # traffic_cone - light orange
}

def get_corners(x, y, w, l, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners = np.array([[w/2, l/2], [w/2, -l/2], [-w/2, -l/2], [-w/2, l/2]]).T
    rotated_corners = np.dot(R, corners)
    corners_x = rotated_corners[0, :] + x
    corners_y = rotated_corners[1, :] + y
    return corners_x, corners_y

def render_bev_ax(ax, points, bboxes, scores, labels, title="BEV 3D Detection", score_thr=0.3):
    ax.set_facecolor('#141414')
    pts_x = points[:, 0]
    pts_y = points[:, 1]
    pts_z = points[:, 2]
    mask = (pts_x > -54) & (pts_x < 54) & (pts_y > -54) & (pts_y < 54)
    
    sc = ax.scatter(pts_x[mask], pts_y[mask], c=pts_z[mask], s=0.7, cmap='viridis', alpha=0.45, vmin=-4, vmax=3)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label('Height Z (m)', color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Ego vehicle at (0, 0)
    ax.plot(0, 0, marker='^', markersize=14, color='red', label='Ego Vehicle')

    detected_classes = set()
    detection_count = 0
    for i, bbox in enumerate(bboxes):
        score = scores[i]
        label_idx = labels[i]
        if score < score_thr:
            continue
        detection_count += 1
        detected_classes.add(label_idx)
        color = CLASS_COLORS.get(label_idx, '#ffffff')
        
        x, y, z, dx, dy, dz, yaw = bbox[:7]
        cx, cy = get_corners(x, y, dx, dy, yaw)
        cx = np.append(cx, cx[0])
        cy = np.append(cy, cy[0])
        ax.plot(cx, cy, c=color, linewidth=2.0)
        
        class_name = CLASSES[label_idx] if label_idx < len(CLASSES) else str(label_idx)
        ax.text(x, y, f'{class_name}\n{score:.2f}', color='white', fontsize=7, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.65, edgecolor=color, boxstyle='round,pad=0.25'))

    legend_elements = [Line2D([0], [0], color='red', marker='^', lw=0, markersize=8, label='Ego Vehicle')]
    for idx in sorted(detected_classes):
        legend_elements.append(Line2D([0], [0], color=CLASS_COLORS[idx], lw=2, label=CLASSES[idx]))
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', facecolor='#222222', edgecolor='gray', labelcolor='white', fontsize=8)

    ax.set_title(f"{title} ({detection_count} detections >= {score_thr})", fontsize=13, fontweight='bold', color='white', pad=10)
    ax.set_xlabel("X (m)", color='white', fontsize=10)
    ax.set_ylabel("Y (m)", color='white', fontsize=10)
    ax.set_xlim(-54, 54)
    ax.set_ylim(-54, 54)
    ax.tick_params(colors='white')
    ax.grid(True, linestyle='--', alpha=0.25, color='gray')
    ax.set_aspect('equal')

def extract_timestamp(path_str):
    """Extracts integer timestamp from a nuScenes filename."""
    base = os.path.basename(path_str)
    parts = base.split("__")
    if len(parts) >= 3:
        ts_part = parts[-1].split(".")[0]
        try:
            return int(ts_part)
        except ValueError:
            pass
    return None

def find_closest_camera_file(cam_files, target_ts):
    """Finds the camera image with the closest timestamp to the LiDAR sweep."""
    if not cam_files:
        return None
    if target_ts is None:
        return cam_files[0]
    
    best_file = None
    min_diff = float('inf')
    for f in cam_files:
        ts = extract_timestamp(f)
        if ts is not None:
            diff = abs(ts - target_ts)
            if diff < min_diff:
                min_diff = diff
                best_file = f
    return best_file if best_file else cam_files[0]

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 75)
    print("nuScenes Mini: 360-Degree Surround 6 Cameras + BEV LiDAR GIF Generator")
    print("=" * 75)

    # 1. Locate dataset samples folder
    possible_roots = [
        os.path.join(base_dir, "data", "nuscenes", "samples"),
        os.path.join(base_dir, "data", "v1.0-mini", "samples"),
        os.path.join(base_dir, "data", "samples"),
        os.path.join(base_dir, "nuscenes", "samples"),
        "/content/data/nuscenes/samples",
        "/content/nuscenes/samples"
    ]
    samples_dir = next((p for p in possible_roots if os.path.exists(p) and os.path.exists(os.path.join(p, "LIDAR_TOP"))), None)

    if not samples_dir:
        print("Error: nuScenes samples directory not found!")
        print("Searched locations:\n" + "\n".join(possible_roots))
        return

    print(f"Using dataset from: {samples_dir}")

    demo_dir = os.path.join(base_dir, "demo")
    preds_dir = os.path.join(demo_dir, "outputs", "preds")
    out_dir = os.path.join(demo_dir, "outputs", "nuscenes_mini_gif")
    os.makedirs(out_dir, exist_ok=True)

    # 6 Cameras layout
    cam_defs = [
        ('CAM_FRONT_LEFT', 'Front Left', 0, 0),
        ('CAM_FRONT', 'Front', 0, 1),
        ('CAM_FRONT_RIGHT', 'Front Right', 0, 2),
        ('CAM_BACK_LEFT', 'Back Left', 1, 0),
        ('CAM_BACK', 'Back', 1, 1),
        ('CAM_BACK_RIGHT', 'Back Right', 1, 2),
    ]

    # Pre-index camera files for fast lookup
    cam_file_map = {}
    for cam_key, _, _, _ in cam_defs:
        folder = os.path.join(samples_dir, cam_key)
        cam_file_map[cam_key] = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        print(f"Loaded {len(cam_file_map[cam_key])} images for {cam_key}")

    # Use the 10 LiDAR files that BEVFusion has already processed
    lidar_files = sorted(glob.glob(os.path.join(demo_dir, "*.bin")))
    if not lidar_files:
        # Fallback to LIDAR_TOP
        lidar_files = sorted(glob.glob(os.path.join(samples_dir, "LIDAR_TOP", "*.bin")))[:10]

    print(f"\nProcessing {len(lidar_files)} LiDAR frames...")

    rendered_frames = []
    for idx, lidar_path in enumerate(lidar_files, start=1):
        filename = os.path.basename(lidar_path)
        lidar_ts = extract_timestamp(filename)

        # Load LiDAR point cloud
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)

        # Load BEVFusion predictions if available
        json_name = filename.replace('.bin', '.json')
        pred_json = os.path.join(preds_dir, json_name)
        bboxes, scores, labels = [], [], []
        if os.path.exists(pred_json):
            with open(pred_json, 'r') as f:
                data = json.load(f)
            bboxes = data.get('bboxes_3d', [])
            scores = data.get('scores_3d', [])
            labels = data.get('labels_3d', [])

        # Create Figure: 2 rows x 5 cols
        # Cols 0, 1, 2: 6 surround cameras
        # Cols 3, 4: BEV 3D detection map
        fig = plt.figure(figsize=(26, 11), facecolor='#0d0d0d')
        gs = gridspec.GridSpec(2, 5, figure=fig, width_ratios=[1.0, 1.0, 1.0, 1.45, 1.45], wspace=0.14, hspace=0.18)

        # Render 6 cameras with exact millisecond sync
        for cam_key, cam_title, r, c in cam_defs:
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor('#0d0d0d')
            
            matching_img = find_closest_camera_file(cam_file_map.get(cam_key, []), lidar_ts)
            if matching_img and os.path.exists(matching_img):
                img = cv2.imread(matching_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Image not found", color='gray', ha='center', va='center')
            
            ax.set_title(f"[{cam_title} Camera]", color='white', fontsize=11, fontweight='bold', pad=5)
            ax.axis('off')

        # Render BEV map on the right
        ax_bev = fig.add_subplot(gs[:, 3:])
        render_bev_ax(ax_bev, points, bboxes, scores, labels, 
                      title=f"Bird's-Eye View 3D LiDAR (Frame {idx:02d}/{len(lidar_files)})", score_thr=0.3)

        plt.suptitle(f"nuScenes Autonomous Driving: 360-Degree Surround Vision + BEV Detection [Frame {idx:02d}/{len(lidar_files)}]", 
                     fontsize=17, fontweight='bold', color='white', y=0.98)

        out_frame_path = os.path.join(out_dir, f"frame_{idx:02d}.png")
        plt.savefig(out_frame_path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()
        rendered_frames.append(out_frame_path)
        print(f"  -> Generated Dashboard Frame {idx:02d}/{len(lidar_files)}: {os.path.basename(out_frame_path)}")

    # Generate Animated GIF
    if len(rendered_frames) > 1:
        gif_out_path = os.path.join(out_dir, "nuscenes_360_surround_bev_animation.gif")
        frames = [Image.open(p) for p in rendered_frames]
        frames[0].save(
            gif_out_path,
            save_all=True,
            append_images=frames[1:],
            duration=400,  # 400ms per frame
            loop=0
        )
        print("=" * 75)
        print(f"SUCCESS! Animated 360-Degree Surround + BEV GIF created at:\n  -> {gif_out_path}")
        print("=" * 75)

if __name__ == "__main__":
    main()
