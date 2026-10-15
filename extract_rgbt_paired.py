# extract_rgbt_paired.py
import os
import json
import numpy as np

# Native sensor resolutions in Anti-UAV-RGBT
RGB_W, RGB_H = 1920.0, 1080.0
IR_W, IR_H = 640.0, 512.0


def parse_modality_json(json_data, width, height):
    """Parses normalized centroids, apparent scale (depth proxy), and visibility mask."""
    exist_list = json_data.get("exist", [])
    gt_rects = json_data.get("gt_rect", [])
    n_frames = min(len(exist_list), len(gt_rects))

    centers = np.zeros((n_frames, 2), dtype=np.float32)
    scales = np.zeros(n_frames, dtype=np.float32)
    valid_mask = np.zeros(n_frames, dtype=np.bool_)

    last_center = np.array([0.5, 0.5], dtype=np.float32)
    last_scale = 0.025

    for t in range(n_frames):
        ex = exist_list[t]
        rect = gt_rects[t]
        # Safely handle empty lists [] or [0,0,0,0] when target is occluded
        if ex == 1 and isinstance(rect, (list, tuple)) and len(rect) == 4 and rect[2] > 1 and rect[3] > 1:
            cx = (rect[0] + 0.5 * rect[2]) / width
            cy = (rect[1] + 0.5 * rect[3]) / height
            scale = np.sqrt((rect[2] / width) * (rect[3] / height))
            last_center = np.array([cx, cy], dtype=np.float32)
            last_scale = max(float(scale), 0.015)
            valid_mask[t] = True

        centers[t] = last_center
        scales[t] = last_scale

    return centers, scales, valid_mask


def compute_scale_adjusted_velocities(centers, scales, alpha=0.30, max_speed=2.5):
    """Rejects camera-panning jumps, adjusts by target depth scale, and applies EMA smoothing."""
    raw_deltas = np.diff(centers, axis=0)
    step_scales = scales[1:, None]

    # Reject instantaneous camera-recentering slews (> 8% screen jump in 1 frame)
    jump_mask = np.linalg.norm(raw_deltas, axis=1) > 0.08
    raw_deltas[jump_mask] = 0.0

    # Depth-scale adjustment: smaller apparent bounding box = larger physical step per pixel
    depth_adjusted = raw_deltas / np.maximum(step_scales, 0.015)

    # Exponential Moving Average (EMA) filter
    smoothed = np.zeros_like(depth_adjusted)
    if len(depth_adjusted) > 0:
        smoothed[0] = depth_adjusted[0]
        for t in range(1, len(depth_adjusted)):
            smoothed[t] = alpha * depth_adjusted[t] + (1.0 - alpha) * smoothed[t - 1]

    std = np.std(smoothed) + 1e-6
    velocities = np.clip((smoothed / std) * (0.55 * max_speed), -max_speed, max_speed)
    return velocities.astype(np.float32)


def extract_unpacked_rgbt(root_dir=".", out_dir="data/processed_rgbt", label_new_dir="label_new"):
    os.makedirs(out_dir, exist_ok=True)
    splits = ["train", "val", "test"]

    datasets = {"train": [], "val": [], "test": []}
    all_rgb_drops, all_ir_drops, all_fused_drops = [], [], []

    for split in splits:
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            print(f"[WARNING] Directory for split '{split}' not found at {split_dir}!")
            continue

        # Load challenge attributes from label_new/{split}.json if present
        attr_file = os.path.join(root_dir, label_new_dir, f"{split}.json")
        split_attributes = {}
        if os.path.exists(attr_file):
            with open(attr_file, "r") as f:
                split_attributes = json.load(f)
            print(f"[INFO] Loaded challenge attributes for {split}: {len(split_attributes)} sequences.")

        seq_dirs = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])
        print(f"[INFO] Found {len(seq_dirs)} sequences in '{split}' split.")

        for seq_name in seq_dirs:
            seq_path = os.path.join(split_dir, seq_name)
            vis_json_path = os.path.join(seq_path, "visible.json")
            ir_json_path = os.path.join(seq_path, "infrared.json")

            if not os.path.exists(vis_json_path) or not os.path.exists(ir_json_path):
                print(f"[SKIP] Missing visible.json or infrared.json in {seq_name}")
                continue

            try:
                with open(vis_json_path, "r") as f:
                    vis_data = json.load(f)
                with open(ir_json_path, "r") as f:
                    ir_data = json.load(f)

                rgb_c, rgb_s, rgb_v = parse_modality_json(vis_data, RGB_W, RGB_H)
                ir_c, ir_s, ir_v = parse_modality_json(ir_data, IR_W, IR_H)

                L = min(len(rgb_c), len(ir_c))
                if L <= 60:
                    print(f"[SKIP] Sequence {seq_name} horizon too short ({L} <= 60 frames)")
                    continue

                rgb_c, rgb_s, rgb_v = rgb_c[:L], rgb_s[:L], rgb_v[:L]
                ir_c, ir_s, ir_v = ir_c[:L], ir_s[:L], ir_v[:L]

                # Dual-Modality Centroid & Scale Fusion:
                # - Both visible: 0.65 RGB + 0.35 IR
                # - Only RGB visible: 1.0 RGB
                # - Only IR visible: 1.0 IR
                # - Neither visible: hold previous fused centroid/scale
                fused_c = np.zeros((L, 2), dtype=np.float32)
                fused_s = np.zeros(L, dtype=np.float32)
                last_fc = np.array([0.5, 0.5], dtype=np.float32)
                last_fs = 0.025

                for t in range(L):
                    v_rgb = rgb_v[t]
                    v_ir = ir_v[t]
                    if v_rgb and v_ir:
                        cur_c = 0.65 * rgb_c[t] + 0.35 * ir_c[t]
                        cur_s = 0.65 * rgb_s[t] + 0.35 * ir_s[t]
                    elif v_rgb:
                        cur_c = rgb_c[t]
                        cur_s = rgb_s[t]
                    elif v_ir:
                        cur_c = ir_c[t]
                        cur_s = ir_s[t]
                    else:
                        cur_c = last_fc
                        cur_s = last_fs

                    fused_c[t] = cur_c
                    fused_s[t] = cur_s
                    last_fc = cur_c
                    last_fs = cur_s

                all_rgb_drops.append(1.0 - float(np.mean(rgb_v)))
                all_ir_drops.append(1.0 - float(np.mean(ir_v)))
                all_fused_drops.append(1.0 - float(np.mean(rgb_v | ir_v)))

                vel_seq = compute_scale_adjusted_velocities(fused_c, fused_s)
                tags = split_attributes.get(seq_name, [])

                datasets[split].append({
                    "seq_id": seq_name,
                    "velocities": vel_seq,
                    "rgb_visible": rgb_v[1:].astype(np.float32),
                    "ir_visible": ir_v[1:].astype(np.float32),
                    "attributes": tags
                })

            except Exception as e:
                print(f"[ERROR] Failed processing {seq_name}: {e}")

        out_file = os.path.join(out_dir, f"{split}_rgbt_trajectories.npy")
        np.save(out_file, np.array(datasets[split], dtype=object), allow_pickle=True)
        print(f"[SAVED] {split.upper():5s} Split: {len(datasets[split]):3d} sequences -> {out_file}\n")

    if all_fused_drops:
        print("=== Anti-UAV-RGBT Empirical Sensor Fusion Report ===")
        print(f"RGB Camera (1920x1080) Dropout Rate : {np.mean(all_rgb_drops):.2%}")
        print(f"Thermal IR (640x512) Dropout Rate   : {np.mean(all_ir_drops):.2%}")
        print(f"Dual RGB-T Fused Dropout Rate       : {np.mean(all_fused_drops):.2%}")

    # Verification of loaded record from train
    train_file = os.path.join(out_dir, "train_rgbt_trajectories.npy")
    if os.path.exists(train_file):
        loaded_train = np.load(train_file, allow_pickle=True)
        print(f"\n=== Verification of {train_file} ===")
        print(f"Total train sequences: {len(loaded_train)}")
        if len(loaded_train) > 0:
            sample = loaded_train[0]
            print("Sample record keys:", list(sample.keys()))
            print(f"seq_id: {sample['seq_id']}")
            print(f"attributes: {sample['attributes']}")
            print(f"velocities shape: {sample['velocities'].shape}, dtype: {sample['velocities'].dtype}")
            print(f"rgb_visible shape: {sample['rgb_visible'].shape}, dtype: {sample['rgb_visible'].dtype}")
            print(f"ir_visible shape: {sample['ir_visible'].shape}, dtype: {sample['ir_visible'].dtype}")
            print(f"velocities min/max: [{sample['velocities'].min():.3f}, {sample['velocities'].max():.3f}]")
            print(f"First 3 velocity steps:\n{sample['velocities'][:3]}")


if __name__ == "__main__":
    extract_unpacked_rgbt()
