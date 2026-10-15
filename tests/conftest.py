import os
import sys
import pytest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.uav_rgbt_env import UAVRGBTTrackingEnv


@pytest.fixture
def synthetic_data_dir(tmp_path):
    """Creates a fast, self-contained mock Anti-UAV-RGBT trajectory dataset in tmp_path."""
    data_dir = tmp_path / "data" / "processed_rgbt"
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(12345)
    splits = ["train", "val", "test"]

    for split in splits:
        records = []
        n_seqs = 6 if split == "train" else 4
        for s in range(n_seqs):
            seq_len = 120
            # Continuous smoothed velocity within [-2.5, 2.5]
            raw_v = rng.normal(0.0, 0.5, size=(seq_len, 2)).astype(np.float32)
            vel = np.clip(raw_v, -2.5, 2.5)

            # Random visibility flags with some dropouts
            rgb_vis = (rng.uniform(0.0, 1.0, size=seq_len) > 0.2).astype(np.float32)
            ir_vis = (rng.uniform(0.0, 1.0, size=seq_len) > 0.15).astype(np.float32)

            records.append({
                "seq_id": f"{split}_seq_{s:03d}",
                "velocities": vel,
                "rgb_visible": rgb_vis,
                "ir_visible": ir_vis,
                "attributes": ["FM", "SV"] if s % 2 == 0 else ["TC-HARD", "LI"]
            })

        out_path = data_dir / f"{split}_rgbt_trajectories.npy"
        np.save(str(out_path), np.array(records, dtype=object), allow_pickle=True)

    return str(data_dir)


@pytest.fixture
def mock_env(synthetic_data_dir):
    """Provides an isolated UAVRGBTTrackingEnv instance backed by synthetic data."""
    return UAVRGBTTrackingEnv(
        num_drones=3,
        num_targets=4,
        split="val",
        data_dir=synthetic_data_dir,
        render_mode=None
    )
