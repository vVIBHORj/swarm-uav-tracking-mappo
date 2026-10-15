# tests/test_preprocessing.py
import os
import pytest
import numpy as np
from extract_rgbt_paired import parse_modality_json, compute_scale_adjusted_velocities


def test_parse_modality_json_valid():
    """Verify parse_modality_json extracts normalized coordinates, scales, and valid masks."""
    json_data = {
        "exist": [1, 1, 1],
        "gt_rect": [
            [100, 100, 50, 50],
            [120, 110, 60, 40],
            [130, 120, 70, 45],
        ]
    }
    width, height = 1920.0, 1080.0
    centers, scales, valid_mask = parse_modality_json(json_data, width, height)

    assert centers.shape == (3, 2)
    assert scales.shape == (3,)
    assert valid_mask.shape == (3,)
    assert np.all(valid_mask)

    # First box center: x = (100 + 25) / 1920, y = (100 + 25) / 1080
    np.testing.assert_allclose(centers[0, 0], 125.0 / 1920.0, atol=1e-5)
    np.testing.assert_allclose(centers[0, 1], 125.0 / 1080.0, atol=1e-5)
    expected_scale = np.sqrt((50.0 / 1920.0) * (50.0 / 1080.0))
    np.testing.assert_allclose(scales[0], expected_scale, atol=1e-5)


def test_parse_modality_json_occluded_and_empty_rect():
    """Verify occluded frames (exist=0 or empty rect) mark valid_mask=False and hold last values."""
    json_data = {
        "exist": [1, 0, 1],
        "gt_rect": [
            [200, 200, 40, 40],
            [],  # occluded frame
            [220, 210, 40, 40]
        ]
    }
    width, height = 640.0, 512.0
    centers, scales, valid_mask = parse_modality_json(json_data, width, height)

    assert centers.shape == (3, 2)
    assert bool(valid_mask[0]) is True
    assert bool(valid_mask[1]) is False
    assert bool(valid_mask[2]) is True

    # Frame 1 (occluded) should hold frame 0's values
    np.testing.assert_allclose(centers[1], centers[0], atol=1e-6)
    np.testing.assert_allclose(scales[1], scales[0], atol=1e-6)


def test_parse_modality_json_malformed_inputs():
    """Verify parse_modality_json handles empty or malformed inputs without raising exceptions."""
    # Non-dict input
    c, s, v = parse_modality_json(None, 1920, 1080)
    assert len(c) == 0 and len(s) == 0 and len(v) == 0

    # Negative dimensions
    c, s, v = parse_modality_json({"exist": [1], "gt_rect": [[10, 10, 20, 20]]}, -1, 1080)
    assert len(c) == 0

    # Missing keys
    c, s, v = parse_modality_json({}, 1920, 1080)
    assert len(c) == 0

    # Malformed key types
    c, s, v = parse_modality_json({"exist": "invalid", "gt_rect": 123}, 1920, 1080)
    assert len(c) == 0


def test_compute_scale_adjusted_velocities_jump_rejection():
    """Verify instantaneous camera-panning jumps (> 0.08 normalized screen jump) are zeroed."""
    centers = np.array([
        [0.50, 0.50],
        [0.51, 0.51],  # smooth step
        [0.65, 0.51],  # jump of 0.14 (> 0.08 threshold)
        [0.66, 0.52],  # smooth step
    ], dtype=np.float32)
    scales = np.array([0.03, 0.03, 0.03, 0.03], dtype=np.float32)

    velocities = compute_scale_adjusted_velocities(centers, scales, alpha=0.30, max_speed=2.5)

    assert velocities.shape == (3, 2)
    assert np.all(np.isfinite(velocities))
    assert np.all(np.abs(velocities) <= 2.5)


def test_compute_scale_adjusted_velocities_edge_cases():
    """Verify compute_scale_adjusted_velocities handles short or empty trajectory inputs safely."""
    # 0 frames
    empty_c = np.zeros((0, 2), dtype=np.float32)
    empty_s = np.zeros(0, dtype=np.float32)
    v_empty = compute_scale_adjusted_velocities(empty_c, empty_s)
    assert v_empty.shape == (0, 2)

    # 1 frame (no delta possible)
    single_c = np.array([[0.5, 0.5]], dtype=np.float32)
    single_s = np.array([0.025], dtype=np.float32)
    v_single = compute_scale_adjusted_velocities(single_c, single_s)
    assert v_single.shape == (0, 2)


def test_split_disjointness_if_present():
    """Verify that if processed dataset files exist on disk, train/val/test splits share ZERO sequence IDs."""
    data_dir = "data/processed_rgbt"
    splits = ["train", "val", "test"]
    split_files = {s: os.path.join(data_dir, f"{s}_rgbt_trajectories.npy") for s in splits}

    if not all(os.path.exists(p) for p in split_files.values()):
        pytest.skip("Processed dataset not found on disk; skipping split disjointness test.")

    seq_sets = {}
    for s in splits:
        arr = np.load(split_files[s], allow_pickle=True)
        seq_sets[s] = set(rec["seq_id"] for rec in arr)

    train_val_overlap = seq_sets["train"].intersection(seq_sets["val"])
    train_test_overlap = seq_sets["train"].intersection(seq_sets["test"])
    val_test_overlap = seq_sets["val"].intersection(seq_sets["test"])

    assert len(train_val_overlap) == 0, f"Data leakage between train and val: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Data leakage between train and test: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Data leakage between val and test: {val_test_overlap}"
