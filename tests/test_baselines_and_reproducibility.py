# tests/test_baselines_and_reproducibility.py
import os
import json
import numpy as np
import pytest
import torch

from evaluate_visualize import (
    random_policy,
    greedy_nearest_policy,
    collision_aware_greedy_policy,
    target_assignment_heuristic_policy,
    run_policy_evaluation,
)
from train import train_mappo_rgbt


class DummyEnvNoState:
    """Mock environment that intentionally has NO target_pos or drone_pos attributes.
    
    Used to mathematically prove observation-only baseline execution without oracle leakage.
    """
    def __init__(self, agents=("drone_0", "drone_1", "drone_2")):
        self.agents = list(agents)
        self.num_drones = len(self.agents)


def test_baselines_strict_observation_only_no_oracle_leakage():
    """Verify all 4 heuristic baselines execute purely from obs_dict without accessing env coordinates."""
    dummy_env = DummyEnvNoState()

    # Create dummy observation dict: 3 drones, obs_dim=39
    obs_dict = {}
    for i, a in enumerate(dummy_env.agents):
        obs = np.zeros(39, dtype=np.float32)
        obs[0:2] = [0.2 + 0.1 * i, 0.5]  # own position
        obs[2:4] = [0.0, 0.0]            # own velocity
        # Target 0 visible at rel pos [0.1, 0.1]
        obs[12:14] = [0.1, 0.1]
        obs[16:18] = [1.0, 1.0]          # rgb, ir flags
        obs_dict[a] = obs

    # 1. Random policy
    rng = np.random.default_rng(42)
    act_rand = random_policy(obs_dict, env=dummy_env, rng=rng)
    assert len(act_rand) == 3
    for a in dummy_env.agents:
        assert act_rand[a].shape == (2,)
        assert np.all(act_rand[a] >= -1.0) and np.all(act_rand[a] <= 1.0)

    # 2. Greedy nearest policy
    act_greedy = greedy_nearest_policy(obs_dict, env=dummy_env)
    assert len(act_greedy) == 3
    for a in dummy_env.agents:
        assert act_greedy[a].shape == (2,)
        assert not np.isnan(act_greedy[a]).any()

    # 3. Collision-aware greedy policy
    act_col = collision_aware_greedy_policy(obs_dict, env=dummy_env)
    assert len(act_col) == 3
    for a in dummy_env.agents:
        assert act_col[a].shape == (2,)
        assert not np.isnan(act_col[a]).any()

    # 4. Target assignment heuristic policy
    act_assign = target_assignment_heuristic_policy(obs_dict, env=dummy_env)
    assert len(act_assign) == 3
    for a in dummy_env.agents:
        assert act_assign[a].shape == (2,)
        assert not np.isnan(act_assign[a]).any()


def test_baseline_hierarchy_teammate_repulsion():
    """Verify collision_aware_greedy pushes drones apart when teammates are within 6.0 sim_units."""
    dummy_env = DummyEnvNoState()

    # Place drone_0 and drone_1 extremely close: drone_1 is 0.02 (2.0 sim_units) to the right of drone_0
    obs_dict = {}

    # Drone 0 observation: teammate 1 is at rel_pos [0.02, 0.0]
    obs_0 = np.zeros(39, dtype=np.float32)
    obs_0[0:2] = [0.5, 0.5]
    obs_0[4:6] = [0.02, 0.0]       # teammate 1 is to the right
    obs_0[8:10] = [0.8, 0.8]       # teammate 2 is far away
    # Target 0 is directly to the right [0.1, 0.0]
    obs_0[12:14] = [0.1, 0.0]
    obs_0[16] = 1.0
    obs_dict["drone_0"] = obs_0

    # Drone 1 observation: teammate 1 (drone_0) is at rel_pos [-0.02, 0.0]
    obs_1 = np.zeros(39, dtype=np.float32)
    obs_1[0:2] = [0.52, 0.5]
    obs_1[4:6] = [-0.02, 0.0]      # teammate 0 is to the left
    obs_1[8:10] = [0.8, 0.8]       # teammate 2 is far away
    obs_1[12:14] = [0.08, 0.0]     # target 0 is to the right
    obs_1[16] = 1.0
    obs_dict["drone_1"] = obs_1

    obs_2 = np.zeros(39, dtype=np.float32)
    obs_2[0:2] = [0.9, 0.9]
    obs_dict["drone_2"] = obs_2

    greedy_acts = greedy_nearest_policy(obs_dict, env=dummy_env)
    col_acts = collision_aware_greedy_policy(obs_dict, env=dummy_env)

    # For drone 0: teammate is to the right (+x), so collision repulsion must push drone 0 to the LEFT (-x)
    # Therefore, col_acts["drone_0"][0] must be strictly smaller (more negative / less positive) than greedy_acts["drone_0"][0]
    assert col_acts["drone_0"][0] < greedy_acts["drone_0"][0]

    # For drone 1: teammate is to the left (-x), so collision repulsion must push drone 1 to the RIGHT (+x)
    assert col_acts["drone_1"][0] > greedy_acts["drone_1"][0]


def test_target_assignment_distinct_target_dispatch():
    """Verify target_assignment_heuristic_policy dispatches drones to distinct observed targets."""
    dummy_env = DummyEnvNoState()

    # Drone 0 is at [0.2, 0.5]
    # Drone 1 is at [0.5, 0.5]
    # Target 0 is at [0.2, 0.6] (closer to Drone 0)
    # Target 1 is at [0.5, 0.6] (closer to Drone 1)
    obs_dict = {}

    obs_0 = np.zeros(39, dtype=np.float32)
    obs_0[0:2] = [0.2, 0.5]
    obs_0[12:14] = [0.0, 0.1]    # Target 0 rel pos
    obs_0[16] = 1.0
    obs_0[18:20] = [0.3, 0.1]    # Target 1 rel pos
    obs_0[22] = 1.0
    obs_dict["drone_0"] = obs_0

    obs_1 = np.zeros(39, dtype=np.float32)
    obs_1[0:2] = [0.5, 0.5]
    obs_1[12:14] = [-0.3, 0.1]   # Target 0 rel pos
    obs_1[16] = 1.0
    obs_1[18:20] = [0.0, 0.1]    # Target 1 rel pos
    obs_1[22] = 1.0
    obs_dict["drone_1"] = obs_1

    obs_2 = np.zeros(39, dtype=np.float32)
    obs_2[0:2] = [0.8, 0.5]
    obs_dict["drone_2"] = obs_2

    assign_acts = target_assignment_heuristic_policy(obs_dict, env=dummy_env)

    # Drone 0 should head towards Target 0 (+y, 0 x)
    # Drone 1 should head towards Target 1 (+y, 0 x)
    assert abs(assign_acts["drone_0"][0]) < 0.2
    assert assign_acts["drone_0"][1] > 0.5
    assert abs(assign_acts["drone_1"][0]) < 0.2
    assert assign_acts["drone_1"][1] > 0.5


def test_evaluation_reproducibility(mock_env):
    """Verify run_policy_evaluation produces bit-for-bit identical results on identical seeds."""
    res1 = run_policy_evaluation(mock_env, greedy_nearest_policy, episodes=3)
    res2 = run_policy_evaluation(mock_env, greedy_nearest_policy, episodes=3)

    assert res1["mean_tcr"] == pytest.approx(res2["mean_tcr"], abs=1e-6)
    assert res1["std_tcr"] == pytest.approx(res2["std_tcr"], abs=1e-6)
    assert res1["mean_ate"] == pytest.approx(res2["mean_ate"], abs=1e-6)
    assert res1["std_ate"] == pytest.approx(res2["std_ate"], abs=1e-6)
    assert res1["total_collisions"] == res2["total_collisions"]

    # Also verify random policy reproducibility with seed propagation
    rand_res1 = run_policy_evaluation(mock_env, random_policy, episodes=3)
    rand_res2 = run_policy_evaluation(mock_env, random_policy, episodes=3)

    assert rand_res1["mean_tcr"] == pytest.approx(rand_res2["mean_tcr"], abs=1e-6)
    assert rand_res1["total_collisions"] == rand_res2["total_collisions"]


def test_train_config_metadata_serialization(synthetic_data_dir, monkeypatch):
    """Verify train_mappo_rgbt serializes complete experiment metadata to JSON."""
    work_dir = os.path.abspath(os.path.join(synthetic_data_dir, "..", ".."))
    monkeypatch.chdir(work_dir)

    # Run tiny 2-step train run without slow validation evaluation
    train_mappo_rgbt(
        total_timesteps=4,
        rollout_steps=2,
        eval_interval=50,
        batch_size=2,
        seed=123,
        exp_name="test_audit_exp",
        w_overlap=1.5,
        w_col=8.0,
        indiv_weight=0.3
    )

    config_file = os.path.join(work_dir, "results", "train_config_test_audit_exp.json")
    assert os.path.exists(config_file)

    with open(config_file, "r") as f:
        data = json.load(f)

    assert data["exp_name"] == "test_audit_exp"
    assert data["seed"] == 123
    assert data["reward_weights"]["w_overlap"] == 1.5
    assert data["reward_weights"]["w_col"] == 8.0
    assert data["reward_weights"]["indiv_weight"] == 0.3
    assert "git_commit" in data["system"]
    assert "reproducibility_notes" in data
