# tests/test_env_logic.py
import pytest
import numpy as np
from envs.uav_rgbt_env import UAVRGBTTrackingEnv


# ---------------------------------------------------------------------------
# 1. Reset & Step (Req 1)
# ---------------------------------------------------------------------------

def test_reset_determinism(synthetic_data_dir):
    """Verify reset(seed=42) produces 100% identical initial states and observations."""
    env1 = UAVRGBTTrackingEnv(num_drones=3, num_targets=4, split="val", data_dir=synthetic_data_dir)
    obs1, info1 = env1.reset(seed=42)

    env2 = UAVRGBTTrackingEnv(num_drones=3, num_targets=4, split="val", data_dir=synthetic_data_dir)
    obs2, info2 = env2.reset(seed=42)

    np.testing.assert_allclose(env1.drone_pos, env2.drone_pos)
    np.testing.assert_allclose(env1.target_pos, env2.target_pos)
    np.testing.assert_allclose(env1.pre_vel, env2.pre_vel)

    for agent in env1.agents:
        np.testing.assert_allclose(obs1[agent], obs2[agent])
        np.testing.assert_allclose(info1[agent]["state"], info2[agent]["state"])


def test_global_prng_isolation(synthetic_data_dir):
    """Verify env.reset(seed=...) does NOT alter the process-wide global np.random state."""
    np.random.seed(8888)
    global_before = np.random.uniform(size=5)

    env = UAVRGBTTrackingEnv(num_drones=3, num_targets=4, split="val", data_dir=synthetic_data_dir)
    env.reset(seed=42)

    np.random.seed(8888)
    global_after = np.random.uniform(size=5)

    np.testing.assert_allclose(global_before, global_after, err_msg="Global np.random state was mutated!")


def test_step_return_types(mock_env):
    """Verify step() returns standard PettingZoo 5-tuple with correct agent keys and types."""
    obs_dict, _ = mock_env.reset(seed=10)
    zero_actions = {a: np.zeros(2, dtype=np.float32) for a in mock_env.agents}

    next_obs, rewards, terminations, truncations, infos = mock_env.step(zero_actions)

    assert set(next_obs.keys()) == set(mock_env.agents)
    assert set(rewards.keys()) == set(mock_env.agents)
    assert set(terminations.keys()) == set(mock_env.agents)
    assert set(truncations.keys()) == set(mock_env.agents)
    assert set(infos.keys()) == set(mock_env.agents)

    for a in mock_env.agents:
        assert isinstance(rewards[a], float)
        assert isinstance(terminations[a], (bool, np.bool_))
        assert isinstance(truncations[a], (bool, np.bool_))
        assert isinstance(infos[a], dict)


# ---------------------------------------------------------------------------
# 2. Observation & State Validity (Req 2)
# ---------------------------------------------------------------------------

def test_obs_and_state_dimensions_and_finite(mock_env):
    """Verify obs_dim == 39, state_dim == 28, and all values are finite."""
    obs_dict, infos = mock_env.reset(seed=100)

    assert mock_env.obs_dim == 39
    assert mock_env.state_dim == 28

    state = mock_env.state()
    assert state.shape == (28,)
    assert np.all(np.isfinite(state))

    for a in mock_env.agents:
        obs = obs_dict[a]
        assert obs.shape == (39,)
        assert np.all(np.isfinite(obs))


def test_occlusion_zeroing_at_all_ranges(mock_env):
    """Verify occluded targets (rgb=0, ir=0) return strictly zeros at both < 25.0 and > 25.0 sim_units."""
    mock_env.reset(seed=1)

    # Force target 0 to be occluded across all timesteps
    mock_env.pre_rgb[0, :] = 0.0
    mock_env.pre_ir[0, :] = 0.0

    # Range A: Inside FOV (< 25.0 sim_units)
    mock_env.drone_pos[0] = np.array([50.0, 50.0], dtype=np.float32)
    mock_env.target_pos[0] = np.array([58.0, 50.0], dtype=np.float32) # dist = 8.0
    obs_near = mock_env._get_obs()["drone_0"][12:18] # target 0 slice
    np.testing.assert_allclose(obs_near, 0.0, atol=1e-7, err_msg="Near occluded target was not zeroed!")

    # Range B: Outside FOV (> 25.0 sim_units)
    mock_env.target_pos[0] = np.array([85.0, 50.0], dtype=np.float32) # dist = 35.0
    obs_far = mock_env._get_obs()["drone_0"][12:18] # target 0 slice
    np.testing.assert_allclose(obs_far, 0.0, atol=1e-7, err_msg="Far occluded target was not zeroed!")


def test_continuous_feature_scaling_across_boundary(mock_env):
    """Verify relative position coordinates scale continuously across the 25.0 sim_units threshold."""
    mock_env.reset(seed=1)
    mock_env.pre_rgb[1, :] = 1.0
    mock_env.pre_ir[1, :] = 1.0
    mock_env.drone_pos[0] = np.array([50.0, 50.0], dtype=np.float32)
    mock_env.drone_vel[0] = np.array([0.0, 0.0], dtype=np.float32)
    mock_env.target_vel[1] = np.array([0.0, 0.0], dtype=np.float32)

    # Average over 300 samples to verify continuous mean expectation
    near_samples = []
    far_samples = []
    for s in range(300):
        mock_env.np_random = np.random.default_rng(s)
        mock_env.target_pos[1] = np.array([50.0 + 24.9, 50.0], dtype=np.float32)
        near_samples.append(mock_env._get_obs()["drone_0"][18]) # Target 1 dx

        mock_env.np_random = np.random.default_rng(s)
        mock_env.target_pos[1] = np.array([50.0 + 25.1, 50.0], dtype=np.float32)
        far_samples.append(mock_env._get_obs()["drone_0"][18]) # Target 1 dx

    mean_near = np.mean(near_samples)
    mean_far = np.mean(far_samples)

    # Both must be normalized by arena_size (100.0) -> ~0.249 and ~0.251
    assert abs(mean_near - 0.249) < 0.003
    assert abs(mean_far - 0.251) < 0.003
    assert abs(mean_near - mean_far) < 0.005


# ---------------------------------------------------------------------------
# 3. Action Bounds & Kinematics (Req 3)
# ---------------------------------------------------------------------------

def test_action_clipping_and_kinematics_bounds(mock_env):
    """Verify out-of-bound actions are clipped, and speeds/positions stay strictly within bounds."""
    mock_env.reset(seed=5)
    mock_env.drone_pos[0] = np.array([99.0, 99.0], dtype=np.float32)

    # Extreme action: should be clipped to [1.0, 1.0]
    extreme_actions = {a: np.array([10.0, -10.0], dtype=np.float32) for a in mock_env.agents}

    for _ in range(20):
        mock_env.step(extreme_actions)
        assert np.all(np.abs(mock_env.drone_vel) <= mock_env.max_drone_speed + 1e-5)
        assert np.all(mock_env.drone_pos >= 0.0)
        assert np.all(mock_env.drone_pos <= mock_env.arena_size)


def test_target_boundary_reflection(mock_env):
    """Verify targets reflect at [5.0, 95.0] margins only when moving outward, inverting velocity and drift."""
    mock_env.reset(seed=7)
    mock_env.ou_weight = 0.0  # Disable stochastic OU noise to test kinematics deterministically

    # Case 1: Target at 4.0 (outside margin) but moving INWARD (+1.0 sim_units/s) -> should NOT reflect!
    mock_env.pre_vel[0, :] = 1.0
    mock_env.target_pos[0] = np.array([4.0, 50.0], dtype=np.float32)
    mock_env.reflection_sign[0] = np.array([1.0, 1.0], dtype=np.float32)

    zero_actions = {a: np.zeros(2, dtype=np.float32) for a in mock_env.agents}
    mock_env.step(zero_actions)

    # Moving inward: reflection_sign should remain 1.0 on x
    assert mock_env.reflection_sign[0, 0] == 1.0

    # Case 2: Target at 4.0 moving OUTWARD (-2.0 sim_units/s) -> should reflect and clip to 5.0
    mock_env.pre_vel[0, :] = -2.0
    mock_env.target_pos[0] = np.array([4.0, 50.0], dtype=np.float32)
    mock_env.reflection_sign[0] = np.array([1.0, 1.0], dtype=np.float32)

    mock_env.step(zero_actions)

    assert mock_env.reflection_sign[0, 0] == -1.0
    assert mock_env.target_pos[0, 0] >= 5.0


# ---------------------------------------------------------------------------
# 4. Collision Detection at Boundary Conditions (Req 4)
# ---------------------------------------------------------------------------

def test_collision_detection_threshold(mock_env):
    """Verify collision threshold at d_safe=2.5: distance 2.49 = 1 collision; distance 2.50 = 0 collisions."""
    mock_env.reset(seed=12)

    # Drones 0 and 1 placed at 2.49 apart (< 2.5) -> 1 collision
    mock_env.drone_pos[0] = np.array([50.0, 50.0], dtype=np.float32)
    mock_env.drone_pos[1] = np.array([52.49, 50.0], dtype=np.float32)
    mock_env.drone_pos[2] = np.array([10.0, 10.0], dtype=np.float32) # far away

    zero_actions = {a: np.zeros(2, dtype=np.float32) for a in mock_env.agents}
    _, _, _, _, infos = mock_env.step(zero_actions)
    assert infos["drone_0"]["collisions"] == 1

    # Drones 0 and 1 placed at 2.50 apart (>= 2.5) -> 0 collisions
    mock_env.drone_pos[0] = np.array([50.0, 50.0], dtype=np.float32)
    mock_env.drone_pos[1] = np.array([52.50, 50.0], dtype=np.float32)
    mock_env.drone_pos[2] = np.array([10.0, 10.0], dtype=np.float32)

    _, _, _, _, infos = mock_env.step(zero_actions)
    assert infos["drone_0"]["collisions"] == 0


def test_three_drone_pairwise_collision(mock_env):
    """Verify when all 3 drones are within < 2.5 of each other, 3 pairwise collisions are recorded."""
    mock_env.reset(seed=12)
    mock_env.drone_pos[0] = np.array([50.0, 50.0], dtype=np.float32)
    mock_env.drone_pos[1] = np.array([51.0, 50.0], dtype=np.float32)
    mock_env.drone_pos[2] = np.array([50.5, 50.8], dtype=np.float32)

    zero_actions = {a: np.zeros(2, dtype=np.float32) for a in mock_env.agents}
    _, _, _, _, infos = mock_env.step(zero_actions)
    assert infos["drone_0"]["collisions"] == 3


# ---------------------------------------------------------------------------
# 5. Target Coverage & Redundant Overlap (Req 5)
# ---------------------------------------------------------------------------

def test_target_coverage_and_overlap_counts(mock_env):
    """Verify target coverage ratio and redundant overlaps at deterministic positions."""
    mock_env.reset(seed=20)
    mock_env.pre_vel[:, :] = 0.0 # freeze target motions

    # Place 4 targets far apart in corners
    mock_env.target_pos[0] = np.array([10.0, 10.0], dtype=np.float32)
    mock_env.target_pos[1] = np.array([90.0, 10.0], dtype=np.float32)
    mock_env.target_pos[2] = np.array([10.0, 90.0], dtype=np.float32)
    mock_env.target_pos[3] = np.array([90.0, 90.0], dtype=np.float32)

    zero_actions = {a: np.zeros(2, dtype=np.float32) for a in mock_env.agents}

    # Case A: 0 drones near any target -> TCR = 0.0, overlap = 0
    mock_env.drone_pos[0] = np.array([50.0, 50.0], dtype=np.float32)
    mock_env.drone_pos[1] = np.array([50.0, 50.0], dtype=np.float32)
    mock_env.drone_pos[2] = np.array([50.0, 50.0], dtype=np.float32)
    _, _, _, _, infos = mock_env.step(zero_actions)
    assert infos["drone_0"]["tcr"] == 0.0
    assert infos["drone_0"]["overlap"] == 0

    # Case B: 1 drone covering target 0 -> TCR = 1/4 = 0.25, overlap = 0
    mock_env.drone_pos[0] = np.array([10.0, 10.0], dtype=np.float32)
    mock_env.drone_pos[1] = np.array([50.0, 50.0], dtype=np.float32)
    mock_env.drone_pos[2] = np.array([50.0, 50.0], dtype=np.float32)
    _, _, _, _, infos = mock_env.step(zero_actions)
    assert infos["drone_0"]["tcr"] == 0.25
    assert infos["drone_0"]["overlap"] == 0

    # Case C: 2 drones covering target 0 -> TCR = 0.25, overlap = 1
    mock_env.drone_pos[0] = np.array([10.0, 10.0], dtype=np.float32)
    mock_env.drone_pos[1] = np.array([12.0, 10.0], dtype=np.float32)
    mock_env.drone_pos[2] = np.array([50.0, 50.0], dtype=np.float32)
    _, _, _, _, infos = mock_env.step(zero_actions)
    assert infos["drone_0"]["tcr"] == 0.25
    assert infos["drone_0"]["overlap"] == 1

    # Case D: 3 drones all covering target 0 -> TCR = 0.25, overlap = 2
    mock_env.drone_pos[0] = np.array([10.0, 10.0], dtype=np.float32)
    mock_env.drone_pos[1] = np.array([12.0, 10.0], dtype=np.float32)
    mock_env.drone_pos[2] = np.array([10.0, 12.0], dtype=np.float32)
    _, _, _, _, infos = mock_env.step(zero_actions)
    assert infos["drone_0"]["tcr"] == 0.25
    assert infos["drone_0"]["overlap"] == 2


# ---------------------------------------------------------------------------
# 6. Reward Components & Aggregation (Req 6)
# ---------------------------------------------------------------------------

def test_reward_breakdown_and_formula(mock_env):
    """Verify reward_breakdown dictionary and exact mathematical aggregation formula."""
    mock_env.reset(seed=30)
    zero_actions = {a: np.zeros(2, dtype=np.float32) for a in mock_env.agents}

    _, rewards, _, _, infos = mock_env.step(zero_actions)
    rb = infos["drone_0"]["reward_breakdown"]

    expected_team = (
        rb["cov_reward"]
        + rb["lock_bonus"]
        - rb["dist_penalty"]
        - rb["overlap_penalty"]
        - rb["col_penalty"]
        - rb["prox_penalty"]
        - rb["smooth_penalty"]
    )
    assert abs(rb["team_reward"] - expected_team) < 1e-5

    # Check per-agent reward aggregation
    for i, a in enumerate(mock_env.agents):
        indiv_r = rb["indiv_rewards"][i]
        expected_agent_reward = (1.0 - mock_env.indiv_weight) * rb["team_reward"] + mock_env.indiv_weight * indiv_r
        assert abs(rewards[a] - expected_agent_reward) < 1e-5


def test_configurable_reward_weights(synthetic_data_dir):
    """Verify setting ablation weights (e.g., w_overlap=0.0, w_col=0.0) zeroes out respective penalties."""
    ablated_env = UAVRGBTTrackingEnv(
        data_dir=synthetic_data_dir,
        w_overlap=0.0,
        w_col=0.0,
        render_mode=None
    )
    ablated_env.reset(seed=31)

    # Force a collision and an overlap
    ablated_env.drone_pos[0] = np.array([50.0, 50.0], dtype=np.float32)
    ablated_env.drone_pos[1] = np.array([51.0, 50.0], dtype=np.float32) # collision
    ablated_env.target_pos[0] = np.array([50.0, 50.0], dtype=np.float32) # overlap for both drones

    zero_actions = {a: np.zeros(2, dtype=np.float32) for a in ablated_env.agents}
    _, _, _, _, infos = ablated_env.step(zero_actions)
    rb = infos["drone_0"]["reward_breakdown"]

    assert rb["overlap_penalty"] == 0.0
    assert rb["col_penalty"] == 0.0


# ---------------------------------------------------------------------------
# 7. Episode Termination & Truncation (Req 7)
# ---------------------------------------------------------------------------

def test_truncation_at_max_steps(mock_env):
    """Verify episode does not terminate early, and truncates exactly when step_count reaches max_steps (200)."""
    mock_env.reset(seed=40)
    zero_actions = {a: np.zeros(2, dtype=np.float32) for a in mock_env.agents}

    for step_i in range(1, mock_env.max_steps):
        _, _, terms, truncs, _ = mock_env.step(zero_actions)
        assert not any(terms.values())
        assert not any(truncs.values())

    # Step 200: should truncate
    _, _, terms, truncs, _ = mock_env.step(zero_actions)
    assert not any(terms.values())
    assert all(truncs.values())
    assert mock_env.step_count == 200
