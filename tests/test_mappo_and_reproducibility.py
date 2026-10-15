# tests/test_mappo_and_reproducibility.py
import os
import pytest
import numpy as np
import torch
from algos.mappo_agent import MAPPOManager
from algos.networks import SharedActor, CentralizedCritic
from envs.uav_rgbt_env import UAVRGBTTrackingEnv


# ---------------------------------------------------------------------------
# 1. Per-Agent GAE & Credit Assignment (AUD-01)
# ---------------------------------------------------------------------------

def test_per_agent_gae_credit_assignment():
    """Verify 2D compute_gae properly assigns an isolated penalty on agent 1 without corrupting agent 0 or 2."""
    agent = MAPPOManager(obs_dim=39, state_dim=28, device="cpu")
    T, N = 10, 3

    # Agents 0 and 2 receive neutral/positive rewards (+1.0)
    # Agent 1 receives a severe collision penalty (-5.0) at step 5
    rewards = np.ones((T, N), dtype=np.float32)
    rewards[5, 1] = -5.0

    values = np.ones(T, dtype=np.float32) * 0.5
    dones = np.zeros(T, dtype=np.float32)
    next_val = 0.5

    advs, rets = agent.compute_gae(rewards, values, dones, next_val)

    assert advs.shape == (T, N)
    assert rets.shape == (T, N)

    # At step 5, agent 1's delta is -5.005 while agents 0 and 2 have delta +0.995.
    # The relative advantage penalty (advs[5, 0] - advs[5, 1]) must be exactly 6.0.
    assert (advs[5, 0] - advs[5, 1]) == pytest.approx(6.0, abs=1e-4)
    assert advs[5, 1] < 0.0
    assert advs[5, 0] > 0.0
    assert advs[5, 2] > 0.0
    assert abs(advs[5, 0] - advs[5, 2]) < 1e-6


def test_gae_backward_compatibility_1d():
    """Verify compute_gae handles 1D scalar reward streams without errors."""
    agent = MAPPOManager(obs_dim=39, state_dim=28, device="cpu")
    T = 15
    rewards = np.ones(T, dtype=np.float32)
    values = np.zeros(T, dtype=np.float32)
    dones = np.zeros(T, dtype=np.float32)

    advs, rets = agent.compute_gae(rewards, values, dones, next_value=0.0)
    assert advs.shape == (T,)
    assert rets.shape == (T,)
    assert np.all(np.isfinite(advs))


# ---------------------------------------------------------------------------
# 2. PPO Ratio & RunningMeanStd Normalization Timing (AUD-06)
# ---------------------------------------------------------------------------

def test_initial_ppo_importance_sampling_ratio():
    """Verify that prior to gradient steps, the importance sampling ratio is strictly 1.0."""
    agent = MAPPOManager(obs_dim=39, state_dim=28, device="cpu")
    obs_batch = np.random.randn(32, 39).astype(np.float32)
    obs_t = torch.from_numpy(obs_batch)

    # 1. Rollout sampling
    with torch.no_grad():
        actions, rollout_logprobs, _ = agent.actor.get_action_and_logprob(obs_t)

    # 2. Re-evaluating logprob with exact same network weights
    with torch.no_grad():
        _, current_logprobs, _ = agent.actor.get_action_and_logprob(obs_t, actions)
        ratio = torch.exp(current_logprobs - rollout_logprobs).numpy()

    np.testing.assert_allclose(ratio, 1.0, atol=1e-5, err_msg="Initial ratio must be 1.0!")


def test_running_mean_std_sync_after_update():
    """Verify obs_norm updates after update() and syncs to actor_cpu."""
    agent = MAPPOManager(obs_dim=39, state_dim=28, device="cpu")

    # Capture initial mean
    initial_mean = agent.actor.obs_norm.mean.clone().numpy()
    initial_cpu_mean = agent.actor_cpu.obs_norm.mean.clone().numpy()
    np.testing.assert_allclose(initial_mean, initial_cpu_mean)

    # Generate synthetic batch with shifted mean (+5.0)
    obs_b = (np.random.randn(64, 39) + 5.0).astype(np.float32)
    state_b = (np.random.randn(64, 28) + 2.0).astype(np.float32)
    act_b = np.zeros((64, 2), dtype=np.float32)
    logprob_b = np.zeros(64, dtype=np.float32)
    val_b = np.zeros(64, dtype=np.float32)
    adv_b = np.ones(64, dtype=np.float32)
    ret_b = np.ones(64, dtype=np.float32)

    stats = agent.update(
        obs_b=obs_b,
        state_b=state_b,
        act_b=act_b,
        logprob_b=logprob_b,
        val_b=val_b,
        adv_b=adv_b,
        ret_b=ret_b,
        epochs=1,
        batch_size=32
    )

    # After update(), mean should have shifted toward +5.0 and synced to CPU mirror
    updated_gpu_mean = agent.actor.obs_norm.mean.clone().numpy()
    updated_cpu_mean = agent.actor_cpu.obs_norm.mean.clone().numpy()

    assert not np.allclose(initial_mean, updated_gpu_mean)
    np.testing.assert_allclose(updated_gpu_mean, updated_cpu_mean)
    assert np.isfinite(stats["actor_loss"])
    assert np.isfinite(stats["critic_loss"])


# ---------------------------------------------------------------------------
# 3. Checkpoint Saving & Loading (Req 9)
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip(tmp_path):
    """Verify saved checkpoint weights and normalization buffers reproduce identical deterministic outputs."""
    agent1 = MAPPOManager(obs_dim=39, state_dim=28, device="cpu")

    # Shift running stats with sample data
    obs_sample = (np.random.randn(100, 39) + 3.0).astype(np.float32)
    state_sample = (np.random.randn(100, 28) + 1.0).astype(np.float32)
    agent1.actor.obs_norm.update(torch.from_numpy(obs_sample))
    agent1.critic.state_norm.update(torch.from_numpy(state_sample))
    agent1.sync_cpu_actor()

    # Save checkpoints to tmp_path
    actor_path = str(tmp_path / "test_actor.pt")
    critic_path = str(tmp_path / "test_critic.pt")
    torch.save(agent1.actor.state_dict(), actor_path)
    torch.save(agent1.critic.state_dict(), critic_path)

    # Load into a fresh agent
    agent2 = MAPPOManager(obs_dim=39, state_dim=28, device="cpu")
    agent2.actor.load_state_dict(torch.load(actor_path, map_location="cpu", weights_only=True))
    agent2.critic.load_state_dict(torch.load(critic_path, map_location="cpu", weights_only=True))
    agent2.sync_cpu_actor()

    # Verify identical deterministic actions and values
    test_obs = torch.from_numpy(np.random.randn(10, 39).astype(np.float32))
    test_state = torch.from_numpy(np.random.randn(10, 28).astype(np.float32))

    with torch.no_grad():
        act1, _, _ = agent1.actor_cpu.get_action_and_logprob(test_obs, deterministic=True)
        act2, _, _ = agent2.actor_cpu.get_action_and_logprob(test_obs, deterministic=True)
        val1 = agent1.critic_cpu(test_state)
        val2 = agent2.critic_cpu(test_state)

    np.testing.assert_allclose(act1.numpy(), act2.numpy(), atol=1e-6)
    np.testing.assert_allclose(val1.numpy(), val2.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Reproducible Evaluation & Small End-to-End Smoke Training (Req 10 & 12)
# ---------------------------------------------------------------------------

def test_deterministic_seeded_evaluation(mock_env):
    """Verify evaluation runs with identical seeds yield identical trajectories and TCR metrics."""
    agent = MAPPOManager(obs_dim=mock_env.obs_dim, state_dim=mock_env.state_dim, device="cpu")
    agent.actor_cpu.eval()

    def run_eval_ep(seed):
        obs_dict, _ = mock_env.reset(seed=seed)
        tcr_history = []
        for _ in range(25):
            obs_arr = np.stack([obs_dict[a] for a in mock_env.agents]).astype(np.float32)
            with torch.no_grad():
                act_t, _, _ = agent.actor_cpu.get_action_and_logprob(torch.from_numpy(obs_arr), deterministic=True)
            actions = {a: act_t[i].numpy() for i, a in enumerate(mock_env.agents)}
            obs_dict, _, _, _, infos = mock_env.step(actions)
            tcr_history.append(infos["drone_0"]["tcr"])
        return tcr_history

    traj1 = run_eval_ep(seed=42)
    traj2 = run_eval_ep(seed=42)
    np.testing.assert_allclose(traj1, traj2, atol=1e-6)


def test_cpu_smoke_training_loop(synthetic_data_dir, tmp_path):
    """Run a tiny 2-rollout (32-step) end-to-end CPU training loop without touching repo checkpoints/results."""
    env = UAVRGBTTrackingEnv(num_drones=3, num_targets=4, split="train", data_dir=synthetic_data_dir)
    agent = MAPPOManager(obs_dim=env.obs_dim, state_dim=env.state_dim, lr=1e-3, device="cpu")

    obs_dict, info_dict = env.reset(seed=123)
    rollout_steps = 16
    N = env.num_drones

    for _ in range(2): # 2 rollouts
        obs_buf, state_buf, act_buf, logprob_buf = [], [], [], []
        rew_buf, val_buf, done_buf = [], [], []

        for _ in range(rollout_steps):
            global_state = info_dict["drone_0"]["state"]
            obs_array = np.stack([obs_dict[a] for a in env.agents]).astype(np.float32)
            obs_tensor = torch.from_numpy(obs_array)
            state_tensor = torch.from_numpy(global_state.astype(np.float32)).unsqueeze(0)

            with torch.no_grad():
                actions_t, logprobs_t, _ = agent.actor_cpu.get_action_and_logprob(obs_tensor)
                value_t = agent.critic_cpu(state_tensor).item()

            actions_np = actions_t.numpy()
            logprobs_np = logprobs_t.numpy()
            action_dict = {a: actions_np[i] for i, a in enumerate(env.agents)}

            next_obs_dict, rewards, terms, truncs, next_info_dict = env.step(action_dict)
            done = any(terms.values()) or any(truncs.values())

            obs_buf.append(obs_array)
            state_buf.append(np.repeat(global_state[None, :], N, axis=0))
            act_buf.append(actions_np)
            logprob_buf.append(logprobs_np)
            rew_buf.append([rewards[a] for a in env.agents])
            val_buf.append(value_t)
            done_buf.append(float(done))

            obs_dict, info_dict = (env.reset() if done else (next_obs_dict, next_info_dict))

        obs_b = np.array(obs_buf).reshape(-1, env.obs_dim)
        state_b = np.array(state_buf).reshape(-1, env.state_dim)

        with torch.no_grad():
            next_val = agent.critic_cpu(torch.from_numpy(info_dict["drone_0"]["state"]).unsqueeze(0)).item()

        advs, rets = agent.compute_gae(rew_buf, val_buf, done_buf, next_val)

        update_stats = agent.update(
            obs_b=obs_b,
            state_b=state_b,
            act_b=np.array(act_buf).reshape(-1, 2),
            logprob_b=np.array(logprob_buf).reshape(-1),
            val_b=np.repeat(np.array(val_buf)[:, None], N, axis=1).reshape(-1),
            adv_b=advs.reshape(-1),
            ret_b=rets.reshape(-1),
            epochs=2,
            batch_size=16
        )

        assert np.isfinite(update_stats["actor_loss"])
        assert np.isfinite(update_stats["critic_loss"])
        assert np.isfinite(update_stats["entropy"])
