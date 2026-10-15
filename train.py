# train.py
import os
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from envs.uav_rgbt_env import UAVRGBTTrackingEnv
from algos.mappo_agent import MAPPOManager


def evaluate_validation_split(val_env, agent, num_episodes=10):
    """Evaluates current policy deterministically on the held-out Anti-UAV-RGBT validation split."""
    agent.actor_cpu.eval()
    tcr_scores, col_scores, ate_scores, overlap_scores = [], [], [], []

    for ep in range(num_episodes):
        obs_dict, _ = val_env.reset(seed=1000 + ep)
        done = False
        while not done:
            obs_arr = np.stack([obs_dict[a] for a in val_env.agents]).astype(np.float32)
            obs_t = torch.from_numpy(obs_arr)
            with torch.no_grad():
                act_t, _, _ = agent.actor_cpu.get_action_and_logprob(obs_t, deterministic=True, update_norm=False)
            act_np = act_t.numpy()
            obs_dict, _, terms, truncs, infos = val_env.step({a: act_np[i] for i, a in enumerate(val_env.agents)})
            done = any(terms.values()) or any(truncs.values())

            tcr_scores.append(infos["drone_0"]["tcr"])
            col_scores.append(infos["drone_0"]["collisions"])
            ate_scores.append(infos["drone_0"]["ate"])
            overlap_scores.append(infos["drone_0"]["overlap"])

    agent.actor_cpu.train()
    return (
        float(np.mean(tcr_scores)),
        float(np.mean(col_scores)),
        float(np.mean(ate_scores)),
        float(np.mean(overlap_scores)),
    )


def train_mappo_rgbt(
    total_timesteps=400_000,
    rollout_steps=2000,
    eval_interval=10,
    batch_size=512,
    resume=False,
    lr=3e-4
):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    writer = SummaryWriter("runs/mappo_anti_uav_rgbt")

    train_env = UAVRGBTTrackingEnv(num_drones=3, num_targets=4, split="train")
    val_env = UAVRGBTTrackingEnv(num_drones=3, num_targets=4, split="val")
    agent = MAPPOManager(obs_dim=train_env.obs_dim, state_dim=train_env.state_dim, lr=lr, device="cuda")

    start_step = 0
    best_val_tcr = -1.0
    history_file = "results/training_history.npz"

    history_steps = []
    history_rew = []
    history_tcr = []
    history_col = []
    history_ate = []
    history_overlap = []

    if resume:
        actor_ckpt = "checkpoints/mappo_actor_best.pt"
        if not os.path.exists(actor_ckpt):
            actor_ckpt = "checkpoints/mappo_actor_latest.pt"
        critic_ckpt = "checkpoints/mappo_critic_best.pt"
        if not os.path.exists(critic_ckpt):
            critic_ckpt = "checkpoints/mappo_critic_latest.pt"

        if os.path.exists(actor_ckpt):
            print(f"[RESUME] Loading actor weights from {actor_ckpt}")
            agent.actor.load_state_dict(torch.load(actor_ckpt, map_location=agent.device, weights_only=True))
        if os.path.exists(critic_ckpt):
            print(f"[RESUME] Loading critic weights from {critic_ckpt}")
            agent.critic.load_state_dict(torch.load(critic_ckpt, map_location=agent.device, weights_only=True))
        agent.sync_cpu_actor()

        # Update learning rates
        agent.base_lr = lr
        for pg in agent.actor_opt.param_groups:
            pg["lr"] = lr
        for pg in agent.critic_opt.param_groups:
            pg["lr"] = lr

        if os.path.exists(history_file):
            prev = np.load(history_file)
            history_steps = list(prev["steps"])
            history_rew = list(prev["reward"])
            history_tcr = list(prev["tcr"])
            history_col = list(prev["collisions"])
            history_ate = list(prev["ate"])
            history_overlap = list(prev["overlap"]) if "overlap" in prev else [0.0] * len(history_steps)
            if history_steps:
                start_step = int(history_steps[-1])
                best_val_tcr = float(np.max(history_tcr) / 100.0)
            print(f"[RESUME] Resuming from step {start_step:,} | Previous Best Val TCR: {best_val_tcr:.2%}")

    obs_dict, info_dict = train_env.reset()
    global_step = start_step
    target_global_step = start_step + total_timesteps
    rollout_count = 0
    start_time = time.time()

    print(f"[START] Vectorized MAPPO Training on {agent.device} (CPU Rollouts + GPU PPO Updates)")
    print(f"[DIMS]  Local Obs Dim: {train_env.obs_dim} | Centralized State Dim: {train_env.state_dim}")
    print(f"[CONFIG] Run Steps: {total_timesteps:,} (Target: {target_global_step:,}) | Rollout Steps: {rollout_steps:,} | Batch Size: {batch_size} | LR: {lr}")

    while global_step < target_global_step:
        agent.anneal_lr(global_step - start_step, total_timesteps)
        obs_buf, state_buf, act_buf, logprob_buf = [], [], [], []
        rew_buf, val_buf, done_buf = [], [], []
        tcr_list, col_list, ate_list, overlap_list = [], [], [], []

        # High-speed rollout collection (zero CUDA driver latency)
        for _ in range(rollout_steps):
            global_step += 1
            global_state = info_dict["drone_0"]["state"]

            obs_array = np.stack([obs_dict[a] for a in train_env.agents]).astype(np.float32) # (3, 39)
            obs_tensor = torch.from_numpy(obs_array)
            state_tensor = torch.from_numpy(global_state.astype(np.float32)).unsqueeze(0)    # (1, 28)

            with torch.no_grad():
                actions_t, logprobs_t, _ = agent.actor_cpu.get_action_and_logprob(obs_tensor, update_norm=False)
                value_t = agent.critic_cpu(state_tensor, update_norm=False).item()

            actions_np = actions_t.numpy()
            logprobs_np = logprobs_t.numpy()
            action_dict = {a: actions_np[i] for i, a in enumerate(train_env.agents)}

            next_obs_dict, rewards, terms, truncs, next_info_dict = train_env.step(action_dict)
            done = any(terms.values()) or any(truncs.values())

            obs_buf.append(obs_array)
            state_buf.append(np.repeat(global_state[None, :], train_env.num_drones, axis=0))
            act_buf.append(actions_np)
            logprob_buf.append(logprobs_np)
            rew_buf.append(rewards["drone_0"])
            val_buf.append(value_t)
            done_buf.append(float(done))

            tcr_list.append(next_info_dict["drone_0"]["tcr"])
            col_list.append(next_info_dict["drone_0"]["collisions"])
            ate_list.append(next_info_dict["drone_0"]["ate"])
            overlap_list.append(next_info_dict["drone_0"]["overlap"])

            if done:
                obs_dict, info_dict = train_env.reset()
            else:
                obs_dict, info_dict = next_obs_dict, next_info_dict

        # Flatten rollout buffers
        N = train_env.num_drones
        obs_b = np.array(obs_buf).reshape(-1, train_env.obs_dim)
        state_b = np.array(state_buf).reshape(-1, train_env.state_dim)

        # Update running mean/std ONCE per rollout directly on GPU
        obs_batch_t = torch.tensor(obs_b, dtype=torch.float32, device=agent.device)
        state_batch_t = torch.tensor(state_b, dtype=torch.float32, device=agent.device)
        agent.actor.obs_norm.update(obs_batch_t)
        agent.critic.state_norm.update(state_batch_t)
        agent.sync_cpu_actor()

        # Compute next-state bootstrap value
        with torch.no_grad():
            next_state_t = torch.from_numpy(info_dict["drone_0"]["state"].astype(np.float32)).unsqueeze(0)
            next_val = agent.critic_cpu(next_state_t, update_norm=False).item()

        advs, rets = agent.compute_gae(rew_buf, val_buf, done_buf, next_val)

        # GPU PPO update with batch_size
        update_stats = agent.update(
            obs_b=obs_b,
            state_b=state_b,
            act_b=np.array(act_buf).reshape(-1, 2),
            logprob_b=np.array(logprob_buf).reshape(-1),
            val_b=np.repeat(np.array(val_buf)[:, None], N, axis=1).reshape(-1),
            adv_b=np.repeat(advs[:, None], N, axis=1).reshape(-1),
            ret_b=np.repeat(rets[:, None], N, axis=1).reshape(-1),
            epochs=10,
            batch_size=batch_size
        )

        rollout_count += 1
        elapsed = time.time() - start_time
        steps_done = global_step - start_step
        fps = int(steps_done / max(1e-5, elapsed))
        mean_rew = float(np.mean(rew_buf))
        mean_tcr = float(np.mean(tcr_list))
        mean_col = float(np.mean(col_list))
        mean_ate = float(np.mean(ate_list))
        mean_overlap = float(np.mean(overlap_list))

        # Log training metrics
        writer.add_scalar("Train/Reward", mean_rew, global_step)
        writer.add_scalar("Train/Target_Coverage_Ratio", mean_tcr, global_step)
        writer.add_scalar("Train/Collisions", mean_col, global_step)
        writer.add_scalar("Train/Avg_Tracking_Error", mean_ate, global_step)
        writer.add_scalar("Train/Overlap", mean_overlap, global_step)
        writer.add_scalar("Train/FPS", fps, global_step)
        writer.add_scalar("Train/Actor_Loss", update_stats["actor_loss"], global_step)
        writer.add_scalar("Train/Critic_Loss", update_stats["critic_loss"], global_step)
        writer.add_scalar("Train/Entropy", update_stats["entropy"], global_step)

        history_steps.append(global_step)
        history_rew.append(mean_rew)
        history_tcr.append(mean_tcr * 100.0)
        history_col.append(mean_col)
        history_ate.append(mean_ate)
        history_overlap.append(mean_overlap)

        # Periodic Validation Evaluation
        if rollout_count % eval_interval == 0 or global_step >= target_global_step:
            val_tcr, val_col, val_ate, val_overlap = evaluate_validation_split(val_env, agent)
            writer.add_scalar("Val/Target_Coverage_Ratio", val_tcr, global_step)
            writer.add_scalar("Val/Collisions", val_col, global_step)
            writer.add_scalar("Val/Avg_Tracking_Error", val_ate, global_step)
            writer.add_scalar("Val/Overlap", val_overlap, global_step)

            print(
                f"Step: {global_step:7d}/{target_global_step:,} | FPS: {fps:4d} | Train TCR: {mean_tcr:6.2%} | "
                f"Val TCR: {val_tcr:6.2%} | Val Overlap: {val_overlap:.2f} | Val Col: {val_col:.3f} | Val ATE: {val_ate:.1f}m"
            )

            # Save latest checkpoints
            torch.save(agent.actor.state_dict(), "checkpoints/mappo_actor_latest.pt")
            torch.save(agent.critic.state_dict(), "checkpoints/mappo_critic_latest.pt")

            # Save best checkpoints
            if val_tcr >= best_val_tcr:
                best_val_tcr = val_tcr
                torch.save(agent.actor.state_dict(), "checkpoints/mappo_actor_best.pt")
                torch.save(agent.critic.state_dict(), "checkpoints/mappo_critic_best.pt")
                print(f"  --> [NEW BEST] Checkpoint saved: Val TCR = {val_tcr:.2%}")

    writer.close()

    # Save Complete Training History Arrays
    np.savez(
        history_file,
        steps=np.array(history_steps, dtype=np.int32),
        reward=np.array(history_rew, dtype=np.float32),
        tcr=np.array(history_tcr, dtype=np.float32),
        collisions=np.array(history_col, dtype=np.float32),
        ate=np.array(history_ate, dtype=np.float32),
        overlap=np.array(history_overlap, dtype=np.float32)
    )
    print(f"[SAVED] Complete training history arrays saved to {history_file} ({len(history_steps)} records)")

    # Generate and Save 3-Panel Publication-Quality Figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=200)

    axes[0].plot(history_steps, history_rew, color="#2563eb", lw=1.8)
    axes[0].set_title("Shared Team Reward", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Timesteps", fontsize=10)
    axes[0].set_ylabel("Reward", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(history_steps, history_tcr, color="#16a34a", lw=1.8)
    axes[1].set_title("Target Coverage Ratio (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Timesteps", fontsize=10)
    axes[1].set_ylabel("Coverage (%)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(history_steps, history_overlap, color="#eab308", lw=1.8, label="Redundant Overlaps")
    axes[2].plot(history_steps, history_col, color="#dc2626", lw=1.8, label="Collisions / Step")
    axes[2].set_title("Overlaps & Collisions / Step", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Timesteps", fontsize=10)
    axes[2].legend(loc="upper right", frameon=True)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    curve_path = "results/training_curves.png"
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"[DONE] Final Best Validation TCR: {best_val_tcr:.2%} | Full curve saved to {curve_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vectorized MAPPO on Anti-UAV-RGBT")
    parser.add_argument("--total-timesteps", type=int, default=400_000, help="Total environment steps to run")
    parser.add_argument("--rollout-steps", type=int, default=2000, help="Steps per rollout buffer")
    parser.add_argument("--eval-interval", type=int, default=10, help="Evaluation interval in rollouts")
    parser.add_argument("--batch-size", type=int, default=512, help="PPO minibatch size")
    parser.add_argument("--resume", action="store_true", help="Resume from best checkpoint and append history")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    train_mappo_rgbt(
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        eval_interval=args.eval_interval,
        batch_size=args.batch_size,
        resume=args.resume,
        lr=args.lr
    )
