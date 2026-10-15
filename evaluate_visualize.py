# evaluate_visualize.py
import os
import json
import argparse
from collections import defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from envs.uav_rgbt_env import UAVRGBTTrackingEnv
from algos.networks import SharedActor


def run_policy_evaluation(env, policy_fn, episodes=30, record_trail=False):
    all_tcr = []
    all_overlap = []
    all_col = []
    all_ate = []

    tag_total_counts = defaultdict(int)
    tag_covered_counts = defaultdict(int)

    drone_trails = [[] for _ in range(env.num_drones)]
    target_trails = [[] for _ in range(env.num_targets)]
    trail_metadata = {}

    for ep in range(episodes):
        obs_dict, info_dict = env.reset(seed=5000 + ep)
        done = False
        ep_tcr, ep_overlap, ep_col, ep_ate = [], [], [], []

        record_this_ep = record_trail and (ep == 0)
        if record_this_ep:
            trail_metadata["records"] = [
                {
                    "seq_id": env.active_records[m].get("seq_id", f"target_{m}"),
                    "attributes": env.active_records[m].get("attributes", [])
                }
                for m in range(env.num_targets)
            ]

        while not done:
            if record_this_ep:
                for i in range(env.num_drones):
                    drone_trails[i].append(np.copy(env.drone_pos[i]))
                for m in range(env.num_targets):
                    target_trails[m].append(np.copy(env.target_pos[m]))

            action_dict = policy_fn(obs_dict, env)
            obs_dict, _, terms, truncs, infos = env.step(action_dict)
            done = any(terms.values()) or any(truncs.values())

            info = infos["drone_0"]
            ep_tcr.append(info["tcr"])
            ep_overlap.append(info["overlap"])
            ep_col.append(info["collisions"])
            ep_ate.append(info["ate"])

            # Challenge attributes coverage breakdown
            dist_matrix = np.linalg.norm(env.drone_pos[:, None, :] - env.target_pos[None, :, :], axis=-1)
            covered_mask = np.any(dist_matrix <= env.sensor_radius, axis=0)
            for m in range(env.num_targets):
                rec = env.active_records[m]
                tags = rec.get("attributes", [])
                for tag in tags:
                    tag_total_counts[tag] += 1
                    if covered_mask[m]:
                        tag_covered_counts[tag] += 1

        all_tcr.append(float(np.mean(ep_tcr)))
        all_overlap.append(float(np.mean(ep_overlap)))
        all_col.append(int(np.sum(ep_col)))
        all_ate.append(float(np.mean(ep_ate)))

    tag_breakdown = {}
    for tag in sorted(tag_total_counts.keys()):
        tot = tag_total_counts[tag]
        cov = tag_covered_counts[tag]
        tag_breakdown[tag] = {
            "tcr": float(cov / max(1, tot)),
            "covered": int(cov),
            "total": int(tot)
        }

    results = {
        "mean_tcr": float(np.mean(all_tcr)),
        "mean_ate": float(np.mean(all_ate)),
        "mean_overlap": float(np.mean(all_overlap)),
        "total_collisions": int(np.sum(all_col)),
        "mean_collisions_per_ep": float(np.mean(all_col)),
        "tag_breakdown": tag_breakdown
    }

    if record_trail:
        return results, drone_trails, target_trails, trail_metadata
    return results


def plot_swarm_trajectory_trails(drone_trails, target_trails, trail_metadata, save_path="results/swarm_trajectory_trails.png"):
    """Generates publication-quality 2D arena trajectory trails with 25m sensor rings."""
    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")

    # Grid
    for g in range(0, 101, 10):
        ax.axvline(g, color="#1e293b", linestyle="--", lw=0.8, alpha=0.6)
        ax.axhline(g, color="#1e293b", linestyle="--", lw=0.8, alpha=0.6)

    drone_colors = ["#38bdf8", "#818cf8", "#c084fc"]
    target_colors = ["#22c55e", "#eab308", "#f97316", "#ef4444"]

    # Plot Target Trails
    for m in range(len(target_trails)):
        t_arr = np.array(target_trails[m])
        col = target_colors[m % len(target_colors)]
        meta = trail_metadata["records"][m]
        tag_str = ",".join(meta["attributes"][:3])
        label = f"Target-{m} ({meta['seq_id'][:12]}.. [{tag_str}])"

        ax.plot(t_arr[:, 0], t_arr[:, 1], color=col, lw=2.2, linestyle="-.", label=label, alpha=0.85)
        ax.scatter(t_arr[0, 0], t_arr[0, 1], color=col, s=70, marker="o", edgecolors="#ffffff", lw=1.2, zorder=5)
        ax.scatter(t_arr[-1, 0], t_arr[-1, 1], color=col, s=90, marker="s", edgecolors="#ffffff", lw=1.2, zorder=5)
        ax.text(t_arr[0, 0] + 1.2, t_arr[0, 1] + 1.2, f"T{m}_start", color=col, fontsize=9, fontweight="bold")
        ax.text(t_arr[-1, 0] + 1.2, t_arr[-1, 1] + 1.2, f"T{m}_end", color=col, fontsize=9, fontweight="bold")

    # Plot Drone Trails
    sensor_radius = 25.0
    for i in range(len(drone_trails)):
        d_arr = np.array(drone_trails[i])
        col = drone_colors[i % len(drone_colors)]

        ax.plot(d_arr[:, 0], d_arr[:, 1], color=col, lw=2.5, label=f"UAV-{i} Swarm Track", alpha=0.95)
        ax.scatter(d_arr[0, 0], d_arr[0, 1], color=col, s=80, marker="o", edgecolors="#ffffff", lw=1.5, zorder=6)
        ax.scatter(d_arr[-1, 0], d_arr[-1, 1], color=col, s=120, marker="*", edgecolors="#ffffff", lw=1.5, zorder=6)
        ax.text(d_arr[0, 0] - 3.5, d_arr[0, 1] - 3.0, f"UAV-{i}_start", color=col, fontsize=9, fontweight="bold")
        ax.text(d_arr[-1, 0] - 3.5, d_arr[-1, 1] + 2.0, f"UAV-{i}_end", color=col, fontsize=10, fontweight="bold")

        # 25m Sensor FOV Ring at final timestep
        fov_circle = plt.Circle(
            (d_arr[-1, 0], d_arr[-1, 1]),
            sensor_radius,
            color=col,
            fill=True,
            alpha=0.08,
            linestyle="--",
            lw=1.5,
            ec=col
        )
        ax.add_patch(fov_circle)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_title(
        "Anti-UAV-RGBT Cooperative Swarm: 200-Step Trajectory Trails & Task Splitting",
        color="#f8fafc",
        fontsize=13,
        fontweight="bold",
        pad=15
    )
    ax.set_xlabel("Arena X (meters)", color="#94a3b8", fontsize=11)
    ax.set_ylabel("Arena Y (meters)", color="#94a3b8", fontsize=11)
    ax.tick_params(colors="#94a3b8")

    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor="#1e293b", edgecolor="#334155", fontsize=9)
    for text in leg.get_texts():
        text.set_color("#f8fafc")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] High-resolution trajectory trails saved to {save_path}")


def compare_all_baselines(checkpoint_path="checkpoints/mappo_actor_best.pt", episodes=30):
    print(f"\n=======================================================")
    print(f"   FULL BASELINE COMPARISON SUITE ({episodes} SEEDED TEST EPISODES)")
    print(f"=======================================================")

    env = UAVRGBTTrackingEnv(num_drones=3, num_targets=4, split="test", render_mode=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Random Policy
    def random_policy(obs_dict, env):
        return {a: np.random.uniform(-1.0, 1.0, size=(2,)).astype(np.float32) for a in env.agents}

    print("\n[1/3] Evaluating Random Policy Baseline...")
    rand_metrics = run_policy_evaluation(env, random_policy, episodes=episodes)

    # 2. Greedy Nearest-Target Heuristic
    def greedy_policy(obs_dict, env):
        actions = {}
        for i, a in enumerate(env.agents):
            dists = np.linalg.norm(env.target_pos - env.drone_pos[i], axis=-1)
            closest_m = np.argmin(dists)
            diff = env.target_pos[closest_m] - env.drone_pos[i]
            norm = np.linalg.norm(diff) + 1e-6
            actions[a] = np.clip(diff / norm, -1.0, 1.0).astype(np.float32)
        return actions

    print("[2/3] Evaluating Uncoordinated Greedy Nearest-Target Heuristic...")
    greedy_metrics = run_policy_evaluation(env, greedy_policy, episodes=episodes)

    # 3. Trained MAPPO CTDE Swarm (Ours)
    actor = SharedActor(obs_dim=env.obs_dim, act_dim=2).to(device)
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "checkpoints/mappo_actor_latest.pt"
    actor.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    actor.eval()

    def mappo_policy(obs_dict, env):
        obs_array = np.stack([obs_dict[a] for a in env.agents]).astype(np.float32)
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=device)
        with torch.no_grad():
            act_t, _, _ = actor.get_action_and_logprob(obs_tensor, deterministic=True, update_norm=False)
        actions_np = act_t.cpu().numpy()
        return {a: actions_np[i] for i, a in enumerate(env.agents)}

    print(f"[3/3] Evaluating Trained MAPPO CTDE Swarm ({checkpoint_path})...")
    mappo_metrics, d_trails, t_trails, t_meta = run_policy_evaluation(
        env, mappo_policy, episodes=episodes, record_trail=True
    )

    # Generate swarm trajectory trail plot for episode 1
    plot_swarm_trajectory_trails(d_trails, t_trails, t_meta, save_path="results/swarm_trajectory_trails.png")

    env.close()

    # Formatted Comparison Table
    print("\n" + "=" * 78)
    print(f"{'Metric':<32} | {'Random Policy':<13} | {'Greedy Heuristic':<16} | {'MAPPO Swarm (Ours)':<16}")
    print("-" * 78)
    print(f"{'Target Coverage Ratio (TCR %)':<32} | {rand_metrics['mean_tcr']*100:11.2f}% | {greedy_metrics['mean_tcr']*100:14.2f}% | {mappo_metrics['mean_tcr']*100:14.2f}%")
    print(f"{'Avg Tracking Error (ATE m)':<32} | {rand_metrics['mean_ate']:11.2f}m | {greedy_metrics['mean_ate']:14.2f}m | {mappo_metrics['mean_ate']:14.2f}m")
    print(f"{'Redundant Overlaps / step':<32} | {rand_metrics['mean_overlap']:12.3f} | {greedy_metrics['mean_overlap']:15.3f} | {mappo_metrics['mean_overlap']:15.3f}")
    print(f"{'Inter-Drone Collisions / ep':<32} | {rand_metrics['mean_collisions_per_ep']:12.2f} | {greedy_metrics['mean_collisions_per_ep']:15.2f} | {mappo_metrics['mean_collisions_per_ep']:15.2f}")
    print("=" * 78)

    # Per-Attribute Comparison Table
    all_tags = sorted(list(mappo_metrics["tag_breakdown"].keys()))
    print(f"\n--- Challenge Attribute Coverage Breakdown (TCR %) ---")
    print(f"{'Attribute Tag':<14} | {'Random':<10} | {'Greedy':<10} | {'MAPPO (Ours)':<12}")
    print("-" * 52)
    for tag in all_tags:
        r_tcr = rand_metrics["tag_breakdown"].get(tag, {}).get("tcr", 0.0) * 100.0
        g_tcr = greedy_metrics["tag_breakdown"].get(tag, {}).get("tcr", 0.0) * 100.0
        m_tcr = mappo_metrics["tag_breakdown"].get(tag, {}).get("tcr", 0.0) * 100.0
        print(f"{tag:<14} | {r_tcr:8.1f}% | {g_tcr:8.1f}% | {m_tcr:10.1f}%")
    print("=" * 52)

    # Save JSON benchmark metrics
    full_benchmark = {
        "random_policy": rand_metrics,
        "greedy_heuristic": greedy_metrics,
        "mappo_ctde_swarm": mappo_metrics
    }
    json_path = "results/final_benchmark_metrics.json"
    with open(json_path, "w") as f:
        json.dump(full_benchmark, f, indent=2)
    print(f"\n[SAVED] Quantitative metrics saved to {json_path}")

    # Generate 3-Panel Publication Comparison Bar Chart
    methods = ["Random\nPolicy", "Greedy\nHeuristic", "MAPPO Swarm\n(Ours)"]
    tcrs = [rand_metrics["mean_tcr"] * 100.0, greedy_metrics["mean_tcr"] * 100.0, mappo_metrics["mean_tcr"] * 100.0]
    ates = [rand_metrics["mean_ate"], greedy_metrics["mean_ate"], mappo_metrics["mean_ate"]]
    cols = [rand_metrics["mean_collisions_per_ep"], greedy_metrics["mean_collisions_per_ep"], mappo_metrics["mean_collisions_per_ep"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=200)
    bar_colors = ["#94a3b8", "#f59e0b", "#10b981"]

    # 1. TCR
    b1 = axes[0].bar(methods, tcrs, color=bar_colors, width=0.55, edgecolor="#0f172a", lw=1.2)
    axes[0].set_title("Target Coverage Ratio (TCR %)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Coverage (%)", fontsize=10)
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")
    for bar in b1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 2. ATE
    b2 = axes[1].bar(methods, ates, color=bar_colors, width=0.55, edgecolor="#0f172a", lw=1.2)
    axes[1].set_title("Average Tracking Error (ATE m)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Error (meters)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")
    for bar in b2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, yval + 0.8, f"{yval:.1f}m", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 3. Collisions
    b3 = axes[2].bar(methods, cols, color=bar_colors, width=0.55, edgecolor="#0f172a", lw=1.2)
    axes[2].set_title("Inter-Drone Collisions / Episode", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Collisions / Ep", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.5, axis="y")
    for bar in b3:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle("Anti-UAV-RGBT Benchmark: MAPPO Swarm vs. Heuristic Baselines", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    chart_path = "results/baseline_comparison.png"
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Publication comparison chart saved to {chart_path}")


def evaluate_test_split(checkpoint_path="checkpoints/mappo_actor_best.pt", episodes=20, render=False, save_gif=False):
    if not os.path.exists(checkpoint_path):
        fallback = "checkpoints/mappo_actor_latest.pt"
        if os.path.exists(fallback):
            print(f"[INFO] '{checkpoint_path}' not found. Falling back to '{fallback}'.")
            checkpoint_path = fallback
        else:
            raise FileNotFoundError(f"Neither '{checkpoint_path}' nor '{fallback}' found. Train a model first!")

    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluating with device: {device} | Checkpoint: {checkpoint_path}")

    render_mode = "human" if render else ("rgb_array" if save_gif else None)
    env = UAVRGBTTrackingEnv(num_drones=3, num_targets=4, split="test", render_mode=render_mode)

    actor = SharedActor(obs_dim=env.obs_dim, act_dim=2).to(device)
    actor.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    actor.eval()

    all_tcr, all_overlap, all_col, all_ate = [], [], [], []
    tag_total_counts = defaultdict(int)
    tag_covered_counts = defaultdict(int)

    recorded_frames = []
    snapshot_steps = [1, 70, 140, 200]
    snapshots = {}

    print(f"\n=== Evaluating Trained MAPPO Swarm on Held-Out Anti-UAV-RGBT TEST Split ({episodes} Episodes) ===")

    for ep in range(episodes):
        obs_dict, info_dict = env.reset(seed=5000 + ep)
        done = False
        ep_tcr, ep_overlap, ep_col, ep_ate = [], [], [], []
        record_this_ep = save_gif and (ep == 0)

        step_num = 0
        while not done:
            step_num += 1
            obs_array = np.stack([obs_dict[a] for a in env.agents]).astype(np.float32)
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=device)

            with torch.no_grad():
                actions_t, _, _ = actor.get_action_and_logprob(obs_tensor, deterministic=True, update_norm=False)

            actions_np = actions_t.cpu().numpy()
            action_dict = {a: actions_np[i] for i, a in enumerate(env.agents)}
            obs_dict, _, terms, truncs, infos = env.step(action_dict)
            done = any(terms.values()) or any(truncs.values())

            if record_this_ep:
                frame = env.render()
                if frame is not None:
                    recorded_frames.append(frame)
                    if step_num in snapshot_steps:
                        snapshots[step_num] = np.copy(frame)

            info = infos["drone_0"]
            ep_tcr.append(info["tcr"])
            ep_overlap.append(info["overlap"])
            ep_col.append(info["collisions"])
            ep_ate.append(info["ate"])

            dist_matrix = np.linalg.norm(env.drone_pos[:, None, :] - env.target_pos[None, :, :], axis=-1)
            covered_mask = np.any(dist_matrix <= env.sensor_radius, axis=0)
            for m in range(env.num_targets):
                rec = env.active_records[m]
                tags = rec.get("attributes", [])
                for tag in tags:
                    tag_total_counts[tag] += 1
                    if covered_mask[m]:
                        tag_covered_counts[tag] += 1

        all_tcr.append(float(np.mean(ep_tcr)))
        all_overlap.append(float(np.mean(ep_overlap)))
        all_col.append(int(np.sum(ep_col)))
        all_ate.append(float(np.mean(ep_ate)))

        print(
            f"Test Episode {ep + 1:2d} | Coverage (TCR): {np.mean(ep_tcr):6.2%} | "
            f"ATE: {np.mean(ep_ate):4.1f}m | Redundant Overlaps: {np.mean(ep_overlap):.2f} | Collisions: {int(np.sum(ep_col)):2d}"
        )

    env.close()

    print("\n" + "=" * 50)
    print("      HELD-OUT TEST SPLIT OVERALL SUMMARY")
    print("=" * 50)
    print(f"Mean Target Coverage Ratio (TCR) : {np.mean(all_tcr):.2%}")
    print(f"Mean Average Tracking Error (ATE): {np.mean(all_ate):.2f} m")
    print(f"Mean Redundancy / Overlap Rate   : {np.mean(all_overlap):.3f}")
    print(f"Total Inter-Drone Collisions     : {int(np.sum(all_col))}")
    print(f"Average Collisions / Episode     : {np.mean(all_col):.2f}")

    if tag_total_counts:
        print("\n--- Challenge Attribute Breakdown (Target Coverage Ratio) ---")
        for tag in sorted(tag_total_counts.keys()):
            tot = tag_total_counts[tag]
            cov = tag_covered_counts[tag]
            rate = cov / max(1, tot)
            print(f"  Tag [{tag:10s}]: {rate:6.2%} ({cov:,} / {tot:,} target-steps)")
    print("=" * 50)

    if save_gif and recorded_frames:
        gif_path = "results/test_episode_preview.gif"
        png_path = "results/test_episode_preview.png"

        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), dpi=200)
        for idx, s in enumerate(snapshot_steps):
            if s in snapshots:
                axes[idx].imshow(snapshots[s])
            elif recorded_frames:
                f_idx = min(s - 1, len(recorded_frames) - 1)
                axes[idx].imshow(recorded_frames[f_idx])
            axes[idx].set_title(f"Episode Timestep {s}/200", fontsize=12, fontweight="bold")
            axes[idx].axis("off")

        plt.suptitle("Anti-UAV-RGBT MARL Swarm: Multi-Target Cooperative Tracking Progression", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close()

        img_frames = [Image.fromarray(f).resize((370, 370), Image.Resampling.LANCZOS) for f in recorded_frames[::2]]
        if img_frames:
            img_frames[0].save(gif_path, save_all=True, append_images=img_frames[1:], duration=60, loop=0)

        print(f"\n[SAVED] 4-panel progression snapshot saved to {png_path}")
        print(f"[SAVED] Animated test episode saved to {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate & Benchmark MAPPO Swarm on Anti-UAV-RGBT")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mappo_actor_best.pt", help="Path to actor checkpoint")
    parser.add_argument("--episodes", type=int, default=30, help="Number of test episodes")
    parser.add_argument("--compare-baselines", action="store_true", help="Run full 3-policy baseline comparison suite")
    parser.add_argument("--render", action="store_true", help="Launch live Pygame visualizer")
    parser.add_argument("--save-gif", action="store_true", help="Record and save episode preview GIF/PNG")
    args = parser.parse_args()

    if args.compare_baselines:
        compare_all_baselines(checkpoint_path=args.checkpoint, episodes=args.episodes)
    else:
        num_eps = args.episodes if not args.render else (args.episodes if args.episodes != 30 else 5)
        evaluate_test_split(checkpoint_path=args.checkpoint, episodes=num_eps, render=args.render, save_gif=args.save_gif)
