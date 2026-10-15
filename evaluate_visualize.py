# evaluate_visualize.py
import os
import sys
import csv
import json
import argparse
from collections import defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from envs.uav_rgbt_env import UAVRGBTTrackingEnv
from algos.networks import SharedActor


# ---------------------------------------------------------------------------
# Strict Observation-Based Baseline Policies (Fixing AUD-04)
# ---------------------------------------------------------------------------

def random_policy(obs_dict, env=None, rng=None):
    """Uniform random continuous acceleration in [-1.0, 1.0]^2.
    
    Operates strictly on obs_dict keys without reading ground-truth arena state.
    Supports deterministic evaluation when seeded rng is provided.
    """
    if rng is None:
        rng = np.random.default_rng()
    return {a: rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32) for a in obs_dict}


def greedy_nearest_policy(obs_dict, env=None):
    """Uncoordinated Greedy Nearest-Target Pursuit.
    
    Operates strictly on the 39D local observation vector:
      obs[0:2]   : own position / 100.0
      obs[2:4]   : own velocity / 4.0
      obs[12:36] : 4 targets (each 6D: rel_pos / 100.0, rel_vel / 4.0, rgb_flag, ir_flag)
    Picks the nearest observed visible target and applies PD pursuit.
    If all targets are occluded, gently damps toward arena center.
    """
    actions = {}
    for a, obs in obs_dict.items():
        own_pos = obs[0:2]
        target_feats = obs[12:36].reshape(4, 6)
        visible_mask = np.any(target_feats[:, :2] != 0.0, axis=-1) | (target_feats[:, 4] > 0) | (target_feats[:, 5] > 0)

        if not np.any(visible_mask):
            center_dir = np.array([0.5, 0.5], dtype=np.float32) - own_pos
            actions[a] = np.clip(center_dir * 2.0, -1.0, 1.0).astype(np.float32)
            continue

        rel_pos = target_feats[:, :2]
        rel_vel = target_feats[:, 2:4]
        dists = np.where(visible_mask, np.linalg.norm(rel_pos, axis=-1), 1e6)
        best_m = np.argmin(dists)

        # PD pursuit: target relative position + velocity damping
        pursuit_vec = rel_pos[best_m] * 100.0 + 0.25 * (rel_vel[best_m] * 4.0)
        norm = np.linalg.norm(pursuit_vec) + 1e-6
        actions[a] = np.clip(pursuit_vec / norm, -1.0, 1.0).astype(np.float32)

    return actions


def collision_aware_greedy_policy(obs_dict, env=None):
    """Collision-Aware Greedy Heuristic (Stronger Baseline 1).
    
    Combines nearest-target PD pursuit from obs[12:36] with repulsive avoidance
    from observed teammates (obs[4:6] and obs[8:10]) when closer than 6.0 sim_units (0.06 norm).
    Operates strictly on 39D local observations without oracle coordinates.
    """
    actions = {}
    for a, obs in obs_dict.items():
        own_pos = obs[0:2]
        target_feats = obs[12:36].reshape(4, 6)
        visible_mask = np.any(target_feats[:, :2] != 0.0, axis=-1) | (target_feats[:, 4] > 0) | (target_feats[:, 5] > 0)

        if not np.any(visible_mask):
            base_vec = (np.array([0.5, 0.5], dtype=np.float32) - own_pos) * 2.0
        else:
            rel_pos = target_feats[:, :2]
            rel_vel = target_feats[:, 2:4]
            dists = np.where(visible_mask, np.linalg.norm(rel_pos, axis=-1), 1e6)
            best_m = np.argmin(dists)
            pursuit_vec = rel_pos[best_m] * 100.0 + 0.25 * (rel_vel[best_m] * 4.0)
            norm = np.linalg.norm(pursuit_vec) + 1e-6
            base_vec = pursuit_vec / norm

        # Teammate repulsive separation (obs[4:6] and obs[8:10])
        tm1_rel = obs[4:6]
        tm2_rel = obs[8:10]
        repulsion = np.zeros(2, dtype=np.float32)
        safe_margin = 0.06  # 6.0 sim_units

        for tm_rel in [tm1_rel, tm2_rel]:
            tm_dist = np.linalg.norm(tm_rel)
            if 0.0 < tm_dist < safe_margin:
                repulsion += - (tm_rel / (tm_dist + 1e-6)) * (safe_margin - tm_dist) / safe_margin

        combined = base_vec + 2.5 * repulsion
        actions[a] = np.clip(combined, -1.0, 1.0).astype(np.float32)

    return actions


def target_assignment_heuristic_policy(obs_dict, env=None):
    """Observation-Based Bipartite Target Assignment Heuristic (Stronger Baseline 2).
    
    Reconstructs observed targets and teammate positions purely from obs_dict,
    performs greedy bipartite min-distance assignment so drones track distinct targets,
    and adds teammate collision avoidance within 6.0 sim_units.
    """
    agents = list(obs_dict.keys())
    drone_positions = {}
    target_world_estimates = defaultdict(list)

    # 1. Reconstruct visible target world positions from local observations
    for a in agents:
        obs = obs_dict[a]
        own_pos = obs[0:2] * 100.0
        drone_positions[a] = own_pos
        target_feats = obs[12:36].reshape(4, 6)
        for m in range(4):
            is_vis = (np.linalg.norm(target_feats[m, :2]) > 0.0) or (target_feats[m, 4] > 0) or (target_feats[m, 5] > 0)
            if is_vis:
                tgt_world = own_pos + target_feats[m, :2] * 100.0
                target_world_estimates[m].append(tgt_world)

    # Centroid of observed targets
    consensus_targets = {}
    for m, est_list in target_world_estimates.items():
        if est_list:
            consensus_targets[m] = np.mean(est_list, axis=0)

    # 2. Greedy bipartite matching: assign distinct targets to drones
    assigned_targets = {}
    unassigned_drones = list(agents)
    available_target_indices = list(consensus_targets.keys())

    # Build cost pairs (distance, drone, target_m)
    cost_pairs = []
    for a in unassigned_drones:
        d_pos = drone_positions[a]
        for m in available_target_indices:
            dist = float(np.linalg.norm(d_pos - consensus_targets[m]))
            cost_pairs.append((dist, a, m))

    cost_pairs.sort(key=lambda x: x[0])
    used_targets = set()
    used_drones = set()

    for dist, a, m in cost_pairs:
        if a not in used_drones and m not in used_targets:
            assigned_targets[a] = m
            used_drones.add(a)
            used_targets.add(m)

    # 3. Generate actions with collision repulsion
    actions = {}
    for a in agents:
        obs = obs_dict[a]
        own_pos = drone_positions[a]

        if a in assigned_targets:
            tgt_m = assigned_targets[a]
            tgt_world = consensus_targets[tgt_m]
            pursuit_vec = tgt_world - own_pos
            norm = np.linalg.norm(pursuit_vec) + 1e-6
            base_vec = pursuit_vec / norm
        else:
            # Fallback to nearest local target or center
            target_feats = obs[12:36].reshape(4, 6)
            visible_mask = np.any(target_feats[:, :2] != 0.0, axis=-1)
            if np.any(visible_mask):
                dists = np.linalg.norm(target_feats[:, :2], axis=-1)
                best_m = np.argmin(dists)
                base_vec = target_feats[best_m, :2] / (dists[best_m] + 1e-6)
            else:
                base_vec = (np.array([50.0, 50.0]) - own_pos) / 50.0

        # Teammate repulsion within 6.0 sim_units (0.06 norm)
        tm1_rel = obs[4:6]
        tm2_rel = obs[8:10]
        repulsion = np.zeros(2, dtype=np.float32)
        safe_margin = 0.06

        for tm_rel in [tm1_rel, tm2_rel]:
            tm_dist = np.linalg.norm(tm_rel)
            if 0.0 < tm_dist < safe_margin:
                repulsion += - (tm_rel / (tm_dist + 1e-6)) * (safe_margin - tm_dist) / safe_margin

        combined = base_vec + 2.5 * repulsion
        actions[a] = np.clip(combined, -1.0, 1.0).astype(np.float32)

    return actions


# ---------------------------------------------------------------------------
# Evaluation Framework with Mean ± Std & Export
# ---------------------------------------------------------------------------

def run_policy_evaluation(env, policy_fn, episodes=30, record_trail=False):
    """Evaluates a policy over seeded episodes and computes Mean ± Std metrics."""
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
        ep_rng = np.random.default_rng(5000 + ep)
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

            try:
                action_dict = policy_fn(obs_dict, env=env, rng=ep_rng)
            except TypeError:
                action_dict = policy_fn(obs_dict, env=env)

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
        "std_tcr": float(np.std(all_tcr)),
        "mean_ate": float(np.mean(all_ate)),
        "std_ate": float(np.std(all_ate)),
        "mean_overlap": float(np.mean(all_overlap)),
        "std_overlap": float(np.std(all_overlap)),
        "total_collisions": int(np.sum(all_col)),
        "mean_collisions_per_ep": float(np.mean(all_col)),
        "std_collisions_per_ep": float(np.std(all_col)),
        "tag_breakdown": tag_breakdown
    }

    if record_trail:
        return results, drone_trails, target_trails, trail_metadata
    return results


def plot_swarm_trajectory_trails(drone_trails, target_trails, trail_metadata, save_path="results/swarm_trajectory_trails.png"):
    """Visualizes 2D flight paths in dimensionless simulation units (sim_units)."""
    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    ax.set_facecolor("#0b0f19")
    fig.patch.set_facecolor("#020617")

    drone_colors = ["#38bdf8", "#4ade80", "#f43f5e"]
    target_colors = ["#fbbf24", "#a855f7", "#ec4899", "#06b6d4"]

    # 1. Target Trails
    for m in range(len(target_trails)):
        pts = np.array(target_trails[m])
        if len(pts) == 0:
            continue
        c = target_colors[m % len(target_colors)]
        meta = trail_metadata.get("records", [])
        tags = meta[m].get("attributes", []) if m < len(meta) else []
        tag_str = f" [{','.join(tags)}]" if tags else ""
        ax.plot(pts[:, 0], pts[:, 1], color=c, lw=2.2, linestyle="--", alpha=0.9, label=f"Target {m}{tag_str}")
        ax.scatter(pts[0, 0], pts[0, 1], color=c, marker="o", s=80, edgecolors="white", lw=1.5, zorder=5)
        ax.scatter(pts[-1, 0], pts[-1, 1], color=c, marker="*", s=160, edgecolors="white", lw=1.5, zorder=5)

    # 2. Drone Trails
    for i in range(len(drone_trails)):
        pts = np.array(drone_trails[i])
        if len(pts) == 0:
            continue
        c = drone_colors[i % len(drone_colors)]
        ax.plot(pts[:, 0], pts[:, 1], color=c, lw=2.8, alpha=0.95, label=f"Drone {i} (MAPPO)")
        ax.scatter(pts[0, 0], pts[0, 1], color=c, marker="s", s=80, edgecolors="white", lw=1.5, zorder=5)
        ax.scatter(pts[-1, 0], pts[-1, 1], color=c, marker="^", s=140, edgecolors="white", lw=1.5, zorder=5)

        # Draw final FOV radius (25 sim_units)
        fov_circle = plt.Circle(
            (pts[-1, 0], pts[-1, 1]),
            25.0,
            color=c,
            fill=True,
            alpha=0.08,
            linestyle=":",
            lw=1.5
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
    ax.set_xlabel("Simulated 2D Arena X (sim_units)", color="#94a3b8", fontsize=11)
    ax.set_ylabel("Simulated 2D Arena Y (sim_units)", color="#94a3b8", fontsize=11)
    ax.tick_params(colors="#94a3b8")

    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor="#1e293b", edgecolor="#334155", fontsize=9)
    for text in leg.get_texts():
        text.set_color("#f8fafc")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] High-resolution trajectory trails saved to {save_path}")


def compare_all_baselines(checkpoint_paths=None, episodes=30):
    """Evaluates 5 observation-based policies across seeded episodes and reports Mean ± Std."""
    if checkpoint_paths is None:
        checkpoint_paths = ["checkpoints/mappo_actor_best.pt"]
    elif isinstance(checkpoint_paths, str):
        checkpoint_paths = [checkpoint_paths]

    print("\n" + "=" * 80)
    print(f"   5-POLICY OBSERVATION-ONLY BENCHMARK SUITE ({episodes} SEEDED TEST EPISODES)")
    print("=" * 80)

    env = UAVRGBTTrackingEnv(num_drones=3, num_targets=4, split="test", render_mode=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_benchmarks = {}

    # 1. Random Policy
    print("\n[1/5] Evaluating Random Policy Baseline...")
    rand_metrics = run_policy_evaluation(env, random_policy, episodes=episodes)
    all_benchmarks["random_policy"] = rand_metrics

    # 2. Greedy Nearest-Target Heuristic
    print("[2/5] Evaluating Uncoordinated Greedy Nearest-Target Heuristic...")
    greedy_metrics = run_policy_evaluation(env, greedy_nearest_policy, episodes=episodes)
    all_benchmarks["greedy_nearest"] = greedy_metrics

    # 3. Collision-Aware Greedy
    print("[3/5] Evaluating Collision-Aware Greedy Heuristic...")
    col_greedy_metrics = run_policy_evaluation(env, collision_aware_greedy_policy, episodes=episodes)
    all_benchmarks["collision_aware_greedy"] = col_greedy_metrics

    # 4. Target Assignment Heuristic
    print("[4/5] Evaluating Target Assignment Bipartite Matching Heuristic...")
    assign_metrics = run_policy_evaluation(env, target_assignment_heuristic_policy, episodes=episodes)
    all_benchmarks["target_assignment_heuristic"] = assign_metrics

    # 5. Trained MAPPO CTDE Swarm
    mappo_eval_list = []
    last_d_trails, last_t_trails, last_t_meta = None, None, None

    for idx, ckpt in enumerate(checkpoint_paths):
        if not os.path.exists(ckpt):
            fallback = "checkpoints/mappo_actor_latest.pt"
            if os.path.exists(fallback):
                ckpt = fallback
            else:
                print(f"[WARNING] Checkpoint {ckpt} not found, skipping.")
                continue

        print(f"[5/5] Evaluating MAPPO Swarm Checkpoint {idx+1}/{len(checkpoint_paths)} ({ckpt})...")
        actor = SharedActor(obs_dim=env.obs_dim, act_dim=2).to(device)
        actor.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        actor.eval()

        def mappo_policy(obs_dict, env=None, rng=None):
            obs_array = np.stack([obs_dict[a] for a in env.agents]).astype(np.float32)
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=device)
            with torch.no_grad():
                act_t, _, _ = actor.get_action_and_logprob(obs_tensor, deterministic=True, update_norm=False)
            actions_np = act_t.cpu().numpy()
            return {a: actions_np[i] for i, a in enumerate(env.agents)}

        m_metrics, d_trails, t_trails, t_meta = run_policy_evaluation(
            env, mappo_policy, episodes=episodes, record_trail=True
        )
        mappo_eval_list.append(m_metrics)
        last_d_trails, last_t_trails, last_t_meta = d_trails, t_trails, t_meta

    if mappo_eval_list:
        # Aggregate across checkpoints if multiple provided
        agg_mappo = {
            "mean_tcr": float(np.mean([m["mean_tcr"] for m in mappo_eval_list])),
            "std_tcr": float(np.mean([m["std_tcr"] for m in mappo_eval_list])),
            "mean_ate": float(np.mean([m["mean_ate"] for m in mappo_eval_list])),
            "std_ate": float(np.mean([m["std_ate"] for m in mappo_eval_list])),
            "mean_overlap": float(np.mean([m["mean_overlap"] for m in mappo_eval_list])),
            "std_overlap": float(np.mean([m["std_overlap"] for m in mappo_eval_list])),
            "total_collisions": int(np.mean([m["total_collisions"] for m in mappo_eval_list])),
            "mean_collisions_per_ep": float(np.mean([m["mean_collisions_per_ep"] for m in mappo_eval_list])),
            "std_collisions_per_ep": float(np.mean([m["std_collisions_per_ep"] for m in mappo_eval_list])),
            "tag_breakdown": mappo_eval_list[0]["tag_breakdown"]
        }
        all_benchmarks["mappo_ctde_swarm"] = agg_mappo
        if last_d_trails is not None:
            plot_swarm_trajectory_trails(last_d_trails, last_t_trails, last_t_meta, save_path="results/swarm_trajectory_trails.png")
    else:
        print("[ERROR] No valid MAPPO checkpoints could be evaluated.")

    env.close()

    # Save quantitative JSON metrics
    json_path = "results/final_benchmark_metrics.json"
    with open(json_path, "w") as f:
        json.dump({
            "metadata": {
                "episodes": episodes,
                "max_steps": 200,
                "seed_range": [5000, 5000 + episodes - 1],
                "units": "dimensionless simulation units (sim_units)"
            },
            "benchmarks": all_benchmarks
        }, f, indent=2)
    print(f"\n[SAVED] Benchmark metrics saved to {json_path}")

    # Save CSV metrics
    csv_path = "results/final_benchmark_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Policy",
            "TCR_Mean_Pct", "TCR_Std_Pct",
            "ATE_Mean_sim_units", "ATE_Std_sim_units",
            "Overlap_Mean", "Overlap_Std",
            "Collisions_Mean_per_ep", "Collisions_Std_per_ep",
            "Total_Collisions"
        ])
        for p_name, m in all_benchmarks.items():
            writer.writerow([
                p_name,
                f"{m['mean_tcr']*100:.2f}", f"{m['std_tcr']*100:.2f}",
                f"{m['mean_ate']:.2f}", f"{m['std_ate']:.2f}",
                f"{m['mean_overlap']:.3f}", f"{m['std_overlap']:.3f}",
                f"{m['mean_collisions_per_ep']:.2f}", f"{m['std_collisions_per_ep']:.2f}",
                m["total_collisions"]
            ])
    print(f"[SAVED] Tabular CSV metrics saved to {csv_path}")

    # Formatted Comparison Table
    print("\n" + "=" * 105)
    headers = f"{'Metric':<30} | {'Random':<12} | {'Greedy':<12} | {'Col-Greedy':<12} | {'Target-Assign':<14} | {'MAPPO (Ours)':<14}"
    print(headers)
    print("-" * 105)

    def fmt_stat(m, key, mult=1.0, fmt=".1f"):
        mean_v = m[f"mean_{key}"] * mult
        std_v = m[f"std_{key}"] * mult
        return f"{mean_v:{fmt}} +/- {std_v:{fmt}}"

    print(f"{'Target Coverage (TCR %)':<30} | {fmt_stat(rand_metrics, 'tcr', 100.0, '.1f'):<12} | {fmt_stat(greedy_metrics, 'tcr', 100.0, '.1f'):<12} | {fmt_stat(col_greedy_metrics, 'tcr', 100.0, '.1f'):<12} | {fmt_stat(assign_metrics, 'tcr', 100.0, '.1f'):<14} | {fmt_stat(all_benchmarks.get('mappo_ctde_swarm', rand_metrics), 'tcr', 100.0, '.1f'):<14}")
    print(f"{'Tracking Error (ATE sim_units)':<30} | {fmt_stat(rand_metrics, 'ate', 1.0, '.1f'):<12} | {fmt_stat(greedy_metrics, 'ate', 1.0, '.1f'):<12} | {fmt_stat(col_greedy_metrics, 'ate', 1.0, '.1f'):<12} | {fmt_stat(assign_metrics, 'ate', 1.0, '.1f'):<14} | {fmt_stat(all_benchmarks.get('mappo_ctde_swarm', rand_metrics), 'ate', 1.0, '.1f'):<14}")
    print(f"{'Redundant Overlaps / step':<30} | {fmt_stat(rand_metrics, 'overlap', 1.0, '.2f'):<12} | {fmt_stat(greedy_metrics, 'overlap', 1.0, '.2f'):<12} | {fmt_stat(col_greedy_metrics, 'overlap', 1.0, '.2f'):<12} | {fmt_stat(assign_metrics, 'overlap', 1.0, '.2f'):<14} | {fmt_stat(all_benchmarks.get('mappo_ctde_swarm', rand_metrics), 'overlap', 1.0, '.2f'):<14}")
    print(f"{'Inter-Drone Collisions / ep':<30} | {fmt_stat(rand_metrics, 'collisions_per_ep', 1.0, '.2f'):<12} | {fmt_stat(greedy_metrics, 'collisions_per_ep', 1.0, '.2f'):<12} | {fmt_stat(col_greedy_metrics, 'collisions_per_ep', 1.0, '.2f'):<12} | {fmt_stat(assign_metrics, 'collisions_per_ep', 1.0, '.2f'):<14} | {fmt_stat(all_benchmarks.get('mappo_ctde_swarm', rand_metrics), 'collisions_per_ep', 1.0, '.2f'):<14}")
    print("=" * 105)

    # 3-Panel Publication-Quality Comparison Bar Chart with Error Bars
    policy_labels = ["Random", "Greedy", "Col-Greedy", "Target-Assign", "MAPPO Swarm\n(Ours)"]
    metric_keys = ["random_policy", "greedy_nearest", "collision_aware_greedy", "target_assignment_heuristic", "mappo_ctde_swarm"]
    bar_colors = ["#94a3b8", "#f59e0b", "#3b82f6", "#8b5cf6", "#10b981"]

    tcr_means = [all_benchmarks[k]["mean_tcr"] * 100.0 for k in metric_keys if k in all_benchmarks]
    tcr_stds = [all_benchmarks[k]["std_tcr"] * 100.0 for k in metric_keys if k in all_benchmarks]

    ate_means = [all_benchmarks[k]["mean_ate"] for k in metric_keys if k in all_benchmarks]
    ate_stds = [all_benchmarks[k]["std_ate"] for k in metric_keys if k in all_benchmarks]

    col_means = [all_benchmarks[k]["mean_collisions_per_ep"] for k in metric_keys if k in all_benchmarks]
    col_stds = [all_benchmarks[k]["std_collisions_per_ep"] for k in metric_keys if k in all_benchmarks]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.0), dpi=200)

    # 1. TCR
    b1 = axes[0].bar(policy_labels[:len(tcr_means)], tcr_means, yerr=tcr_stds, capsize=4, color=bar_colors[:len(tcr_means)], width=0.55, edgecolor="#0f172a", lw=1.2)
    axes[0].set_title("Target Coverage Ratio (TCR %)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Coverage (%)", fontsize=10)
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")
    for bar in b1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, max(2.0, yval - 7.0), f"{yval:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="white")

    # 2. ATE
    b2 = axes[1].bar(policy_labels[:len(ate_means)], ate_means, yerr=ate_stds, capsize=4, color=bar_colors[:len(ate_means)], width=0.55, edgecolor="#0f172a", lw=1.2)
    axes[1].set_title("Average Tracking Error (ATE sim_units)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Tracking Error (sim_units)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")
    for bar in b2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, max(1.0, yval - 5.0), f"{yval:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="white")

    # 3. Collisions
    b3 = axes[2].bar(policy_labels[:len(col_means)], col_means, yerr=col_stds, capsize=4, color=bar_colors[:len(col_means)], width=0.55, edgecolor="#0f172a", lw=1.2)
    axes[2].set_title("Inter-Drone Collisions / Episode", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Collisions / Ep", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.5, axis="y")
    for bar in b3:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle("Anti-UAV-RGBT Benchmark: MAPPO Swarm vs. 4 Observation-Only Baselines", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    chart_path = "results/baseline_comparison.png"
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Publication comparison chart saved to {chart_path}")


def evaluate_test_split(checkpoint_path="checkpoints/mappo_actor_best.pt", episodes=20, render=False, save_gif=False):
    """Evaluates a single policy checkpoint deterministically on held-out test split."""
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
            f"ATE: {np.mean(ep_ate):4.1f} sim_units | Redundant Overlaps: {np.mean(ep_overlap):.2f} | Collisions: {int(np.sum(ep_col)):2d}"
        )

    env.close()

    print("\n" + "=" * 50)
    print("      HELD-OUT TEST SPLIT OVERALL SUMMARY")
    print("=" * 50)
    print(f"Mean Target Coverage Ratio (TCR) : {np.mean(all_tcr):.2%} ± {np.std(all_tcr):.2%}")
    print(f"Mean Average Tracking Error (ATE): {np.mean(all_ate):.2f} ± {np.std(all_ate):.2f} sim_units")
    print(f"Mean Redundancy / Overlap Rate   : {np.mean(all_overlap):.3f} ± {np.std(all_overlap):.3f}")
    print(f"Total Inter-Drone Collisions     : {int(np.sum(all_col))}")
    print(f"Average Collisions / Episode     : {np.mean(all_col):.2f} ± {np.std(all_col):.2f}")

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
    parser.add_argument("--checkpoints", nargs="*", default=None, help="Optional multiple actor checkpoints for cross-seed benchmark")
    parser.add_argument("--episodes", type=int, default=30, help="Number of test episodes")
    parser.add_argument("--compare-baselines", action="store_true", help="Run full 5-policy baseline comparison suite")
    parser.add_argument("--render", action="store_true", help="Launch live Pygame visualizer")
    parser.add_argument("--save-gif", action="store_true", help="Record and save episode preview GIF/PNG")
    args = parser.parse_args()

    ckpts = args.checkpoints if args.checkpoints else [args.checkpoint]

    if args.compare_baselines:
        compare_all_baselines(checkpoint_paths=ckpts, episodes=args.episodes)
    else:
        num_eps = args.episodes if not args.render else (args.episodes if args.episodes != 30 else 5)
        evaluate_test_split(checkpoint_path=args.checkpoint, episodes=num_eps, render=args.render, save_gif=args.save_gif)
