import json
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_env import Config

matplotlib.use("Agg")


class EnvControllerPolicy:
    def __init__(self, controller):
        self.controller = controller

    def select_actions(self, _obs_n):
        return self.controller.select_actions()

    def reset(self):
        if hasattr(self.controller, "reset"):
            self.controller.reset()


class MADDPGPolicy:
    def __init__(self, controller):
        self.controller = controller

    def select_actions(self, obs_n):
        return self.controller.select_actions(obs_n, noise_std=0.0)

    def reset(self):
        return None


class IDDPGPolicy:
    def __init__(self, controller):
        self.controller = controller

    def select_actions(self, obs_n):
        return self.controller.select_actions(obs_n, noise_std=0.0)

    def reset(self):
        return None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def avg(values):
    return float(np.mean(values)) if values else 0.0


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def summarize_metrics(metrics):
    return {
        "min_all_detect_step": avg(metrics["min_all_detect_steps"]),
        "total_detection_count": avg(metrics["total_detection_counts"]),
        "overlap_rate": avg(metrics["overlap_rates"]),
        "collision_count": avg(metrics["collision_counts"]),
        "coverage_efficiency": avg(metrics["coverage_efficiencies"]),
    }


def save_metrics(result_dir, metrics_map):
    ensure_dir(result_dir)
    summary = {name: summarize_metrics(metrics) for name, metrics in metrics_map.items()}
    save_json(os.path.join(result_dir, "metrics_raw.json"), metrics_map)
    save_json(os.path.join(result_dir, "metrics_summary.json"), summary)
    return summary


def sample_eval_seeds(eval_seeds, sample_size=10):
    seeds = list(eval_seeds)
    if len(seeds) <= sample_size:
        return seeds
    return random.sample(seeds, sample_size)


def _run_episode(env, policy, seed, max_steps):
    obs_n, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    for _ in range(max_steps):
        actions = policy.select_actions(obs_n)
        obs_n, _, terminated, truncated, _ = env.step(actions)
        if any(terminated) or any(truncated):
            break
    return env.uav_trajectories, env.target_trajectories, env.obstacles


def plot_trajectories(uav_trajs, target_trajs, obstacles, out_path):
    plt.figure(figsize=(6.5, 6.5))
    for idx, traj in enumerate(uav_trajs):
        xs, ys = zip(*traj)
        plt.plot(xs, ys, label=f"UAV-{idx}", linewidth=1.4)
        plt.scatter(xs[0], ys[0], marker="o", s=18)
    for idx, traj in enumerate(target_trajs):
        xs, ys = zip(*traj)
        plt.plot(xs, ys, "--", label=f"Target-{idx}", linewidth=1.2)
        plt.scatter(xs[0], ys[0], marker="x", s=20)

    ax = plt.gca()
    for ox, oy, radius in obstacles:
        circle = plt.Circle((ox, oy), radius, color="gray", alpha=0.25)
        ax.add_patch(circle)

    plt.xlim(0, Config.MAP_SIZE)
    plt.ylim(0, Config.MAP_SIZE)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectories")
    plt.legend(loc="upper right", fontsize=7)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_trajectory_plots(env_factory, policy_factory, seeds, max_steps, out_dir):
    ensure_dir(out_dir)
    for seed in seeds:
        env = env_factory()
        policy = policy_factory(env)
        uav_trajs, target_trajs, obstacles = _run_episode(env, policy, seed, max_steps)
        out_path = os.path.join(out_dir, f"seed_{seed}.png")
        plot_trajectories(uav_trajs, target_trajs, obstacles, out_path)


def load_saved_models(models_dir):
    top_models_path = os.path.join(models_dir, "top_models.json")
    episode_4000_path = os.path.join(models_dir, "episode_4000.json")
    top_models = []
    episode_4000 = None
    if os.path.exists(top_models_path):
        with open(top_models_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
            top_models = payload.get("top_models", [])
    if os.path.exists(episode_4000_path):
        with open(episode_4000_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
            episode_4000 = payload.get("path")
    return top_models, episode_4000
