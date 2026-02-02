from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from uav_search_rl.envs.particle_filter import ParticleFilterConfig
from uav_search_rl.envs.uav_search_env import EnvConfig, UAVSearchEnv
from uav_search_rl.eval.evaluator import aggregate_episode, summarize_metrics
from uav_search_rl.marl.maddpg import MADDPG, MADDPGConfig
from uav_search_rl.utils.checkpoint import load_checkpoint
from uav_search_rl.utils.seed import set_seed
from uav_search_rl.train import flatten_obs, flatten_state


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_env(cfg: Dict, rng: np.random.Generator) -> UAVSearchEnv:
    pf_cfg = cfg["env"]["pf"]
    pf = ParticleFilterConfig(
        num_particles=pf_cfg["num_particles"],
        init_pos_sigma=pf_cfg["init_pos_sigma"],
        speed_range=tuple(pf_cfg["speed_range"]),
        heading_range=tuple(pf_cfg["heading_range"]),
        yaw_rate_range=tuple(pf_cfg["yaw_rate_range"]),
        process_noise=pf_cfg["process_noise"],
        meas_noise=pf_cfg["meas_noise"],
        neg_decay=pf_cfg["neg_decay"],
        sigmoid_k=pf_cfg["sigmoid_k"],
        soft_vmax=pf_cfg["soft_vmax"],
        soft_yaw_rate_max=pf_cfg["soft_yaw_rate_max"],
    )
    reward_cfg = cfg["reward"]
    reward_weights = {
        "cover": reward_cfg["cover_weight"],
        "explore": reward_cfg["explore_weight"],
        "overlap": reward_cfg["overlap_weight"],
        "share": reward_cfg["share_weight"],
        "smooth": reward_cfg["smooth_weight"],
        "obstacle": reward_cfg["obstacle_weight"],
        "collision": reward_cfg["collision_weight"],
        "entropy": reward_cfg["entropy_weight"],
        "warning": reward_cfg["warning_weight"],
        "energy": reward_cfg["energy_weight"],
    }
    env_cfg = EnvConfig(
        grid_m=cfg["env"]["grid_m"],
        grid_n=cfg["env"]["grid_n"],
        obstacle_ratio=cfg["env"]["obstacle_ratio"],
        num_uavs=cfg["env"]["num_uavs"],
        num_targets=cfg["env"]["num_targets"],
        dt=cfg["env"]["dt"],
        fov_radius=cfg["env"]["fov_radius"],
        comm_range=cfg["env"]["comm_range"],
        safe_distance=cfg["env"]["safe_distance"],
        warning_distance=cfg["env"]["warning_distance"],
        min_speed=cfg["env"]["min_speed"],
        max_speed=cfg["env"]["max_speed"],
        max_accel=cfg["env"]["max_accel"],
        max_yaw_rate=cfg["env"]["max_yaw_rate"],
        obs_patch_size=cfg["env"]["obs_patch_size"],
        max_neighbors=cfg["env"]["max_neighbors"],
        detect_prob=cfg["env"]["detect_prob"],
        pf=pf,
        reward_weights=reward_weights,
    )
    return UAVSearchEnv(env_cfg, rng)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])
    rng = np.random.default_rng(cfg["train"]["seed"] + 2000)
    env = build_env(cfg, rng)
    state, obs = env.reset()

    obs_dim = flatten_obs(obs).shape[1]
    state_dim = flatten_state(state).shape[0]
    action_dim = 2

    maddpg_cfg = MADDPGConfig(
        num_agents=cfg["env"]["num_uavs"],
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=np.array([cfg["env"]["max_accel"], cfg["env"]["max_yaw_rate"]], dtype=np.float32),
        hidden_dim=cfg["train"]["hidden_dim"],
        actor_lr=cfg["train"]["actor_lr"],
        critic_lr=cfg["train"]["critic_lr"],
        gamma=cfg["train"]["gamma"],
        tau=cfg["train"]["tau"],
        grad_clip=cfg["train"]["grad_clip"],
        device=cfg["train"]["device"],
    )
    maddpg = MADDPG(maddpg_cfg)
    checkpoint = load_checkpoint(args.ckpt, map_location=cfg["train"]["device"])
    maddpg.load_state_dict(checkpoint["model"])

    metrics = []
    for _ in range(args.episodes):
        env = build_env(cfg, rng)
        state, obs = env.reset()
        info_history: List[Dict[str, float]] = []
        for _ in range(cfg["train"]["max_steps"]):
            actions = maddpg.act(flatten_obs(obs))
            actions[:, 0] *= cfg["env"]["max_accel"]
            actions[:, 1] *= cfg["env"]["max_yaw_rate"]
            state, obs, _, _, info = env.step(actions)
            info_history.append(info)
        metrics.append(
            aggregate_episode(info_history, cfg["env"]["grid_m"] * cfg["env"]["grid_n"], cfg["train"]["max_steps"])
        )

    summary = summarize_metrics(metrics)
    print("Evaluation summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
