from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter

from uav_search_rl.envs.particle_filter import ParticleFilterConfig
from uav_search_rl.envs.uav_search_env import EnvConfig, UAVSearchEnv
from uav_search_rl.eval.evaluator import aggregate_episode, compute_score, summarize_metrics
from uav_search_rl.marl.maddpg import MADDPG, MADDPGConfig
from uav_search_rl.marl.noise import GaussianNoise, OUNoise
from uav_search_rl.marl.replay_buffer import ReplayBuffer
from uav_search_rl.utils.checkpoint import load_checkpoint, save_checkpoint
from uav_search_rl.utils.logger import MetricLogger
from uav_search_rl.utils.seed import set_seed


def flatten_state(state: Dict[str, np.ndarray]) -> np.ndarray:
    obstacles = state["obstacles"].astype(np.float32).flatten()
    uav_states = state["uav_states"].astype(np.float32).flatten()
    possibility = state["possibility"].astype(np.float32).flatten()
    env_cov = state["env_cov"].astype(np.float32).flatten()
    return np.concatenate([obstacles, uav_states, possibility, env_cov], axis=0)


def flatten_obs(obs_list: List[Dict[str, np.ndarray]]) -> np.ndarray:
    obs_vecs = []
    for obs in obs_list:
        parts = [
            obs["selfstate"].astype(np.float32).flatten(),
            obs["partobs"].astype(np.float32).flatten(),
            obs["partp"].astype(np.float32).flatten(),
            obs["neighbors"].astype(np.float32).flatten(),
            obs["partcov"].astype(np.float32).flatten(),
        ]
        obs_vecs.append(np.concatenate(parts, axis=0))
    return np.stack(obs_vecs)


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
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])
    rng = np.random.default_rng(cfg["train"]["seed"])

    env = build_env(cfg, rng)
    state, obs = env.reset()
    state_vec = flatten_state(state)
    obs_vec = flatten_obs(obs)

    obs_dim = obs_vec.shape[1]
    state_dim = state_vec.shape[0]
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

    buffer = ReplayBuffer(
        capacity=cfg["train"]["buffer_size"],
        num_agents=cfg["env"]["num_uavs"],
        state_dim=state_dim,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    if cfg["train"]["noise_type"] == "ou":
        noise = OUNoise(action_dim, theta=cfg["train"]["noise_theta"], sigma=cfg["train"]["noise_sigma"])
    else:
        noise = GaussianNoise(action_dim, sigma=cfg["train"]["noise_sigma"])

    start_episode = 0
    best_score = -1e9
    checkpoints_dir = Path("checkpoints")
    writer = SummaryWriter(log_dir="runs/train")

    if args.resume:
        checkpoint = load_checkpoint(args.resume, map_location=cfg["train"]["device"])
        maddpg.load_state_dict(checkpoint["model"])
        for opt, state in zip(maddpg.actor_opts, checkpoint["actor_opts"]):
            opt.load_state_dict(state)
        for opt, state in zip(maddpg.critic_opts, checkpoint["critic_opts"]):
            opt.load_state_dict(state)
        start_episode = checkpoint.get("episode", 0)
        best_score = checkpoint.get("best_score", best_score)

    logger = MetricLogger(window=cfg["train"]["log_interval"])
    global_step = 0

    for episode in range(start_episode, cfg["train"]["episodes"]):
        state, obs = env.reset()
        episode_rewards = np.zeros(cfg["env"]["num_uavs"], dtype=np.float32)
        info_history: List[Dict[str, float]] = []
        noise.reset()

        for step in range(cfg["train"]["max_steps"]):
            state_vec = flatten_state(state)
            obs_vec = flatten_obs(obs)
            actions = maddpg.act(obs_vec)
            actions += noise.sample()
            actions[:, 0] *= cfg["env"]["max_accel"]
            actions[:, 1] *= cfg["env"]["max_yaw_rate"]
            next_state, next_obs, rewards, done, info = env.step(actions)
            next_state_vec = flatten_state(next_state)
            next_obs_vec = flatten_obs(next_obs)

            buffer.add(state_vec, obs_vec, actions, rewards, next_state_vec, next_obs_vec, float(done))
            state, obs = next_state, next_obs
            episode_rewards += rewards
            info_history.append(info)
            global_step += 1

            if len(buffer) >= cfg["train"]["batch_size"] and global_step > cfg["train"]["warmup_steps"]:
                batch = buffer.sample(cfg["train"]["batch_size"])
                metrics = maddpg.update(batch)
                for key, value in metrics.items():
                    logger.update(key, value)

        avg_reward = float(np.mean(episode_rewards))
        writer.add_scalar("train/episode_reward", avg_reward, episode)
        writer.add_scalar("train/cover_rate", info_history[-1]["cover_rate"], episode)
        writer.add_scalar("train/explore_rate", info_history[-1]["explore_rate"], episode)

        if (episode + 1) % cfg["train"]["log_interval"] == 0:
            print(f"Episode {episode + 1}: reward={avg_reward:.3f}")
            for key in logger.history.keys():
                print(f"  {key}: {logger.mean(key):.4f}")
            logger.reset()

        if (episode + 1) % cfg["train"]["eval_interval"] == 0:
            eval_metrics = []
            eval_rng = np.random.default_rng(cfg["train"]["seed"] + 1000)
            for _ in range(cfg["train"]["eval_episodes"]):
                eval_env = build_env(cfg, eval_rng)
                state, obs = eval_env.reset()
                info_history = []
                for step in range(cfg["train"]["max_steps"]):
                    state_vec = flatten_state(state)
                    obs_vec = flatten_obs(obs)
                    actions = maddpg.act(obs_vec)
                    actions[:, 0] *= cfg["env"]["max_accel"]
                    actions[:, 1] *= cfg["env"]["max_yaw_rate"]
                    state, obs, _, _, info = eval_env.step(actions)
                    info_history.append(info)
                eval_metrics.append(
                    aggregate_episode(info_history, cfg["env"]["grid_m"] * cfg["env"]["grid_n"], cfg["train"]["max_steps"])
                )
            summary = summarize_metrics(eval_metrics)
            score = compute_score(summary)
            writer.add_scalar("eval/score", score, episode)
            for key, value in summary.items():
                writer.add_scalar(f"eval/{key}", value, episode)
            if score > best_score:
                best_score = score
                save_checkpoint(
                    checkpoints_dir / "best",
                    {
                        "model": maddpg.state_dict(),
                        "actor_opts": [opt.state_dict() for opt in maddpg.actor_opts],
                        "critic_opts": [opt.state_dict() for opt in maddpg.critic_opts],
                        "config": cfg,
                        "best_score": best_score,
                        "episode": episode,
                        "summary": summary,
                    },
                )

        if (episode + 1) % cfg["train"]["save_interval"] == 0:
            save_checkpoint(
                checkpoints_dir / "last",
                {
                    "model": maddpg.state_dict(),
                    "actor_opts": [opt.state_dict() for opt in maddpg.actor_opts],
                    "critic_opts": [opt.state_dict() for opt in maddpg.critic_opts],
                    "config": cfg,
                    "best_score": best_score,
                    "episode": episode,
                },
            )

    save_checkpoint(
        checkpoints_dir / "last",
        {
            "model": maddpg.state_dict(),
            "actor_opts": [opt.state_dict() for opt in maddpg.actor_opts],
            "critic_opts": [opt.state_dict() for opt in maddpg.critic_opts],
            "config": cfg,
            "best_score": best_score,
            "episode": cfg["train"]["episodes"],
        },
    )
    writer.close()


if __name__ == "__main__":
    main()
