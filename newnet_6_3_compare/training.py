import json
import os

import numpy as np

from eval_seeds import get_eval_seeds
from maddpg import MADDPG
from iddpg import IDDPG
from metrics import evaluate_policy
from uav_env import UAVSwarmEnv


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


class PolicyWrapper:
    def __init__(self, controller):
        self.controller = controller

    def select_actions(self, obs_n):
        return self.controller.select_actions(obs_n, noise_std=0.0)


def _run_eval(env, controller, eval_seeds, max_steps):
    policy = PolicyWrapper(controller)
    metrics = evaluate_policy(env, policy, eval_seeds, max_steps=max_steps)
    avg_min_step = float(np.mean(metrics["min_all_detect_steps"]))
    avg_total_det = float(np.mean(metrics["total_detection_counts"]))
    avg_overlap = float(np.mean(metrics["overlap_rates"]))
    avg_collision = float(np.mean(metrics["collision_counts"]))
    avg_coverage = float(np.mean(metrics["coverage_efficiencies"]))
    return {
        "avg_min_all_detect_step": avg_min_step,
        "avg_total_detection_count": avg_total_det,
        "avg_overlap_rate": avg_overlap,
        "avg_collision_count": avg_collision,
        "avg_coverage_efficiency": avg_coverage,
    }


def train_maddpg(
    run_dir,
    n_uav,
    n_target,
    use_pf=True,
    use_pf_obs=True,
    max_episodes=5000,
    max_steps=200,
    eval_interval=50,
):
    _ensure_dir(run_dir)
    models_dir = os.path.join(run_dir, "models")
    _ensure_dir(models_dir)

    env = UAVSwarmEnv(n_uav=n_uav, n_target=n_target, use_pf=use_pf, use_pf_obs=use_pf_obs)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    global_obs_dim = obs_dim * env.n_agents

    maddpg = MADDPG(env.n_agents, obs_dim, act_dim, global_obs_dim)

    eval_seeds = get_eval_seeds()
    best_score = float("inf")
    best_model_dir = os.path.join(models_dir, "best")
    eval_history = []

    for episode in range(1, max_episodes + 1):
        obs_n, _ = env.reset()
        global_obs = env.get_global_state()
        noise_decay_episodes = max(1, int(0.8 * max_episodes))
        progress = min(episode, noise_decay_episodes) / noise_decay_episodes
        noise_std = 0.5 * ((0.01 / 0.5) ** progress)

        for _ in range(max_steps):
            actions = maddpg.select_actions(obs_n, noise_std=noise_std)
            next_obs_n, rewards_n, terminated, truncated, _ = env.step(actions)
            done = any(terminated) or any(truncated)
            next_global_obs = env.get_global_state()

            maddpg.memory.push(
                np.array(obs_n),
                global_obs,
                np.array(actions),
                np.array(rewards_n),
                np.array(next_obs_n),
                next_global_obs,
                np.array([done] * env.n_agents),
            )
            maddpg.update()
            obs_n = next_obs_n
            global_obs = next_global_obs

        if episode % eval_interval == 0:
            eval_env = UAVSwarmEnv(
                n_uav=n_uav,
                n_target=n_target,
                use_pf=use_pf,
                use_pf_obs=use_pf_obs,
            )
            eval_stats = _run_eval(eval_env, maddpg, eval_seeds, max_steps)
            eval_stats["episode"] = episode
            eval_history.append(eval_stats)
            if eval_stats["avg_min_all_detect_step"] < best_score:
                best_score = eval_stats["avg_min_all_detect_step"]
                if os.path.exists(best_model_dir):
                    for fname in os.listdir(best_model_dir):
                        os.remove(os.path.join(best_model_dir, fname))
                else:
                    _ensure_dir(best_model_dir)
                maddpg.save_models(best_model_dir)

    with open(os.path.join(run_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(eval_history, f, indent=2, ensure_ascii=False)

    return best_model_dir, eval_history


def train_iddpg(
    run_dir,
    n_uav,
    n_target,
    max_episodes=5000,
    max_steps=200,
    eval_interval=50,
):
    _ensure_dir(run_dir)
    models_dir = os.path.join(run_dir, "models")
    _ensure_dir(models_dir)

    env = UAVSwarmEnv(n_uav=n_uav, n_target=n_target, use_pf=True, use_pf_obs=True)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    iddpg = IDDPG(env.n_agents, obs_dim, act_dim)

    eval_seeds = get_eval_seeds()
    best_score = float("inf")
    best_model_dir = os.path.join(models_dir, "best")
    eval_history = []

    for episode in range(1, max_episodes + 1):
        obs_n, _ = env.reset()
        noise_decay_episodes = max(1, int(0.8 * max_episodes))
        progress = min(episode, noise_decay_episodes) / noise_decay_episodes
        noise_std = 0.5 * ((0.01 / 0.5) ** progress)

        for _ in range(max_steps):
            actions = iddpg.select_actions(obs_n, noise_std=noise_std)
            next_obs_n, rewards_n, terminated, truncated, _ = env.step(actions)
            done = float(any(terminated) or any(truncated))
            for idx, agent in enumerate(iddpg.agents):
                agent.memory.push(
                    obs_n[idx],
                    actions[idx],
                    [rewards_n[idx]],
                    next_obs_n[idx],
                    [done],
                )
            iddpg.update()
            obs_n = next_obs_n

        if episode % eval_interval == 0:
            eval_env = UAVSwarmEnv(n_uav=n_uav, n_target=n_target, use_pf=True, use_pf_obs=True)
            eval_stats = _run_eval(eval_env, iddpg, eval_seeds, max_steps)
            eval_stats["episode"] = episode
            eval_history.append(eval_stats)
            if eval_stats["avg_min_all_detect_step"] < best_score:
                best_score = eval_stats["avg_min_all_detect_step"]
                if os.path.exists(best_model_dir):
                    for fname in os.listdir(best_model_dir):
                        os.remove(os.path.join(best_model_dir, fname))
                else:
                    _ensure_dir(best_model_dir)
                iddpg.save(best_model_dir)

    with open(os.path.join(run_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(eval_history, f, indent=2, ensure_ascii=False)

    return best_model_dir, eval_history
