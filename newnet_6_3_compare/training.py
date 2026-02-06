import json
import os
import shutil

import numpy as np

from eval_seeds import get_eval_seeds
from maddpg import MADDPG
from iddpg import IDDPG
from metrics import evaluate_policy
from uav_env import UAVSwarmEnv


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _clear_dir(path):
    if os.path.exists(path):
        for fname in os.listdir(path):
            file_path = os.path.join(path, fname)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)


def _save_snapshot(save_fn, snapshot_dir):
    if os.path.exists(snapshot_dir):
        _clear_dir(snapshot_dir)
    else:
        _ensure_dir(snapshot_dir)
    save_fn(snapshot_dir)


def _update_top_models(top_models, score, episode, max_size=3):
    filtered = [item for item in top_models if item["episode"] != episode]
    filtered.append({"score": score, "episode": episode})
    filtered.sort(key=lambda item: item["score"])
    trimmed = filtered[:max_size]
    in_top = any(item["episode"] == episode for item in trimmed)
    return trimmed, in_top


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
    top_models_dir = os.path.join(models_dir, "top_models")
    _ensure_dir(top_models_dir)
    top_models = []
    top_model_dirs = {}
    episode_4000_dir = None
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

        if episode == 4000:
            episode_4000_dir = os.path.join(models_dir, "episode_4000")
            _save_snapshot(maddpg.save_models, episode_4000_dir)

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
            score = eval_stats["avg_min_all_detect_step"]
            if score < best_score:
                best_score = score
                _save_snapshot(maddpg.save_models, best_model_dir)

            top_models, in_top = _update_top_models(top_models, score, episode, max_size=3)
            if in_top:
                snapshot_dir = os.path.join(top_models_dir, f"episode_{episode}")
                _save_snapshot(maddpg.save_models, snapshot_dir)
                top_model_dirs[episode] = snapshot_dir

    keep_episodes = {item["episode"] for item in top_models}
    for episode, path in list(top_model_dirs.items()):
        if episode not in keep_episodes and os.path.exists(path):
            shutil.rmtree(path)
            top_model_dirs.pop(episode, None)

    top_models_payload = []
    for item in top_models:
        episode = item["episode"]
        top_models_payload.append(
            {
                "episode": episode,
                "score": item["score"],
                "path": top_model_dirs.get(episode),
            }
        )

    with open(os.path.join(run_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(eval_history, f, indent=2, ensure_ascii=False)

    with open(os.path.join(models_dir, "top_models.json"), "w", encoding="utf-8") as f:
        json.dump({"top_models": top_models_payload}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(models_dir, "episode_4000.json"), "w", encoding="utf-8") as f:
        json.dump({"path": episode_4000_dir}, f, indent=2, ensure_ascii=False)

    return (
        {
            "best": best_model_dir,
            "top_models": top_models_payload,
            "episode_4000": episode_4000_dir,
        },
        eval_history,
    )


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
    top_models_dir = os.path.join(models_dir, "top_models")
    _ensure_dir(top_models_dir)
    top_models = []
    top_model_dirs = {}
    episode_4000_dir = None
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

        if episode == 4000:
            episode_4000_dir = os.path.join(models_dir, "episode_4000")
            _save_snapshot(iddpg.save, episode_4000_dir)

        if episode % eval_interval == 0:
            eval_env = UAVSwarmEnv(n_uav=n_uav, n_target=n_target, use_pf=True, use_pf_obs=True)
            eval_stats = _run_eval(eval_env, iddpg, eval_seeds, max_steps)
            eval_stats["episode"] = episode
            eval_history.append(eval_stats)
            score = eval_stats["avg_min_all_detect_step"]
            if score < best_score:
                best_score = score
                _save_snapshot(iddpg.save, best_model_dir)

            top_models, in_top = _update_top_models(top_models, score, episode, max_size=3)
            if in_top:
                snapshot_dir = os.path.join(top_models_dir, f"episode_{episode}")
                _save_snapshot(iddpg.save, snapshot_dir)
                top_model_dirs[episode] = snapshot_dir

    keep_episodes = {item["episode"] for item in top_models}
    for episode, path in list(top_model_dirs.items()):
        if episode not in keep_episodes and os.path.exists(path):
            shutil.rmtree(path)
            top_model_dirs.pop(episode, None)

    top_models_payload = []
    for item in top_models:
        episode = item["episode"]
        top_models_payload.append(
            {
                "episode": episode,
                "score": item["score"],
                "path": top_model_dirs.get(episode),
            }
        )

    with open(os.path.join(run_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(eval_history, f, indent=2, ensure_ascii=False)

    with open(os.path.join(models_dir, "top_models.json"), "w", encoding="utf-8") as f:
        json.dump({"top_models": top_models_payload}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(models_dir, "episode_4000.json"), "w", encoding="utf-8") as f:
        json.dump({"path": episode_4000_dir}, f, indent=2, ensure_ascii=False)

    return (
        {
            "best": best_model_dir,
            "top_models": top_models_payload,
            "episode_4000": episode_4000_dir,
        },
        eval_history,
    )
