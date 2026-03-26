import json
import os
import shutil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from algo import MADDPG
from eval_seeds import get_eval_seeds
from uav_env import Config, UAVSwarmEnv

# ===========================
# 1. 训练超参数设置
# ===========================
MAX_EPISODES = 5000
MAX_STEPS = 200
EVAL_INTERVAL = 50
SAVE_DIR = "./models"
RESULT_DIR = "./results"
LOG_FILE = os.path.join(RESULT_DIR, "training_log.txt")
TOP_K_MODELS = 10
EVAL_SEEDS = get_eval_seeds()
TARGET_TRAINING_SEQUENCE = [4,5]

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def log(message):
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")


REWARD_KEYS = [
    "coverage",
    "explore",
    "unknown",
    "detect",
    "smooth",
    "spin",
    "obstacle",
    "overlap",
    "crash",
    "share",
]


def format_reward_breakdown(breakdown):
    return ", ".join(f"{key}={breakdown.get(key, 0.0):.3f}" for key in REWARD_KEYS)


def plot_trajectories(output_dir, map_size, obstacles, uav_trajectories, target_trajectories, detection_points, episode):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 8))
    boundary = plt.Rectangle(
        (0, 0),
        map_size,
        map_size,
        fill=False,
        edgecolor="black",
        linestyle="--",
        linewidth=1.5,
        label="Map Boundary"
    )
    ax = plt.gca()
    ax.add_patch(boundary)

    if obstacles:
        for idx, (ox, oy, radius) in enumerate(obstacles):
            circle = plt.Circle((ox, oy), radius, color="black", fill=False, linewidth=1.2,
                                label="Obstacle" if idx == 0 else None)
            ax.add_patch(circle)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for idx, traj in enumerate(uav_trajectories):
        traj = np.array(traj)
        if traj.size == 0:
            continue
        color = colors[idx % len(colors)]
        plt.plot(traj[:, 0], traj[:, 1], color=color, label=f"UAV {idx + 1}")

    for idx, traj in enumerate(target_trajectories):
        traj = np.array(traj)
        if traj.size == 0:
            continue
        plt.plot(
            traj[:, 0],
            traj[:, 1],
            linestyle="--",
            linewidth=2.0,
            marker="o",
            markersize=2.5,
            label=f"Target {idx + 1}"
        )

    if detection_points:
        det_points = np.array(detection_points)
        plt.scatter(det_points[:, 0], det_points[:, 1], s=20, c="red", label="Detection")

    margin = 50.0
    plt.xlim(-margin, map_size + margin)
    plt.ylim(-margin, map_size + margin)
    plt.title(f"UAV Trajectories & Detections (Ep {episode})")
    plt.legend(loc="upper right")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"trajectory_ep{episode}.png"))
    plt.close()


def plot_reward_curve(output_dir, reward_history, window_sizes, episode):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(reward_history, label="Avg Reward")
    for window_size in window_sizes:
        if len(reward_history) >= window_size:
            ma_values = np.convolve(reward_history, np.ones(window_size) / window_size, mode="valid")
            plt.plot(range(window_size - 1, window_size - 1 + len(ma_values)), ma_values, label=f"MA{window_size}")
    plt.title(f"Reward Curve (Ep {episode})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curve.png"))
    plt.close()


def _step_detection_stats(env):
    """统计当前 step 的目标检测数（按目标计数，而不是按无人机计数）。"""
    detected_count = 0
    detected_targets = np.zeros(env.n_targets, dtype=bool)
    for target_idx, target in enumerate(env.targets):
        for agent in env.agents:
            dist = np.sqrt((agent['x'] - target.x) ** 2 + (agent['y'] - target.y) ** 2)
            if dist <= Config.SENSOR_RANGE:
                detected_targets[target_idx] = True
                detected_count += 1
                break
    return detected_targets, detected_count


# ===========================
# 2. 评估函数 (固定种子, 无噪声测试)
# ===========================
def evaluate(env, maddpg, eval_seeds):
    """
    返回:
        - avg_reward: 平均总奖励
        - avg_min_all_detect_step: 平均“所有目标至少被检测一次”的最小步数（越小越好）
        - avg_total_detection_count: 平均总检测次数（用于持续追踪评估，不参与模型选择）
    """
    total_reward = 0.0
    total_min_all_detect_step = 0.0
    total_detection_count = 0.0
    seed_records = []

    for seed in eval_seeds:
        obs_n, _ = env.reset(seed=seed)
        episode_reward = 0.0
        total_detect_this_episode = 0
        target_seen_once = np.zeros(env.n_targets, dtype=bool)
        min_all_detect_step = MAX_STEPS + 1

        for step in range(MAX_STEPS):
            actions = maddpg.select_actions(obs_n, noise_std=0.0)
            next_obs_n, rewards_n, terminated, truncated, infos = env.step(actions)
            del terminated, truncated, infos
            episode_reward += np.sum(rewards_n)

            detected_targets, detected_count = _step_detection_stats(env)
            target_seen_once = np.logical_or(target_seen_once, detected_targets)
            total_detect_this_episode += detected_count

            if np.all(target_seen_once):
                min_all_detect_step = step + 1
                obs_n = next_obs_n
                break

            obs_n = next_obs_n

        total_reward += episode_reward
        total_min_all_detect_step += min_all_detect_step
        total_detection_count += total_detect_this_episode
        seed_records.append({
            "seed": int(seed),
            "episode_reward": float(episode_reward),
            "min_all_detect_step": float(min_all_detect_step),
            "total_detection_count": float(total_detect_this_episode),
        })

    n_episodes = len(eval_seeds)
    return (
        total_reward / n_episodes,
        total_min_all_detect_step / n_episodes,
        total_detection_count / n_episodes,
        seed_records,
    )


def save_topk_models(maddpg, top_models, save_root):
    top_models_dir = os.path.join(save_root, "top_models")
    os.makedirs(top_models_dir, exist_ok=True)

    for folder in os.listdir(top_models_dir):
        folder_path = os.path.join(top_models_dir, folder)
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)

    for rank, record in enumerate(top_models, start=1):
        model_dir = os.path.join(
            top_models_dir,
            f"rank_{rank:02d}_ep_{record['episode']}_step_{record['avg_min_all_detect_step']:.3f}"
        )
        os.makedirs(model_dir, exist_ok=True)
        current_dir = os.path.join(model_dir, "weights")
        os.makedirs(current_dir, exist_ok=True)
        maddpg.load_models(record["temp_model_dir"])
        maddpg.save_models(current_dir)

    summary = [
        {
            "rank": idx + 1,
            "episode": rec["episode"],
            "avg_min_all_detect_step": rec["avg_min_all_detect_step"],
            "avg_total_detection_count": rec["avg_total_detection_count"],
        }
        for idx, rec in enumerate(top_models)
    ]
    with open(os.path.join(top_models_dir, "top_models_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def maintain_topk_models(maddpg, top_models, episode, avg_min_all_detect_step, avg_total_detection_count):
    candidate = {
        "episode": episode,
        "avg_min_all_detect_step": float(avg_min_all_detect_step),
        "avg_total_detection_count": float(avg_total_detection_count),
        "temp_model_dir": os.path.join(SAVE_DIR, "_eval_candidates", f"ep_{episode}"),
    }

    os.makedirs(os.path.dirname(candidate["temp_model_dir"]), exist_ok=True)
    if os.path.exists(candidate["temp_model_dir"]):
        shutil.rmtree(candidate["temp_model_dir"])
    os.makedirs(candidate["temp_model_dir"], exist_ok=True)
    maddpg.save_models(candidate["temp_model_dir"])

    top_models.append(candidate)
    top_models.sort(key=lambda x: (x["avg_min_all_detect_step"], -x["avg_total_detection_count"]))

    removed = []
    if len(top_models) > TOP_K_MODELS:
        removed = top_models[TOP_K_MODELS:]
        top_models = top_models[:TOP_K_MODELS]

    for rec in removed:
        if os.path.exists(rec["temp_model_dir"]):
            shutil.rmtree(rec["temp_model_dir"])

    save_topk_models(maddpg, top_models, SAVE_DIR)
    return top_models


# ===========================
# 3. 训练主程序
# ===========================
def run_training(target_count):
    global SAVE_DIR, RESULT_DIR, LOG_FILE

    SAVE_DIR = os.path.join("./models", f"target_{target_count}")
    RESULT_DIR = os.path.join("./results", f"target_{target_count}")
    LOG_FILE = os.path.join(RESULT_DIR, "training_log.txt")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    env = UAVSwarmEnv(n_target=target_count)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    global_obs_dim = obs_dim * env.n_agents

    maddpg = MADDPG(env.n_agents, obs_dim, act_dim, global_obs_dim)

    scores = []
    all_detect_steps = []
    top_models = []
    eval_seed_details = []

    log(f"Start Training: UAVs={env.n_agents}, Targets={target_count}, Map={Config.MAP_SIZE}x{Config.MAP_SIZE}...")
    log(f"Evaluation uses {len(EVAL_SEEDS)} fixed seeds.")

    for i_episode in range(1, MAX_EPISODES + 1):
        obs_n, _ = env.reset()
        global_obs = env.get_global_state()

        episode_reward = 0
        episode_reward_breakdown = [{key: 0.0 for key in REWARD_KEYS} for _ in range(env.n_agents)]

        noise_decay_episodes = max(1, int(0.8 * MAX_EPISODES))
        progress = min(i_episode, noise_decay_episodes) / noise_decay_episodes
        noise_start = 0.5
        noise_end = 0.01
        noise_std = noise_start * ((noise_end / noise_start) ** progress)

        for t in range(MAX_STEPS):
            actions = maddpg.select_actions(obs_n, noise_std=noise_std)
            next_obs_n, rewards_n, terminated, truncated, infos = env.step(actions)
            next_global_obs = env.get_global_state()

            done = any(terminated) or any(truncated)

            maddpg.memory.push(
                np.array(obs_n),
                global_obs,
                np.array(actions),
                np.array(rewards_n),
                np.array(next_obs_n),
                next_global_obs,
                np.array([done] * env.n_agents)
            )

            ret = maddpg.update()
            if ret is not None:
                c_loss, a_loss = ret
            else:
                c_loss, a_loss = None, None
            del c_loss, a_loss

            obs_n = next_obs_n
            global_obs = next_global_obs
            episode_reward += np.sum(rewards_n)
            reward_breakdown = infos.get("reward_breakdown", [])
            for agent_idx, contrib in enumerate(reward_breakdown):
                for key in REWARD_KEYS:
                    episode_reward_breakdown[agent_idx][key] += contrib.get(key, 0.0)

        scores.append(episode_reward)

        log(f"Episode {i_episode}/{MAX_EPISODES} | "
            f"Reward: {episode_reward:.2f} | "
            f"Noise: {noise_std:.3f}")
        for agent_idx, breakdown in enumerate(episode_reward_breakdown):
            log(f"Episode {i_episode} Agent {agent_idx + 1} Reward Breakdown: "
                f"{format_reward_breakdown(breakdown)}")

        if i_episode % 100 == 0:
            trajectory_dir = os.path.join(RESULT_DIR, "trajectories")
            log(f"[Trajectory] Saving trajectory image to {trajectory_dir} (ep {i_episode})")
            plot_trajectories(
                trajectory_dir,
                Config.MAP_SIZE,
                env.obstacles,
                env.uav_trajectories,
                env.target_trajectories,
                env.detection_points,
                i_episode
            )
            log(f"[Reward] Saving reward curve to {RESULT_DIR} (ep {i_episode})")
            plot_reward_curve(RESULT_DIR, scores, [50, 100], i_episode)

        if i_episode % EVAL_INTERVAL == 0:
            eval_reward, avg_min_all_detect_step, avg_total_detection_count, seed_records = evaluate(env, maddpg, EVAL_SEEDS)
            all_detect_steps.append(avg_min_all_detect_step)

            eval_seed_details.append({
                "episode": i_episode,
                "target_count": target_count,
                "n_uav": env.n_agents,
                "seed_metrics": seed_records,
            })
            with open(os.path.join(RESULT_DIR, "eval_seed_details.json"), "w", encoding="utf-8") as f:
                json.dump(eval_seed_details, f, indent=2, ensure_ascii=False)

            log(f"\n--- Evaluation @ Ep {i_episode} ---")
            log(f"Avg Reward: {eval_reward:.2f}")
            log(f"Avg Min-All-Detected Step: {avg_min_all_detect_step:.3f} (lower is better)")
            log(f"Avg Total Detection Count: {avg_total_detection_count:.3f}")
            log("----------------------------\n")

            top_models = maintain_topk_models(
                maddpg,
                top_models,
                i_episode,
                avg_min_all_detect_step,
                avg_total_detection_count,
            )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(all_detect_steps, color='orange')
    plt.title("Avg Min-All-Detected Step")
    plt.xlabel(f"Evaluation Index (every {EVAL_INTERVAL} episodes)")
    plt.ylabel("Steps (lower is better)")
    plt.grid(True)

    plt.savefig(f"{RESULT_DIR}/training_result.png")
    plt.show()

    log(f"Training Finished for target_count={target_count}. Results saved.")


if __name__ == "__main__":
    for target_count in TARGET_TRAINING_SEQUENCE:
        run_training(target_count)
