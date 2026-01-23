import json
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

import maddpg_v3_10 as RL
import targetpre_v3_10 as tp

os.environ["OMP_NUM_THREADS"] = "1"
matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def moving_average(data, window):
    if len(data) < window:
        return np.array([])
    kernel = np.ones(window) / window
    return np.convolve(np.array(data), kernel, mode="valid")


def save_reward_history(output_dir, reward_history, noise_history):
    ensure_dir(output_dir)

    csv_path = os.path.join(output_dir, "reward_history.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("episode,avg_reward,noise_std\n")
        for idx, reward in enumerate(reward_history, start=1):
            noise = noise_history[idx - 1] if idx - 1 < len(noise_history) else None
            f.write(f"{idx},{reward:.6f},{noise:.6f}\n")

    plt.figure(figsize=(8, 4))
    plt.plot(reward_history, label="Avg Reward")
    ma_50 = moving_average(reward_history, 50)
    ma_100 = moving_average(reward_history, 100)
    if ma_50.size > 0:
        plt.plot(range(49, 49 + len(ma_50)), ma_50, label="MA50")
    if ma_100.size > 0:
        plt.plot(range(99, 99 + len(ma_100)), ma_100, label="MA100")
    plt.title("Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curve.png"))
    plt.close()


def save_tracking_metrics(output_dir, metrics_history):
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "tracking_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("episode,avg_detect_streak,avg_lost_streak,lock_count,lock_time_ratio,target_covered_ratio\n")
        for item in metrics_history:
            f.write(
                f"{item['episode']},"
                f"{item['avg_detect_streak']:.6f},"
                f"{item['avg_lost_streak']:.6f},"
                f"{item['lock_count']},"
                f"{item['lock_time_ratio']:.6f},"
                f"{item['target_covered_ratio']:.6f}\n"
            )


def save_detailed_rewards(output_dir, detailed_history):
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "detailed_rewards.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "episode,avg_reward,avg_r_info,avg_r_track,avg_r_action,avg_r_boundary,avg_r_collision,"
            "avg_r_uncertainty,clip_ratio,acquire,keep,lost,unlock,reacquire\n"
        )
        for item in detailed_history:
            f.write(
                f"{item['episode']},"
                f"{item['avg_reward']:.6f},"
                f"{item['avg_r_info']:.6f},"
                f"{item['avg_r_track']:.6f},"
                f"{item['avg_r_action']:.6f},"
                f"{item['avg_r_boundary']:.6f},"
                f"{item['avg_r_collision']:.6f},"
                f"{item['avg_r_uncertainty']:.6f},"
                f"{item['clip_ratio']:.6f},"
                f"{item['acquire']},"
                f"{item['keep']},"
                f"{item['lost']},"
                f"{item['unlock']},"
                f"{item['reacquire']}\n"
            )


def save_config(output_dir, config):
    ensure_dir(output_dir)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def init_predictors_targets(map_size, obstacles, tarcfgs):
    predictors = []
    real_targets = []
    for cfg in tarcfgs:
        start_belief = cfg["pos"]
        v_range = cfg["v_range"]
        theta_range = cfg["theta_range"]
        ID = cfg["ID"]
        priority = cfg["priority"]
        p = tp.TargetPredictor(map_size, obstacles, v_range, theta_range, start_belief, ID=ID, priority=priority)

        initial_v = cfg["initial_v"]
        initial_phi = cfg["initial_phi"]
        enable_reflection = cfg.get("enable_boundary_reflection", True)
        reflection_map_size = map_size if enable_reflection else None
        r = tp.RealTarget(
            ID,
            priority,
            start_belief,
            initial_v,
            initial_phi,
            v_range=v_range,
            phi_range=theta_range,
            map_size=reflection_map_size
        )

        predictors.append(p)
        real_targets.append(r)
    return predictors, real_targets


def compute_local_entropy_for_uavs(uav_list, predictors):
    local_entropies = []
    for uav in uav_list:
        local_entropy = 0.0
        for predictor in predictors:
            local_entropy += predictor.get_local_entropy(uav.pos, uav.detect_radius)
        local_entropies.append(local_entropy)
    return local_entropies


def plot_trajectories(output_dir, map_size, obs_map, uav_trajectories, target_trajectories, detection_points, eval_tag):
    ensure_dir(output_dir)
    plt.figure(figsize=(8, 8))
    if obs_map is not None:
        obstacle_indices = np.argwhere(obs_map == -1)
        if obstacle_indices.size > 0:
            obs_x = obstacle_indices[:, 1] + 0.5
            obs_y = (map_size[0] - 1) - obstacle_indices[:, 0] + 0.5
            plt.scatter(obs_x, obs_y, s=30, c="black", marker="s", label="Obstacle")

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
            linewidth=2.5,
            marker="o",
            markersize=2.5,
            label=f"Target {idx + 1}"
        )

    if detection_points:
        det_points = np.array(detection_points)
        plt.scatter(det_points[:, 0], det_points[:, 1], s=20, c="red", label="Detection")

    ax = plt.gca()
    boundary = Rectangle((0, 0), map_size[1], map_size[0], fill=False, edgecolor="gray", linewidth=1.5)
    ax.add_patch(boundary)

    all_points = []
    for traj in uav_trajectories + target_trajectories:
        if traj:
            all_points.append(np.array(traj))
    if all_points:
        all_points = np.vstack(all_points)
        min_x = min(0.0, float(np.min(all_points[:, 0])))
        max_x = max(map_size[1], float(np.max(all_points[:, 0])))
        min_y = min(0.0, float(np.min(all_points[:, 1])))
        max_y = max(map_size[0], float(np.max(all_points[:, 1])))
    else:
        min_x, max_x = 0.0, float(map_size[1])
        min_y, max_y = 0.0, float(map_size[0])

    margin = 100.0
    plt.xlim(min_x - margin, max_x + margin)
    plt.ylim(min_y - margin, max_y + margin)
    plt.title("UAV Trajectories & Detections")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"trajectory_{eval_tag}.png"))
    plt.close()


def plot_entropy_curve(output_dir, entropy_curve, eval_tag):
    ensure_dir(output_dir)
    plt.figure(figsize=(8, 4))
    plt.plot(entropy_curve, label="Entropy")
    plt.title("Entropy Curve")
    plt.xlabel("Time")
    plt.ylabel("H(t)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"entropy_curve_{eval_tag}.png"))
    plt.close()


def plot_heatmaps(output_dir, predictors, time_step, eval_tag):
    for predictor in predictors:
        grid = predictor._generate_grid()
        plt.figure(figsize=(6, 5))
        plt.imshow(grid, cmap="hot", origin="upper")
        plt.colorbar(label="Belief")
        plt.title(f"Target {predictor.ID} Heatmap t={time_step}")
        plt.tight_layout()
        filename = f"heatmap_target{predictor.ID}_t{time_step}_{eval_tag}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()


def run_evaluation(eval_dir, episode, uav_list, uav_configs, map_size, obs_map, obstacles, tarcfgs, max_steps):
    predictors, real_targets = init_predictors_targets(map_size, obstacles, tarcfgs)
    for i, uav in enumerate(uav_list):
        uav.pos = np.array(uav_configs[i]["pos"], dtype=float)
        uav.v = uav_configs[i]["initial_v"]
        uav.phi = np.radians(uav_configs[i]["phi"])
        uav.reset_episode()

    uav_trajectories = [[] for _ in uav_list]
    target_trajectories = [[] for _ in real_targets]
    detection_points = []
    entropy_curve = []
    time_points = {0, 25, 50, 75, 99}

    for uav in uav_list:
        uav.last_obs = uav.get_observation(map_size, obs_map, uav_list, predictors)

    for t in range(max_steps):
        if t in time_points:
            plot_heatmaps(eval_dir, predictors, t, f"ep{episode}")

        entropy_curve.append(sum(p.get_entropy() for p in predictors))

        action_list = []
        for uav in uav_list:
            action = uav.brain.select_action(uav.last_obs)
            action_list.append(np.clip(action, -1.0, 1.0))

        for i, uav in enumerate(uav_list):
            uav.state_update(action_list[i], map_size, obs_map)
            uav_trajectories[i].append(uav.pos.copy())

        meas_noise_std = 3.0
        innovation_norm = np.zeros(len(predictors))
        for i, p in enumerate(predictors):
            real_pos = real_targets[i].state[0:2]
            target_trajectories[i].append(real_pos.copy())

            det_res = []
            sum_z = 0.0
            sum_detected = 0
            for u_idx, u in enumerate(uav_list):
                dist = np.linalg.norm(u.pos - real_pos)
                is_detected = (dist < 250.0) and (np.random.rand() < 0.9)
                temp_state = {"detected": is_detected, "measurement": None, "uavpos": u.pos, "uavdp": u.detecct_p}
                if is_detected:
                    detection_points.append(u.pos.copy())
                    temp_state["measurement"] = real_pos + np.random.randn(2) * meas_noise_std
                    sum_z += temp_state["measurement"]
                    sum_detected += 1
                det_res.append(temp_state)

            if sum_detected == 0:
                z_average = None
                p.step_update(None, innovation_norm[i], det_res)
                innovation_norm[i] = 0.0
            else:
                z_average = sum_z / sum_detected
                p.step_update(z_average, innovation_norm[i], det_res)
                innovation_norm[i] = np.linalg.norm(z_average - p.state_si[0:2])

            real_targets[i].step_forward()

        for uav in uav_list:
            uav.last_obs = uav.get_observation(map_size, obs_map, uav_list, predictors)

    eval_tag = f"ep{episode}"
    plot_trajectories(eval_dir, map_size, obs_map, uav_trajectories, target_trajectories, detection_points, eval_tag)
    plot_entropy_curve(eval_dir, entropy_curve, eval_tag)

    eval_h_final = entropy_curve[-1] if entropy_curve else 0.0
    eval_delta_h = (entropy_curve[0] - entropy_curve[-1]) if len(entropy_curve) > 1 else 0.0
    return eval_h_final, eval_delta_h


# 主要训练逻辑
def train_with_improvements():
    # 1. 初始化环境与参数
    map_size = (2000, 2000)
    # 简单的障碍物 (Row, Col)
    obstacles = [(600, 600), (600, 601), (601, 600), (601, 601),
                 (1200, 1200), (1200, 1201), (1201, 1200), (1201, 1201)]

    # 转换障碍物地图
    obs_map = np.zeros(map_size)
    for r, c in obstacles:
        if 0 <= r < map_size[0] and 0 <= c < map_size[1]:
            obs_map[r, c] = -1

    def generate_random_tarcfgs(num_targets, base_configs=None, pos_jitter=0.0, angle_jitter_deg=0.0,
                                full_random=False, enable_boundary_reflection=True):
        tarcfgs = []
        for tid in range(1, num_targets + 1):
            v_range = (15, 60)
            if full_random or base_configs is None:
                x = float(np.random.uniform(0, map_size[1]))
                y = float(np.random.uniform(0, map_size[0]))
                initial_v = float(np.random.uniform(v_range[0], v_range[1]))
                initial_phi = float(np.random.uniform(-180.0, 180.0))
            else:
                base_cfg = base_configs[tid - 1]
                x = float(np.clip(base_cfg["pos"][0] + np.random.uniform(-pos_jitter, pos_jitter), 0, map_size[1]))
                y = float(np.clip(base_cfg["pos"][1] + np.random.uniform(-pos_jitter, pos_jitter), 0, map_size[0]))
                initial_v = float(base_cfg["initial_v"])
                initial_phi = float(base_cfg["initial_phi"] + np.random.uniform(-angle_jitter_deg, angle_jitter_deg))
            tarcfgs.append({
                "ID": tid,
                "pos": (x, y),
                "priority": 1,
                "v_range": v_range,
                "theta_range": (-np.pi, np.pi),
                "initial_v": initial_v,
                "initial_phi": initial_phi,
                "enable_boundary_reflection": enable_boundary_reflection
            })
        return tarcfgs

    # 初始化 MADDPG 系统
    state_dim = 21
    action_dim = 2
    # 无人机参数配置
    uav_configs = [
        {"id": 1, "pos": [200, 200], "phi": 45, "initial_v": 30},
        {"id": 2, "pos": [1800, 1800], "phi": 225, "initial_v": 30},
        {"id": 3, "pos": [200, 1800], "phi": 315, "initial_v": 30},
        {"id": 4, "pos": [1800, 200], "phi": 135, "initial_v": 30},
        {"id": 5, "pos": [1000, 1000], "phi": 0, "initial_v": 0}
    ]

    num_uavs = len(uav_configs)
    total_state_dim = state_dim * num_uavs
    total_action_dim = action_dim * num_uavs

    uav_list = []
    for cfg in uav_configs:
        uav = RL.UAVAgent(
            uav_id=cfg["id"],
            initial_pos=cfg["pos"],
            initial_v=cfg["initial_v"],
            initial_phi=cfg["phi"],
            step=1.0,
            total_state_dim=total_state_dim,
            total_action_dim=total_action_dim
        )

        uav.state_dim = state_dim
        uav.brain = RL.MADDPG(state_dim, action_dim, uav.max_action_tensor, total_state_dim, total_action_dim)

        uav_list.append(uav)

    # 初始化全局 Buffer
    global_buffer = RL.MultiAgentReplayBuffer(50000, num_uavs,
                                              [state_dim] * num_uavs,
                                              [action_dim] * num_uavs)

    reward_history = []
    noise_history = []
    metrics_history = []
    detailed_history = []
    MAX_EPISODES = 16000
    MAX_STEPS = 100
    BATCH_SIZE = 512
    noise_std = 0.3
    min_noise = 0.05
    noise_decay = 0.996
    eval_interval = 200

    output_root = os.path.join(os.path.dirname(__file__), "outputs")
    models_dir = os.path.join(output_root, "models")
    eval_dir = os.path.join(output_root, "eval")
    ensure_dir(models_dir)
    ensure_dir(eval_dir)

    config = {
        "MAP_SIZE": map_size,
        "N": num_uavs,
        "MAX_STEPS": MAX_STEPS,
        "MAX_EPISODES": MAX_EPISODES,
        "actor_lr": 1e-4,
        "critic_lr": 1e-3,
        "gamma": 0.99,
        "seed": None,
        "eval_interval": eval_interval,
        "curriculum": {
            "stage1_episodes": 3000,
            "stage2_episodes": 5000,
            "stage2_pos_jitter": 50.0,
            "stage2_angle_jitter_deg": 30.0
        }
    }
    save_config(output_root, config)

    best_record_path = os.path.join(output_root, "best_record.json")
    best_record = {"episode": -1, "eval_H_final": float("inf"), "eval_delta_H": -float("inf")}
    if os.path.exists(best_record_path):
        with open(best_record_path, "r", encoding="utf-8") as f:
            best_record = json.load(f)

    base_tarcfgs = [
        {
            "ID": 1,
            "pos": (500.0, 500.0),
            "priority": 1,
            "v_range": (15, 60),
            "theta_range": (-np.pi, np.pi),
            "initial_v": 30.0,
            "initial_phi": 45.0,
            "enable_boundary_reflection": False
        },
        {
            "ID": 2,
            "pos": (1500.0, 1500.0),
            "priority": 1,
            "v_range": (15, 60),
            "theta_range": (-np.pi, np.pi),
            "initial_v": 30.0,
            "initial_phi": 225.0,
            "enable_boundary_reflection": False
        }
    ]

    # 训练主循环
    print("开始训练 Improved MADDPG")
    for episode in range(MAX_EPISODES):
        if episode < 3000:
            tarcfgs = base_tarcfgs
        elif episode < 8000:
            tarcfgs = generate_random_tarcfgs(
                2,
                base_configs=base_tarcfgs,
                pos_jitter=50.0,
                angle_jitter_deg=30.0,
                full_random=False,
                enable_boundary_reflection=False
            )
        else:
            tarcfgs = generate_random_tarcfgs(2, full_random=True, enable_boundary_reflection=True)
        predictors, real_targets = init_predictors_targets(map_size, obstacles, tarcfgs)

        # 重置无人机部分参数
        for i, uav in enumerate(uav_list):
            uav.pos = np.array(uav_configs[i]["pos"])
            uav.v = uav_configs[i]["initial_v"]
            uav.phi = np.radians(uav_configs[i]["phi"])
            uav.assigned_task_coords = None
            uav.reset_episode()

        episode_reward = 0
        detect_streak_sum = 0.0
        lost_streak_sum = 0.0
        lock_time_steps = 0
        lock_event_count = 0
        target_covered_steps = np.zeros(len(predictors))
        tracking_event_counts = {
            "acquire": 0,
            "keep": 0,
            "lost": 0,
            "unlock": 0,
            "reacquire": 0
        }
        reward_component_sums = {
            "r_info": 0.0,
            "r_track": 0.0,
            "r_action": 0.0,
            "r_boundary": 0.0,
            "r_collision": 0.0,
            "r_uncertainty": 0.0
        }
        clip_count = 0

        # 预先获取初始观测
        for uav in uav_list:
            uav.last_obs = uav.get_observation(map_size, obs_map, uav_list, predictors)

        for t in range(MAX_STEPS):
            local_entropy_before = compute_local_entropy_for_uavs(uav_list, predictors)

            action_list = []
            obs_list = []

            for uav in uav_list:
                obs = uav.last_obs
                obs_list.append(obs)

                raw_action = uav.brain.select_action(obs)
                noise = np.random.normal(0, noise_std, size=2)
                action = np.clip(raw_action + noise, -1.0, 1.0)
                action_list.append(action)

            # 执行动作 & 环境更新
            next_obs_list = []
            reward_list = []
            done_list = []
            uav_detected_any = [False] * len(uav_list)
            uav_detected_targets = [set() for _ in range(len(uav_list))]
            uav_detected_nearest = [-1] * len(uav_list)
            uav_detected_nearest_dist = [float("inf")] * len(uav_list)

            # 无人机运动
            for i, uav in enumerate(uav_list):
                uav.state_update(action_list[i], map_size, obs_map)

            # 预测器更新
            meas_noise_std = 3.0
            innovation_norm = np.zeros(len(predictors))
            for i, p in enumerate(predictors):
                real_pos = real_targets[i].state[0:2]

                det_res = []
                sum_z = 0.0
                sum_detected = 0
                for u_idx, u in enumerate(uav_list):
                    dist = np.linalg.norm(u.pos - real_pos)
                    is_detected = (dist < 250.0) and (np.random.rand() < 0.9)
                    temp_state = {"detected": is_detected, "measurement": None, "uavpos": u.pos, "uavdp": u.detecct_p}
                    if is_detected:
                        uav_detected_any[u_idx] = True
                        uav_detected_targets[u_idx].add(i)
                        if dist < uav_detected_nearest_dist[u_idx]:
                            uav_detected_nearest_dist[u_idx] = dist
                            uav_detected_nearest[u_idx] = i
                        temp_state["measurement"] = real_pos + np.random.randn(2) * meas_noise_std
                        sum_z += temp_state["measurement"]
                        sum_detected += 1
                    det_res.append(temp_state)

                if sum_detected == 0:
                    z_average = None
                    p.step_update(None, innovation_norm[i], det_res)
                    innovation_norm[i] = 0.0
                else:
                    z_average = sum_z / sum_detected
                    p.step_update(z_average, innovation_norm[i], det_res)
                    innovation_norm[i] = np.linalg.norm(z_average - p.state_si[0:2])

                # 真实目标移动
                real_targets[i].step_forward()

            local_entropy_after = compute_local_entropy_for_uavs(uav_list, predictors)

            tracking_events = []
            for i, uav in enumerate(uav_list):
                event = uav.update_tracking_state(uav_detected_targets[i], uav_detected_nearest[i])
                tracking_events.append(event)
                if event in tracking_event_counts:
                    tracking_event_counts[event] += 1
                if event in {"acquire", "reacquire"}:
                    lock_event_count += 1

            lock_count = {}
            for uav in uav_list:
                if uav.tracking_target_id is not None:
                    lock_count[uav.tracking_target_id] = lock_count.get(uav.tracking_target_id, 0) + 1

            lock_time_steps += sum(1 for uav in uav_list if uav.tracking_target_id is not None)
            detect_streak_sum += sum(uav.detect_streak for uav in uav_list)
            lost_streak_sum += sum(uav.lost_streak for uav in uav_list)
            for target_idx in range(len(predictors)):
                if any(target_idx in targets for targets in uav_detected_targets):
                    target_covered_steps[target_idx] += 1

            # 观测下一帧 & 计算奖励
            for i, uav in enumerate(uav_list):
                next_obs = uav.get_observation(map_size, obs_map, uav_list, predictors)
                next_obs_list.append(next_obs)
                uav.last_obs = next_obs

                r, reward_components = uav.calculate_reward(
                    prev_entropy=local_entropy_before[i],
                    curr_entropy=local_entropy_after[i],
                    detected_any=uav_detected_any[i],
                    detected_targets=uav_detected_targets[i],
                    nearest_target=uav_detected_nearest[i],
                    same_target_lock_count=lock_count.get(uav.tracking_target_id, 0),
                    action=action_list[i],
                    map_size=map_size,
                    obstacles_map=obs_map,
                    all_uavs=uav_list,
                    predictors=predictors,
                    return_components=True
                )

                r_final = np.clip(r, -5.0, 5.0)
                if r_final != r:
                    clip_count += 1

                for key in reward_component_sums:
                    reward_component_sums[key] += reward_components[key]

                reward_list.append(r_final)

                d = False
                if t == MAX_STEPS - 1:
                    d = True
                done_list.append(d)

                episode_reward += r_final

            global_buffer.add(obs_list, action_list, reward_list, next_obs_list, done_list)

            if t % 5 == 0 and global_buffer.size > BATCH_SIZE:
                for _ in range(2):
                    RL.train_centralized(uav_list, global_buffer, BATCH_SIZE)

        if episode < 12999:
            noise_std = max(min_noise, noise_std * noise_decay)
        elif episode == 12999:
            noise_std = 0.05
        else:
            if (episode - 13000 + 1) % 100 == 0:
                noise_std *= 0.9
        avg_reward = episode_reward / len(uav_list)
        reward_history.append(avg_reward)
        noise_history.append(noise_std)
        print(f"Episode {episode + 1}/{MAX_EPISODES} | Avg Reward: {avg_reward:.2f} | Noise: {noise_std:.3f}")

        total_samples = MAX_STEPS * len(uav_list)
        avg_detect_streak = detect_streak_sum / total_samples if total_samples > 0 else 0.0
        avg_lost_streak = lost_streak_sum / total_samples if total_samples > 0 else 0.0
        lock_time_ratio = lock_time_steps / total_samples if total_samples > 0 else 0.0
        target_covered_ratio = float(np.sum(target_covered_steps) / (MAX_STEPS * len(predictors))) if predictors else 0.0
        metrics_history.append({
            "episode": episode + 1,
            "avg_detect_streak": avg_detect_streak,
            "avg_lost_streak": avg_lost_streak,
            "lock_count": lock_event_count,
            "lock_time_ratio": lock_time_ratio,
            "target_covered_ratio": target_covered_ratio
        })
        detailed_history.append({
            "episode": episode + 1,
            "avg_reward": avg_reward,
            "avg_r_info": reward_component_sums["r_info"] / total_samples,
            "avg_r_track": reward_component_sums["r_track"] / total_samples,
            "avg_r_action": reward_component_sums["r_action"] / total_samples,
            "avg_r_boundary": reward_component_sums["r_boundary"] / total_samples,
            "avg_r_collision": reward_component_sums["r_collision"] / total_samples,
            "avg_r_uncertainty": reward_component_sums["r_uncertainty"] / total_samples,
            "clip_ratio": clip_count / total_samples if total_samples > 0 else 0.0,
            "acquire": tracking_event_counts["acquire"],
            "keep": tracking_event_counts["keep"],
            "lost": tracking_event_counts["lost"],
            "unlock": tracking_event_counts["unlock"],
            "reacquire": tracking_event_counts["reacquire"]
        })

        if (episode + 1) % 200 == 0:
            save_reward_history(output_root, reward_history, noise_history)
            save_tracking_metrics(output_root, metrics_history)
            save_detailed_rewards(output_root, detailed_history)

        if (episode + 1) % 50 == 0:
            for uav in uav_list:
                torch.save(uav.brain.actor.state_dict(), os.path.join(models_dir, f"uav{uav.ID}_actor_{episode + 1}.pth"))

        if (episode + 1) % eval_interval == 0:
            eval_h_final, eval_delta_h = run_evaluation(
                eval_dir, episode + 1, uav_list, uav_configs, map_size, obs_map, obstacles, tarcfgs, MAX_STEPS
            )

            if eval_h_final < best_record["eval_H_final"] or eval_delta_h > best_record["eval_delta_H"]:
                for uav in uav_list:
                    torch.save(uav.brain.actor.state_dict(), os.path.join(models_dir, f"best_actor_uav{uav.ID}.pth"))

                best_record = {
                    "episode": episode + 1,
                    "eval_H_final": eval_h_final,
                    "eval_delta_H": eval_delta_h
                }
                with open(best_record_path, "w", encoding="utf-8") as f:
                    json.dump(best_record, f, indent=2, ensure_ascii=False)

    save_reward_history(output_root, reward_history, noise_history)
    save_tracking_metrics(output_root, metrics_history)
    save_detailed_rewards(output_root, detailed_history)


if __name__ == "__main__":
    train_with_improvements()
