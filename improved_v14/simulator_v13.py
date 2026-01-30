import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

import maddpg_v13 as RL
import targetpre_v13 as tp

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
        r = tp.RealTarget(
            ID,
            priority,
            start_belief,
            initial_v,
            initial_phi,
            v_range=v_range,
            phi_range=theta_range
        )

        predictors.append(p)
        real_targets.append(r)
    return predictors, real_targets


def sample_uav_init_configs(map_size, num_uavs, margin=350, min_sep=200, v_range=(12, 18), phi_jitter_deg=30, rng=None):
    if rng is None:
        rng = np.random
    M, N = map_size
    cx, cy = N / 2.0, M / 2.0

    regions = [
        (margin, N / 2 - margin, margin, M / 2 - margin),         # SW
        (N / 2 + margin, N - margin, M / 2 + margin, M - margin),  # NE
        (margin, N / 2 - margin, M / 2 + margin, M - margin),      # NW
        (N / 2 + margin, N - margin, margin, M / 2 - margin),      # SE
        (N / 3, 2 * N / 3, N / 3, 2 * M / 3),                      # center-ish
    ]
    regions = regions[:num_uavs]

    configs = []
    positions = []

    for i in range(num_uavs):
        for _ in range(200):
            x0, x1, y0, y1 = regions[i]
            x = rng.uniform(x0, x1)
            y = rng.uniform(y0, y1)
            pos = np.array([x, y])

            if any(np.linalg.norm(pos - p) < min_sep for p in positions):
                continue

            phi = np.degrees(np.arctan2(cy - y, cx - x)) + rng.uniform(-phi_jitter_deg, phi_jitter_deg)
            v = rng.uniform(v_range[0], v_range[1])

            configs.append({"pos": [float(x), float(y)], "phi": float(phi), "initial_v": float(v)})
            positions.append(pos)
            break
        else:
            x = rng.uniform(margin, N - margin)
            y = rng.uniform(margin, M - margin)
            phi = rng.uniform(-180, 180)
            v = rng.uniform(v_range[0], v_range[1])
            configs.append({"pos": [float(x), float(y)], "phi": float(phi), "initial_v": float(v)})

    return configs


def sample_target_init_configs(base_tarcfgs, map_size, margin=300, min_sep=500, rng=None):
    if rng is None:
        rng = np.random
    M, N = map_size
    targets = []
    pos_list = []

    for cfg in base_tarcfgs:
        for _ in range(200):
            x = rng.uniform(margin, N - margin)
            y = rng.uniform(margin, M - margin)
            pos = np.array([x, y])
            if any(np.linalg.norm(pos - p) < min_sep for p in pos_list):
                continue
            phi = rng.uniform(-180, 180)
            v = rng.uniform(8.0, 15.0)

            new_cfg = dict(cfg)
            new_cfg["pos"] = (float(x), float(y))
            new_cfg["initial_phi"] = float(phi)
            new_cfg["initial_v"] = float(v)
            targets.append(new_cfg)
            pos_list.append(pos)
            break
        else:
            targets.append(cfg)

    return targets


def apply_target_boundary(real_target, map_size):
    M, N = map_size
    eps = 1e-3
    max_x = N - eps
    max_y = M - eps

    x, y, vx, vy = real_target.state[0], real_target.state[1], real_target.state[2], real_target.state[3]
    hit = False

    if x < 0 or x > max_x:
        hit = True
        x = np.clip(x, 0.0, max_x)
        vx = -vx
    if y < 0 or y > max_y:
        hit = True
        y = np.clip(y, 0.0, max_y)
        vy = -vy

    if hit:
        real_target.state[0] = x
        real_target.state[1] = y
        real_target.state[2] = vx
        real_target.state[3] = vy
        real_target.state[4] = np.arctan2(vy, vx)


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
    boundary = plt.Rectangle(
        (0, 0),
        map_size[1],
        map_size[0],
        fill=False,
        edgecolor="black",
        linestyle="--",
        linewidth=1.5,
        label="Map Boundary"
    )
    plt.gca().add_patch(boundary)
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


def run_evaluation(eval_dir, episode, uav_list, uav_configs, map_size, obs_map, obstacles, tarcfgs, max_steps, random_init=False):
    rng = np.random.RandomState(episode)
    if random_init:
        tarcfgs_ep = sample_target_init_configs(tarcfgs, map_size, rng=rng)
        uav_init = sample_uav_init_configs(map_size, len(uav_list), rng=rng)
    else:
        tarcfgs_ep = tarcfgs
        uav_init = uav_configs

    predictors, real_targets = init_predictors_targets(map_size, obstacles, tarcfgs_ep)
    for i, uav in enumerate(uav_list):
        uav.pos = np.array(uav_init[i]["pos"], dtype=float)
        uav.v = uav_init[i]["initial_v"]
        uav.phi = np.radians(uav_init[i]["phi"])

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
            apply_target_boundary(real_targets[i], map_size)

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

    # 定义目标配置
    tarcfgs = [
        {"ID": 1, "pos": (500.0, 500.0), "priority": 1, "v_range": (7.5, 30),
         "theta_range": (np.radians(15), np.radians(60)), "initial_v": 10, "initial_phi": 45},
        {"ID": 2, "pos": (1500.0, 1500.0), "priority": 1, "v_range": (7.5, 30),
         "theta_range": (np.radians(225), np.radians(270)), "initial_v": 10, "initial_phi": -120}
    ]

    # 初始化 MADDPG 系统
    state_dim = 17
    action_dim = 2
    # 无人机参数配置
    uav_configs = [
        {"id": 1, "pos": [200, 200], "phi": 45, "initial_v": 15},
        {"id": 2, "pos": [1800, 1800], "phi": 225, "initial_v": 15},
        {"id": 3, "pos": [200, 1800], "phi": 315, "initial_v": 15},
        {"id": 4, "pos": [1800, 200], "phi": 135, "initial_v": 15},
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
    MAX_EPISODES = 20000
    MAX_STEPS = 100
    BATCH_SIZE = 512
    noise_std = 0.3
    min_noise = 0.05
    noise_decay = 0.996
    eval_interval = 200
    training_curve_interval = 200

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
        "eval_interval": eval_interval
    }
    save_config(output_root, config)

    best_record_path = os.path.join(output_root, "best_record.json")
    best_record = {"episode": -1, "eval_H_final": float("inf"), "eval_delta_H": -float("inf")}
    if os.path.exists(best_record_path):
        with open(best_record_path, "r", encoding="utf-8") as f:
            best_record = json.load(f)

    # 训练主循环
    print("开始训练 Improved MADDPG")
    for episode in range(MAX_EPISODES):
        rng = np.random
        if episode < 2000:
            use_fixed = (rng.rand() < 0.7)
        elif episode < 8000:
            use_fixed = (rng.rand() < 0.3)
        else:
            use_fixed = False

        tarcfgs_ep = tarcfgs if use_fixed else sample_target_init_configs(tarcfgs, map_size, rng=rng)
        predictors, real_targets = init_predictors_targets(map_size, obstacles, tarcfgs_ep)

        uav_init = sample_uav_init_configs(map_size, num_uavs, rng=rng)
        for i, uav in enumerate(uav_list):
            uav.pos = np.array(uav_init[i]["pos"], dtype=float)
            uav.v = float(uav_init[i]["initial_v"])
            uav.phi = np.radians(uav_init[i]["phi"])
            uav.assigned_task_coords = None
            uav.prev_detected = False
            uav.steps_since_detect = 0
            uav.prev_out_of_bounds = False
            uav.steps_out_of_bounds = 0
            uav.prev_distance_to_bounds = 0.0
            uav.distance_to_bounds = 0.0
            uav.hit_boundary = False
            uav.force_done = False

        episode_reward = 0

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
            uav_detection_states = [False] * len(uav_list)

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
                        uav_detection_states[u_idx] = True
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
                apply_target_boundary(real_targets[i], map_size)

            local_entropy_after = compute_local_entropy_for_uavs(uav_list, predictors)

            # 观测下一帧 & 计算奖励
            for i, uav in enumerate(uav_list):
                next_obs = uav.get_observation(map_size, obs_map, uav_list, predictors)
                next_obs_list.append(next_obs)
                uav.last_obs = next_obs

                r = uav.calculate_reward(
                    prev_entropy=local_entropy_before[i],
                    curr_entropy=local_entropy_after[i],
                    is_detected=uav_detection_states[i],
                    action=action_list[i],
                    map_size=map_size,
                    obstacles_map=obs_map,
                    all_uavs=uav_list,
                    target_predictors=predictors
                )

                r_final = float(np.clip(r, -20.0, 20.0))
                reward_list.append(r_final)

                episode_reward += r_final

            episode_terminated = any(getattr(u, "force_done", False) for u in uav_list)
            for uav in uav_list:
                d = False
                if episode_terminated or getattr(uav, "force_done", False) or (t == MAX_STEPS - 1):
                    d = True
                done_list.append(d)

            global_buffer.add(obs_list, action_list, reward_list, next_obs_list, done_list)

            if t % 5 == 0 and global_buffer.size > BATCH_SIZE:
                for _ in range(2):
                    RL.train_centralized(uav_list, global_buffer, BATCH_SIZE)

            if episode_terminated:
                break

        if episode + 1 < 15000:
            noise_std = max(min_noise, noise_std * noise_decay)
        else:
            if episode + 1 == 15000:
                noise_std = min_noise
            if (episode + 1 - 15000) % 100 == 0:
                noise_std *= 0.9
        avg_reward = episode_reward / len(uav_list)
        reward_history.append(avg_reward)
        noise_history.append(noise_std)
        print(f"Episode {episode + 1}/{MAX_EPISODES} | Avg Reward: {avg_reward:.2f} | Noise: {noise_std:.3f}")

        if (episode + 1) % 50 == 0:
            for uav in uav_list:
                torch.save(uav.brain.actor.state_dict(), os.path.join(models_dir, f"uav{uav.ID}_actor_{episode + 1}.pth"))

        if (episode + 1) % eval_interval == 0:
            eval_h_final, eval_delta_h = run_evaluation(
                eval_dir, episode + 1, uav_list, uav_configs, map_size, obs_map, obstacles, tarcfgs, MAX_STEPS, random_init=True
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

        if (episode + 1) % training_curve_interval == 0 or episode == MAX_EPISODES - 1:
            save_reward_history(output_root, reward_history, noise_history)


if __name__ == "__main__":
    train_with_improvements()
