import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

import maddpg as RL
import targetpre as tp

os.environ["OMP_NUM_THREADS"] = "1"
matplotlib.use("Agg")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def moving_average(data, window):
    if len(data) < window:
        return np.array([])
    kernel = np.ones(window) / window
    return np.convolve(np.array(data), kernel, mode="valid")


def plot_reward_curve(output_dir, reward_history):
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
    plt.savefig(os.path.join(output_dir, "reward_curve.png"))
    plt.close()


def save_reward_csv(output_dir, reward_history, noise_history):
    csv_path = os.path.join(output_dir, "reward_history.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("episode,avg_reward,noise_std\n")
        for idx, reward in enumerate(reward_history, start=1):
            noise = noise_history[idx - 1] if idx - 1 < len(noise_history) else None
            f.write(f"{idx},{reward:.6f},{noise:.6f}\n")


def plot_trajectories(output_dir, map_size, obs_map, uav_trajectories, target_trajectories, detection_points):
    plt.figure(figsize=(8, 8))
    if obs_map is not None:
        plt.imshow(obs_map == -1, cmap="gray_r", origin="upper")

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
        plt.plot(traj[:, 0], traj[:, 1], linestyle="--", linewidth=2.5, label=f"Target {idx + 1}")

    if detection_points:
        det_points = np.array(detection_points)
        plt.scatter(det_points[:, 0], det_points[:, 1], s=20, c="red", label="Detection")

    plt.xlim(0, map_size[1])
    plt.ylim(0, map_size[0])
    plt.title("UAV Trajectories & Detections")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory.png"))
    plt.close()


def run_original_headless():
    map_size = (2000, 2000)
    obstacles = [(600, 600), (600, 601), (601, 600), (601, 601),
                 (1200, 1200), (1200, 1201), (1201, 1200), (1201, 1201)]

    obs_map = np.zeros(map_size)
    for r, c in obstacles:
        if 0 <= r < map_size[0] and 0 <= c < map_size[1]:
            obs_map[r, c] = -1

    tarcfgs = [
        {"ID": 1, "pos": (500.0, 500.0), "priority": 1, "v_range": (15, 60),
         "theta_range": (np.radians(15), np.radians(60)), "initial_v": 20, "initial_phi": 45},
        {"ID": 2, "pos": (1500.0, 1500.0), "priority": 1, "v_range": (15, 60),
         "theta_range": (np.radians(225), np.radians(270)), "initial_v": 20, "initial_phi": -120}
    ]

    state_dim = 15
    action_dim = 2
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
            total_action_dim=total_action_dim,
        )

        uav.state_dim = state_dim
        uav.brain = RL.MADDPG(state_dim, action_dim, uav.max_action_tensor, total_state_dim, total_action_dim)

        uav_list.append(uav)

    global_buffer = RL.MultiAgentReplayBuffer(50000, num_uavs,
                                              [state_dim] * num_uavs,
                                              [action_dim] * num_uavs)

    reward_history = []
    noise_history = []
    MAX_EPISODES = 15000
    MAX_STEPS = 100
    BATCH_SIZE = 512
    noise_std = 0.3
    min_noise = 0.05
    noise_decay = 0.996

    reward_scaler = RL.RewardScaler(shape=(1,))

    output_root = os.path.join(os.path.dirname(__file__), "original_outputs")
    ensure_dir(output_root)

    print("开始训练 Original MADDPG (headless)")
    for episode in range(MAX_EPISODES):
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
            r = tp.RealTarget(ID, priority, start_belief, initial_v, initial_phi)

            predictors.append(p)
            real_targets.append(r)

        for i, uav in enumerate(uav_list):
            uav.pos = np.array(uav_configs[i]["pos"])
            uav.v = uav_configs[i]["initial_v"]
            uav.phi = np.radians(uav_configs[i]["phi"])
            uav.assigned_task_coords = None

        episode_reward = 0
        uav_trajectories = [[] for _ in uav_list]
        target_trajectories = [[] for _ in real_targets]
        detection_points = []

        for uav in uav_list:
            uav.last_obs = uav.get_observation(map_size, obs_map, uav_list, predictors)

        for t in range(MAX_STEPS):
            entropy_before = sum(p.get_entropy() for p in predictors)

            action_list = []
            obs_list = []

            for uav in uav_list:
                obs = uav.last_obs
                obs_list.append(obs)

                raw_action = uav.brain.select_action(obs)
                noise = np.random.normal(0, noise_std, size=2)
                action = np.clip(raw_action + noise, -1.0, 1.0)
                action_list.append(action)

            next_obs_list = []
            reward_list = []
            done_list = []
            uav_detection_states = [False] * len(uav_list)

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
                        uav_detection_states[u_idx] = True
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

            entropy_after = sum(p.get_entropy() for p in predictors)

            for i, uav in enumerate(uav_list):
                next_obs = uav.get_observation(map_size, obs_map, uav_list, predictors)
                next_obs_list.append(next_obs)
                uav.last_obs = next_obs

                r = uav.calculate_reward(
                    prev_entropy=entropy_before,
                    curr_entropy=entropy_after,
                    is_detected=uav_detection_states[i],
                    action=action_list[i],
                    map_size=map_size,
                    obstacles_map=obs_map,
                    all_uavs=uav_list,
                )

                r_input = np.array([r])
                r_norm = reward_scaler(r_input)[0]
                r_final = np.clip(r_norm, -5.0, 5.0)

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

        noise_std = max(min_noise, noise_std * noise_decay)
        avg_reward = episode_reward / len(uav_list)
        reward_history.append(avg_reward)
        noise_history.append(noise_std)
        print(f"Episode {episode + 1}/{MAX_EPISODES} | Avg Reward: {avg_reward:.2f} | Noise: {noise_std:.3f}")

        if (episode + 1) % 50 == 0:
            for uav in uav_list:
                torch.save(uav.brain.actor.state_dict(), os.path.join(output_root, f"uav{uav.ID}_actor_{episode + 1}.pth"))

        if (episode + 1) % 200 == 0:
            plot_trajectories(output_root, map_size, obs_map, uav_trajectories, target_trajectories, detection_points)

    save_reward_csv(output_root, reward_history, noise_history)
    plot_reward_curve(output_root, reward_history)


if __name__ == "__main__":
    run_original_headless()
