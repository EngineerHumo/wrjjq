import os
import numpy as np
import matplotlib
import torch
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# 引入你的自定义模块
import targetpre as tp
# import task_generation as tg
# import task_allocation as ta
import maddpg as RL
# from evaluator import PredictionEvaluator

os.environ["OMP_NUM_THREADS"] = "1"
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def get_gps_coordinates():
    """
    获取无人机位置
    """
    temp = [(42, 183), (200, 183), (132, 135)]
    return np.array(temp)


def run_yolo_detection():
    """
    获取传感器探测结果
    """
    return np.array([True, False, True])

# 辅助函数
def get_sensor_readings(uav_list, real_target_pos, detection_radius=25.0, p_d=0.8):
    results = []
    for uav in uav_list:
        dist = np.linalg.norm(uav.pos - real_target_pos)
        if dist <= detection_radius:
            results.append(np.random.random() < p_d)
        else:
            results.append(False)
    return results


# 主要训练逻辑
def test7_train():
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
        {"ID": 1, "pos": (500.0, 500.0), "priority": 1, "v_range":(15,60), "theta_range":(np.radians(15),np.radians(60)), "initial_v":20, "initial_phi":45},
        {"ID": 2, "pos": (1500.0, 1500.0), "priority": 1, "v_range": (15,60), "theta_range": (np.radians(225),np.radians(270)), "initial_v": 20, "initial_phi": -120}
    ]

    # 初始化 MADDPG 系统
    state_dim = 15
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
            uav_id=cfg['id'],
            initial_pos=cfg['pos'],
            initial_v=cfg['initial_v'], initial_phi=cfg['phi'],
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
    MAX_EPISODES = 15000
    MAX_STEPS = 100
    BATCH_SIZE = 512
    noise_std = 0.3
    min_noise = 0.05
    noise_decay = 0.996

    reward_scaler = RL.RewardScaler(shape=(1,))  # 初始化奖励归一化器

    # 训练主循环
    print("开始训练 Test 7")
    for episode in range(MAX_EPISODES):
        # 重置目标
        # 初始化 预测器 和 真实目标
        predictors = []
        real_targets = []
        for cfg in tarcfgs:
            # 预测器初始
            start_belief = cfg["pos"]
            v_range = cfg["v_range"]
            theta_range = cfg["theta_range"]
            ID = cfg["ID"]
            priority = cfg["priority"]
            p = tp.TargetPredictor(map_size, obstacles, v_range, theta_range, start_belief, ID=ID, priority=priority)

            # 真实目标
            initial_v = cfg["initial_v"]
            initial_phi = cfg["initial_phi"]
            r = tp.RealTarget(ID, priority, start_belief, initial_v, initial_phi)

            predictors.append(p)
            real_targets.append(r)

        # 重置无人机部分参数
        for i, uav in enumerate(uav_list):
            uav.pos = np.array(uav_configs[i]["pos"])
            uav.v = uav_configs[i]["initial_v"]
            uav.phi = np.radians(uav_configs[i]["phi"])
            uav.assigned_task_coords = None


        episode_reward = 0

        # 预先获取初始观测
        for uav in uav_list:
            uav.last_obs = uav.get_observation(map_size, obs_map, uav_list, predictors)

        for t in range(MAX_STEPS):
            # 记录动作前的全场总熵
            entropy_before = sum([p.get_entropy() for p in predictors])

            # MADDPG
            action_list = []
            obs_list = []

            for uav in uav_list:
                obs = uav.last_obs
                obs_list.append(obs)

                raw_action = uav.brain.select_action(obs)
                noise = np.random.normal(0, noise_std, size=2)
                action = np.clip(raw_action + noise, -1.0, 1.0)  # 增加噪声以鼓励探索
                action_list.append(action)

            # 执行动作 & 环境更新
            next_obs_list = []
            reward_list = []
            done_list = []
            uav_detection_states = [False] * len(uav_list)

            # 无人机运动
            current_uav_pos_list = []
            for i, uav in enumerate(uav_list):
                uav.state_update(action_list[i], map_size, obs_map)
                current_uav_pos_list.append(uav.pos)

            # 预测器更新
            meas_noise_std = 3.0
            innovation_norm = np.zeros(len(predictors))
            # alltar_uav_detection_states = []  # 储存所有的探测状态
            for i, p in enumerate(predictors):
                real_pos = real_targets[i].state[0:2]

                # 模拟传感器 (探测半径250m)
                det_res = []  # 储存每个无人机对当前目标的探测状态
                sum_z = 0.0
                sum_detected = 0
                for u_idx, u in enumerate(uav_list):
                    dist = np.linalg.norm(u.pos - real_pos)
                    is_detected = (dist < 250.0) and (np.random.rand() < 0.9)
                    # det_res.append(is_detected)
                    temp_state = {"detected": is_detected, "measurement": None, "uavpos": u.pos, "uavdp": u.detecct_p}  # 当前无人机对当前目标的探测状态
                    if is_detected:
                        uav_detection_states[u_idx] = True
                        temp_state["measurement"] =real_pos + np.random.randn(2) * meas_noise_std  # z
                        sum_z += temp_state["measurement"]
                        sum_detected += 1
                    det_res.append(temp_state)
                # alltar_uav_detection_states.append(det_res)

                # 预测器更新
                if sum_detected == 0:
                    z_average = None
                    p.step_update(None, innovation_norm[i], det_res)
                    innovation_norm[i] = 0.0
                else:
                    z_average = sum_z / sum_detected  # 平均观测数据
                    p.step_update(z_average, innovation_norm[i], det_res)
                    innovation_norm[i] = np.linalg.norm(z_average - p.state_si[0:2])  # 更新新息

                # 真实目标移动
                real_targets[i].step_forward()

            # 记录动作后的全场总熵
            entropy_after = sum([p.get_entropy() for p in predictors])

            # 观测下一帧 & 计算奖励
            for i, uav in enumerate(uav_list):
                # 获取新观测 (包含更新后的扇区感知)
                next_obs = uav.get_observation(map_size, obs_map, uav_list, predictors)
                next_obs_list.append(next_obs)
                uav.last_obs = next_obs

                # 计算奖励 (核心驱动力：熵减)
                r = uav.calculate_reward(
                    prev_entropy=entropy_before,
                    curr_entropy=entropy_after,
                    is_detected=uav_detection_states[i],
                    action=action_list[i],
                    map_size=map_size,
                    obstacles_map=obs_map,
                    all_uavs=uav_list
                )

                r_input = np.array([r])
                # 通过 Scaler 计算归一化后的奖励
                r_norm = reward_scaler(r_input)[0]
                r_final = np.clip(r_norm, -5.0, 5.0)

                reward_list.append(r_final)

                # 终止条件
                d = False
                if t == MAX_STEPS - 1: d = True
                done_list.append(d)

                episode_reward += r_final

            global_buffer.add(obs_list, action_list, reward_list, next_obs_list, done_list)

            if t % 5 == 0 and global_buffer.size > BATCH_SIZE:
                for _ in range(2):
                    RL.train_centralized(uav_list, global_buffer, BATCH_SIZE)

        # End of Episode
        noise_std = max(min_noise, noise_std * noise_decay)
        avg_reward = episode_reward / len(uav_list)
        reward_history.append(avg_reward)
        print(f"Episode {episode + 1}/{MAX_EPISODES} | Avg Reward: {avg_reward:.2f} | Noise: {noise_std:.3f}")

        # 保存模型
        if (episode + 1) % 50 == 0:
            if not os.path.exists('models'): os.makedirs('models')
            for uav in uav_list:
                torch.save(uav.brain.actor.state_dict(), f"models/uav{uav.ID}_actor_{episode + 1}.pth")

    # 绘制奖励曲线
    plt.figure()
    plt.plot(reward_history)
    plt.title("Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()



if __name__ == "__main__":
    if not os.path.exists('models'):
        os.makedirs('models')
    # 运行训练
    # test_clustering_accuracy()
    # test5_train()
    test7_train()