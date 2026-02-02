import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import time
from collections import deque

# 导入前两部分模块
# 假设你已经保存为 uav_env.py 和 algo.py
from uav_env import UAVSwarmEnv, Config
from algo import MADDPG, AlgoConfig

# ===========================
# 1. 训练超参数设置
# ===========================
MAX_EPISODES = 2000  # 总训练回合数
MAX_STEPS = 200  # 单回合最大步数 (对应论文中的 Time Step)
EVAL_INTERVAL = 50  # 每多少回合评估一次
SAVE_DIR = "./models"  # 模型保存路径
RESULT_DIR = "./results"  # 结果保存路径

# 确保目录存在
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ===========================
# 2. 评估函数 (无噪声测试)
# ===========================
def evaluate(env, maddpg, n_episodes=5):
    """
    在无噪声模式下运行几个回合，评估当前策略性能
    返回: 平均奖励, 平均覆盖率
    """
    avg_reward = 0.0
    avg_coverage = 0.0

    for _ in range(n_episodes):
        obs_n, _ = env.reset()
        episode_reward = 0

        for step in range(MAX_STEPS):
            # 获取动作 (noise_std=0 表示纯利用，不探索)
            actions = maddpg.select_actions(obs_n, noise_std=0.0)

            # 环境步进
            next_obs_n, rewards_n, terminated, truncated, _ = env.step(actions)

            # 累加奖励 (取团队平均或总和)
            episode_reward += np.sum(rewards_n)
            obs_n = next_obs_n

        # 计算本回合最终覆盖率
        # 覆盖率 = 覆盖网格数 / 总网格数
        coverage_rate = np.sum(env.global_map_cover) / (Config.GRID_ROWS * Config.GRID_COLS)

        avg_reward += episode_reward
        avg_coverage += coverage_rate

    return avg_reward / n_episodes, avg_coverage / n_episodes


# ===========================
# 3. 训练主程序
# ===========================
if __name__ == "__main__":
    # 初始化环境
    env = UAVSwarmEnv()

    # 获取维度信息
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # 全局状态维度 (Critic用): 拼接所有观测
    # 注意: 如果在 algo.py 中 Critic 输入包含 map 特征，这里需要调整
    # 这里我们简化为所有 agent 的 obs 拼接
    global_obs_dim = obs_dim * Config.N_UAV

    # 初始化 MADDPG 控制器
    maddpg = MADDPG(Config.N_UAV, obs_dim, act_dim, global_obs_dim)

    # 记录指标
    scores = []
    coverages = []
    best_coverage = 0.0

    print(f"Start Training: UAVs={Config.N_UAV}, Map={Config.MAP_SIZE}x{Config.MAP_SIZE}...")

    for i_episode in range(1, MAX_EPISODES + 1):
        obs_n, _ = env.reset()
        global_obs = env.get_global_state()  # 获取初始全局状态

        episode_reward = 0

        # 噪声衰减 (Curriculum Learning 思想: 初期探索，后期利用)
        noise_std = max(0.05, AlgoConfig.NOISE_STD * (1 - i_episode / MAX_EPISODES))

        for t in range(MAX_STEPS):
            # 1. 选择动作 (带噪声)
            actions = maddpg.select_actions(obs_n, noise_std=noise_std)

            # 2. 执行动作
            next_obs_n, rewards_n, terminated, truncated, _ = env.step(actions)
            next_global_obs = env.get_global_state()

            done = any(terminated) or any(truncated)

            # 3. 存储经验 (所有智能体共享一个 buffer 或者 存入各自 buffer，这里用统一 buffer)
            # 需要把 list 转为 numpy array
            maddpg.memory.push(
                np.array(obs_n),
                global_obs,
                np.array(actions),
                np.array(rewards_n),
                np.array(next_obs_n),
                next_global_obs,
                np.array([done] * Config.N_UAV)  # 简化 done 信号
            )

            # 4. 更新模型
            # 并不是每步都更新，可以在 buffer 存够一定量后开始
            #c_loss, a_loss = maddpg.update()
            ret = maddpg.update()
            if ret is not None:
                c_loss, a_loss = ret
            else:
                c_loss, a_loss = None, None

            # 状态转移
            obs_n = next_obs_n
            global_obs = next_global_obs
            episode_reward += np.sum(rewards_n)

        # 记录本回合结果
        scores.append(episode_reward)
        current_coverage = np.sum(env.global_map_cover) / (Config.GRID_ROWS * Config.GRID_COLS)
        coverages.append(current_coverage)

        # 打印进度
        print(f"Episode {i_episode}/{MAX_EPISODES} | "
              f"Reward: {episode_reward:.2f} | "
              f"Cov: {current_coverage:.2%} | "
              f"Noise: {noise_std:.3f}")

        # ===========================
        # 评估与保存
        # ===========================
        if i_episode % EVAL_INTERVAL == 0:
            eval_reward, eval_cov = evaluate(env, maddpg)
            print(f"\n--- Evaluation @ Ep {i_episode} ---")
            print(f"Avg Reward: {eval_reward:.2f}")
            print(f"Avg Coverage: {eval_cov:.2%}")
            print(f"----------------------------\n")

            # 保存最佳模型 (以覆盖率为主要指标，符合论文目标)
            if eval_cov > best_coverage:
                best_coverage = eval_cov
                maddpg.save_models(SAVE_DIR)
                print(f"*** New Best Model Saved (Cov: {best_coverage:.2%}) ***")

    # ===========================
    # 4. 结果可视化
    # ===========================
    plt.figure(figsize=(12, 5))

    # 奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)

    # 覆盖率曲线
    plt.subplot(1, 2, 2)
    plt.plot(coverages, color='orange')
    plt.title("Coverage Rate per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Coverage Rate")
    plt.grid(True)

    plt.savefig(f"{RESULT_DIR}/training_result.png")
    plt.show()

    print("Training Finished. Results saved.")