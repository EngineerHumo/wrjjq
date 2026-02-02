import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy


# ===========================
# 1. 基础配置与工具
# ===========================
class AlgoConfig:
    """算法超参数配置"""
    LR_ACTOR = 1e-4  # Actor 学习率
    LR_CRITIC = 1e-3  # Critic 学习率
    GAMMA = 0.95  # 折扣因子 [cite: 141]
    TAU = 0.01  # 软更新系数
    BATCH_SIZE = 128  # 批次大小
    BUFFER_SIZE = 100000  # 经验池大小 [cite: 156]
    HIDDEN_DIM = 64  # 隐藏层维度
    NOISE_STD = 0.1  # 探索噪声标准差


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================
# 2. 神经网络定义
# ===========================
class Actor(nn.Module):
    """
    策略网络 (Policy Network)
    输入: 局部观测 (Local Observation)
    输出: 连续动作 (Action: [acc, omega])
    """

    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, AlgoConfig.HIDDEN_DIM)
        self.fc2 = nn.Linear(AlgoConfig.HIDDEN_DIM, AlgoConfig.HIDDEN_DIM)
        self.fc3 = nn.Linear(AlgoConfig.HIDDEN_DIM, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 使用 Tanh 将输出限制在 [-1, 1] 之间 [cite: 304]
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """
    价值网络 (Value Network) - 集中式 Critic
    输入: 全局状态 (Global State) + 所有智能体的动作 (All Actions)
    输出: Q值
    """

    def __init__(self, global_obs_dim, total_act_dim):
        super(Critic, self).__init__()
        # Critic 输入维度 = 全局状态维度 + 所有无人机动作拼接的维度
        self.fc1 = nn.Linear(global_obs_dim + total_act_dim, AlgoConfig.HIDDEN_DIM * 2)
        self.fc2 = nn.Linear(AlgoConfig.HIDDEN_DIM * 2, AlgoConfig.HIDDEN_DIM * 2)
        self.fc3 = nn.Linear(AlgoConfig.HIDDEN_DIM * 2, 1)

    def forward(self, state, actions):
        # state: (batch, global_obs_dim)
        # actions: (batch, total_act_dim) -> 拼接所有 agent 动作
        x = torch.cat([state, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ===========================
# 3. 经验回放池
# ===========================
class ReplayBuffer:
    """
    经验回放机制 [cite: 156]
    存储 (S, A, R, S', Done)
    """

    def __init__(self, capacity, n_agents, obs_dim, act_dim, global_obs_dim):
        self.capacity = capacity
        self.n_agents = n_agents
        self.pointer = 0
        self.size = 0

        # 初始化缓冲区容器
        self.obs_buf = np.zeros((capacity, n_agents, obs_dim))
        self.global_obs_buf = np.zeros((capacity, global_obs_dim))
        self.act_buf = np.zeros((capacity, n_agents, act_dim))
        self.rew_buf = np.zeros((capacity, n_agents))
        self.next_obs_buf = np.zeros((capacity, n_agents, obs_dim))
        self.next_global_obs_buf = np.zeros((capacity, global_obs_dim))
        self.done_buf = np.zeros((capacity, n_agents))

    def push(self, obs, global_obs, act, rew, next_obs, next_global_obs, done):
        idx = self.pointer
        self.obs_buf[idx] = obs
        self.global_obs_buf[idx] = global_obs
        self.act_buf[idx] = act
        self.rew_buf[idx] = rew
        self.next_obs_buf[idx] = next_obs
        self.next_global_obs_buf[idx] = next_global_obs
        self.done_buf[idx] = done

        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)

        # 转换为 Tensor
        return (
            torch.FloatTensor(self.obs_buf[idxs]).to(device),
            torch.FloatTensor(self.global_obs_buf[idxs]).to(device),
            torch.FloatTensor(self.act_buf[idxs]).to(device),
            torch.FloatTensor(self.rew_buf[idxs]).to(device),
            torch.FloatTensor(self.next_obs_buf[idxs]).to(device),
            torch.FloatTensor(self.next_global_obs_buf[idxs]).to(device),
            torch.FloatTensor(self.done_buf[idxs]).to(device)
        )


# ===========================
# 4. 单个智能体定义
# ===========================
class Agent:
    def __init__(self, obs_dim, act_dim, global_obs_dim, total_act_dim, agent_idx):
        self.agent_idx = agent_idx

        # 初始化 Actor 和 Critic
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(global_obs_dim, total_act_dim).to(device)

        # 初始化目标网络
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=AlgoConfig.LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=AlgoConfig.LR_CRITIC)

    def get_action(self, obs, noise_std=0.0):
        """获取动作，训练时加入噪声用于探索"""
        obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()[0]

        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        return action

    def soft_update(self):
        """目标网络软更新: theta' = tau * theta + (1-tau) * theta'"""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - AlgoConfig.TAU) + param.data * AlgoConfig.TAU)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - AlgoConfig.TAU) + param.data * AlgoConfig.TAU)


# ===========================
# 5. MADDPG 控制器
# ===========================
class MADDPG:
    """
    多智能体控制器，管理所有 Agent 的训练过程
    """

    def __init__(self, n_agents, obs_dim, act_dim, global_obs_dim):
        self.n_agents = n_agents
        self.act_dim = act_dim
        total_act_dim = n_agents * act_dim

        # 创建 Agent 列表
        self.agents = [Agent(obs_dim, act_dim, global_obs_dim, total_act_dim, i) for i in range(n_agents)]

        # 经验回放池
        self.memory = ReplayBuffer(AlgoConfig.BUFFER_SIZE, n_agents, obs_dim, act_dim, global_obs_dim)

    def select_actions(self, obs_n, noise_std=0.0):
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.get_action(obs_n[i], noise_std)
            actions.append(action)
        return actions

    def update(self):
        if self.memory.size < AlgoConfig.BATCH_SIZE:
            return None

        # 采样一批数据
        # obs_batch: (batch, n_agents, obs_dim)
        # act_batch: (batch, n_agents, act_dim)
        obs_batch, global_obs_batch, act_batch, rew_batch, next_obs_batch, next_global_obs_batch, done_batch = self.memory.sample(
            AlgoConfig.BATCH_SIZE)

        # ----------------------------
        # 1. 更新 Critic (集中式训练)
        # ----------------------------
        # 准备下一步的目标动作 (Target Actions)
        # 需要把 next_obs 输入到每个 Agent 的 target_actor 中
        next_actions = []
        with torch.no_grad():
            for i, agent in enumerate(self.agents):
                # 获取第 i 个 agent 的 target action
                n_o = next_obs_batch[:, i, :]
                n_a = agent.target_actor(n_o)
                next_actions.append(n_a)
            # 拼接所有 agent 的 target actions: (batch, total_act_dim)
            next_actions_cat = torch.cat(next_actions, dim=1)

        # 对每个 Agent 进行 Critic 更新
        critic_losses = []
        for i, agent in enumerate(self.agents):
            # 计算目标 Q 值: y = r + gamma * Q'(s', a')
            with torch.no_grad():
                target_q = agent.target_critic(next_global_obs_batch, next_actions_cat)
                y = rew_batch[:, i].unsqueeze(1) + AlgoConfig.GAMMA * target_q * (1 - done_batch[:, i].unsqueeze(1))

            # 计算当前 Q 值: Q(s, a)
            # 需要拼接当前所有 agent 的动作
            # act_batch shape: (batch, n, act_dim) -> reshape to (batch, total_act_dim)
            current_actions_cat = act_batch.view(AlgoConfig.BATCH_SIZE, -1)
            current_q = agent.critic(global_obs_batch, current_actions_cat)

            # Critic 损失函数 (MSE)
            critic_loss = F.mse_loss(current_q, y)

            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()
            critic_losses.append(critic_loss.item())

        # ----------------------------
        # 2. 更新 Actor (根据 Critic 指导)
        # ----------------------------
        actor_losses = []
        for i, agent in enumerate(self.agents):
            # 为了计算 Actor 梯度，我们需要重新计算当前动作，但这步必须要保留梯度 (requires_grad=True)
            # 其他 agent 的动作不需要梯度，可以使用 batch 中的数据，或者为了更准确也可以用当前策略重新生成
            # 这里采用标准 MADDPG 做法：当前 agent 用最新策略，其他 agent 用 buffer 中的动作（或当前策略的 detach）

            curr_actions_list = []
            for j, other_agent in enumerate(self.agents):
                o = obs_batch[:, j, :]
                if i == j:
                    # 当前需要更新的 agent，动作带有梯度
                    curr_actions_list.append(agent.actor(o))
                else:
                    # 其他 agent 的动作，detach 掉，视为环境的一部分
                    # 或者使用 act_batch 中的动作 (Approximation)
                    # 论文中通常建议使用 act_batch，或者使用 other_agent.actor(o).detach()
                    # 这里使用 other_agent.actor(o).detach() 以保证策略的一致性
                    curr_actions_list.append(other_agent.actor(o).detach())

            curr_actions_cat = torch.cat(curr_actions_list, dim=1)

            # Actor Loss: 最大化 Q 值 -> 最小化 -Q
            actor_loss = -agent.critic(global_obs_batch, curr_actions_cat).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()
            actor_losses.append(actor_loss.item())

        # ----------------------------
        # 3. 软更新目标网络
        # ----------------------------
        for agent in self.agents:
            agent.soft_update()

        return np.mean(critic_losses), np.mean(actor_losses)

    def save_models(self, path):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), f"{path}/agent_{i}_actor.pth")
            torch.save(agent.critic.state_dict(), f"{path}/agent_{i}_critic.pth")

    def load_models(self, path):
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(torch.load(f"{path}/agent_{i}_actor.pth"))
            agent.critic.load_state_dict(torch.load(f"{path}/agent_{i}_critic.pth"))