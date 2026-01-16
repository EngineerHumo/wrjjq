import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 检查是否有GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. 多智能体经验回放池 (Global Buffer) ---
class MultiAgentReplayBuffer:
    def __init__(self, max_size, num_agents, state_dims, action_dims):
        """
        :param max_size: 经验池最大容量
        :param num_agents: 智能体数量
        :param state_dims: list, 每个智能体的状态维度 [dim1, dim2, ...]
        :param action_dims: list, 每个智能体的动作维度 [dim1, dim2, ...]
        """
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.num_agents = num_agents

        # 初始化存储容器：self.obs_n[i] 存储第 i 个智能体的观测历史
        self.obs_n = [np.zeros((self.max_size, state_dims[i])) for i in range(num_agents)]
        self.act_n = [np.zeros((self.max_size, action_dims[i])) for i in range(num_agents)]
        self.rew_n = np.zeros((self.max_size, num_agents))  # 奖励矩阵 [size, num_agents]
        self.next_obs_n = [np.zeros((self.max_size, state_dims[i])) for i in range(num_agents)]
        self.done_n = np.zeros((self.max_size, num_agents))

    def add(self, obs_list, act_list, rew_list, next_obs_list, done_list):
        """添加一条多智能体联合样本"""
        idx = self.ptr
        for i in range(self.num_agents):
            self.obs_n[i][idx] = obs_list[i]
            self.act_n[i][idx] = act_list[i]
            self.next_obs_n[i][idx] = next_obs_list[i]

        self.rew_n[idx] = np.array(rew_list)
        self.done_n[idx] = np.array(done_list)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """随机采样"""
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch_obs_n = [torch.FloatTensor(self.obs_n[i][idxs]).to(device) for i in range(self.num_agents)]
        batch_act_n = [torch.FloatTensor(self.act_n[i][idxs]).to(device) for i in range(self.num_agents)]
        batch_rew_n = torch.FloatTensor(self.rew_n[idxs]).to(device)
        batch_next_obs_n = [torch.FloatTensor(self.next_obs_n[i][idxs]).to(device) for i in range(self.num_agents)]
        batch_done_n = torch.FloatTensor(self.done_n[idxs]).to(device)

        return batch_obs_n, batch_act_n, batch_rew_n, batch_next_obs_n, batch_done_n


# --- 2. 神经网络定义 ---
class Actor(nn.Module):
    """演员网络：只输入自己的 state"""
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    """评论家网络 (Global Critic)：输入所有人的 states 和 actions"""
    def __init__(self, total_state_dim, total_action_dim):
        super(Critic, self).__init__()
        # 输入维度 = sum(all_states) + sum(all_actions)
        self.l1 = nn.Linear(total_state_dim + total_action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state_cat, action_cat):
        x = torch.cat([state_cat, action_cat], dim=1)
        q = F.relu(self.l1(x))
        q = F.relu(self.l2(q))
        return self.l3(q)


# --- 3. MADDPG 核心结构 ---
class MADDPG:
    """每个智能体持有的算法核心（网络 + 优化器）"""
    def __init__(self, state_dim, action_dim, max_action, total_state_dim, total_action_dim,
                 actor_lr=1e-4, critic_lr=1e-3, tau=0.005, gamma=0.99):
        # Local Actor
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Global Critic
        self.critic = Critic(total_state_dim, total_action_dim).to(device)
        self.critic_target = Critic(total_state_dim, total_action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.discount = gamma
        self.tau = tau  # 软更新系数

    def select_action(self, state):
        state = np.array(state, dtype=np.float32)
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action_tensor = self.actor(state_tensor).detach().cpu()
        return action_tensor.numpy().flatten()


# --- 4. 集中式训练函数 (Centralized Training) ---
def train_centralized(agents, buffer, batch_size=64, gamma=0.99):
    """
    对所有智能体执行一步 MADDPG 更新
    """
    if buffer.size < batch_size:
        return

    # 1. 从全局 Buffer 采样
    # obs_n: list of tensors, shape [batch, state_dim]
    obs_n, act_n, rew_n, next_obs_n, done_n = buffer.sample(batch_size)

    # 2. 拼接全局向量 (for Critic)
    obs_cat = torch.cat(obs_n, dim=1)  # [batch, total_state_dim]
    act_cat = torch.cat(act_n, dim=1)  # [batch, total_action_dim]
    next_obs_cat = torch.cat(next_obs_n, dim=1)

    # 3. 遍历每个智能体进行更新
    for i, agent in enumerate(agents):
        brain = agent.brain

        # --- Update Critic ---
        with torch.no_grad():
            # 计算所有智能体在下一时刻的目标动作
            target_actions = []
            for j, other_agent in enumerate(agents):
                # 注意：使用每个 Agent 自己的 Target Actor
                target_actions.append(other_agent.brain.actor_target(next_obs_n[j]))
            target_act_cat = torch.cat(target_actions, dim=1)

            # 计算 Target Q (使用当前 Agent 的 Target Critic)
            target_Q = brain.critic_target(next_obs_cat, target_act_cat)
            target_Q = rew_n[:, i].view(-1, 1) + (1 - done_n[:, i].view(-1, 1)) * gamma * target_Q

        # Current Q
        current_Q = brain.critic(obs_cat, act_cat)

        # Loss & Step
        critic_loss = F.mse_loss(current_Q, target_Q)
        brain.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.critic.parameters(), 0.5)  # 梯度裁剪
        brain.critic_optimizer.step()

        # --- Update Actor ---
        # 计算当前动作组合 (用于评估 Actor Performance)
        # 技巧：只有当前 Agent i 的动作需要梯度，其他 Agent 的动作视为环境常量
        curr_actions = []
        for j, other_agent in enumerate(agents):
            action_j = brain.actor(obs_n[j])  # 使用当前网络计算
            if i != j:
                action_j = action_j.detach() # 其他智能体的动作视为环境一部分，不传梯度
            curr_actions.append(action_j)

        curr_act_cat = torch.cat(curr_actions, dim=1)

        # Actor Loss: 最大化 Critic 的评分
        actor_loss = -brain.critic(obs_cat, curr_act_cat).mean()

        brain.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.actor.parameters(), 0.5)  # 梯度裁剪
        brain.actor_optimizer.step()

        # --- Soft Update ---
        for param, target_param in zip(brain.critic.parameters(), brain.critic_target.parameters()):
            target_param.data.copy_(brain.tau * param.data + (1 - brain.tau) * target_param.data)

        for param, target_param in zip(brain.actor.parameters(), brain.actor_target.parameters()):
            target_param.data.copy_(brain.tau * param.data + (1 - brain.tau) * target_param.data)


# --- 5. UAV Agent 类 ---
class UAVAgent:
    def __init__(self, uav_id, initial_pos, initial_v, initial_phi, step,
                 total_state_dim=None, total_action_dim=None,
                 actor_lr=1e-4, critic_lr=1e-3, tau=0.005, gamma=0.99):
        # 物理属性
        self.step = step
        self.ID = uav_id
        self.pos = np.array(initial_pos, dtype=float)
        self.v = float(initial_v)
        self.phi = np.radians(initial_phi)
        # self.a = float(initial_a)
        # self.w = np.radians(initial_w)

        # 约束
        self.velocity_range = [28.0, 70.0]
        self.acc_range = [-8, 8]
        self.w_range = [-np.radians(5), np.radians(5)]
        self.comm_range = 60.0
        self.detect_radius = 250.0  # 无人机探测半径
        self.detecct_p = 0.9

        # 状态与动作维度
        self.state_dim = 15
        self.action_dim = 2
        self.max_action_tensor = torch.tensor([1.0, 1.0]).to(device)

        # 初始化 MADDPG
        if total_state_dim is None: total_state_dim = self.state_dim
        if total_action_dim is None: total_action_dim = self.action_dim

        self.brain = MADDPG(self.state_dim, self.action_dim, self.max_action_tensor,
                                 total_state_dim, total_action_dim,
                            actor_lr, critic_lr, tau, gamma)

        self.assigned_task_coords = None
        # self.last_obs = None  # 暂存上一步观测
        # self.last_act = None  # 暂存上一步动作
        # self.prev_dist_to_task = None
        # self.prev_dist = 0.0

    def get_observation(self, map_size, obstacles_map, all_uavs, target_predictors):
        """
        获取观测状态
        """
        M, N = map_size
        x, y = self.pos

        # --- 1. 基础自身信息 ---
        # 如果没有任务，目标点设为自己当前位置 (距离为0)，避免飞向地图中心
        # target_x, target_y = self.assigned_task_coords if self.assigned_task_coords is not None else (x, y)
        # dx = target_x - x
        # dy = target_y - y
        # dist = np.sqrt(dx ** 2 + dy ** 2)
        # target_angle = np.arctan2(dy, dx)
        # rel_angle = target_angle - self.phi

        # --- 2. 处理通信网络 (邻居) ---
        neighbors = []
        for other in all_uavs:
            if other is self: continue
            d_vec = other.pos - self.pos
            if np.linalg.norm(d_vec) <= self.comm_range:
                neighbors.append(other)

        num_neighbors = len(neighbors)
        has_neighbor = 0.0
        n_dx, n_dy = 0.0, 0.0

        if num_neighbors > 0:
            closest_uav = min(neighbors, key=lambda u: np.linalg.norm(u.pos - self.pos))
            n_dx = closest_uav.pos[0] - x
            n_dy = closest_uav.pos[1] - y
            has_neighbor = 1.0

        # --- 3. 处理协同障碍物 ---
        def scan_obstacles(agent_pos):
            detected = []
            if obstacles_map is None: return []
            r_int = int(self.detect_radius)
            c_center = int(agent_pos[0])
            r_center = int((M - 1) - agent_pos[1])
            r_min = max(0, r_center - r_int)
            r_max = min(M, r_center + r_int + 1)
            c_min = max(0, c_center - r_int)
            c_max = min(N, c_center + r_int + 1)
            local_map = obstacles_map[r_min:r_max, c_min:c_max]
            obs_indices = np.argwhere(local_map == -1)
            for idx in obs_indices:
                row_global = r_min + idx[0]
                col_global = c_min + idx[1]
                obs_x = col_global + 0.5
                obs_y = (M - 1) - row_global + 0.5
                if (obs_x - agent_pos[0]) ** 2 + (obs_y - agent_pos[1]) ** 2 <= self.detect_radius ** 2:
                    detected.append(np.array([obs_x, obs_y]))
            return detected

        all_known = scan_obstacles(self.pos)
        for n_uav in neighbors:
            all_known.extend(scan_obstacles(n_uav.pos))

        has_obstacle = 0.0
        o_dx, o_dy = 0.0, 0.0

        if len(all_known) > 0:
            dists = [np.linalg.norm(op - self.pos) for op in all_known]
            min_idx = np.argmin(dists)
            nearest_obs = all_known[min_idx]
            o_dx = nearest_obs[0] - x
            o_dy = nearest_obs[1] - y
            has_obstacle = 1.0

        # --- 4. 组装 Observation ---
        obs = np.array([
            x / N, y / M,
            (self.v-self.velocity_range[0]) / (self.velocity_range[1]-self.velocity_range[0]),  # 归一化
            self.phi / np.pi,
            num_neighbors / 5.0,
            has_neighbor,
            n_dx / self.comm_range,
            n_dy / self.comm_range,
            has_obstacle,
            o_dx / self.detect_radius,
            o_dy / self.detect_radius
        ])

        # 不确定性扇区感知
        uncertainty_sectors = np.zeros(4)  # [前, 左, 后, 右]

        sense_radius = self.detect_radius  # 250.0

        for predictor in target_predictors:
            # 1. 计算相对距离
            d_vec = predictor.particles[:, 0:2] - self.pos
            dists = np.linalg.norm(d_vec, axis=1)

            # 2. 只统计探测半径内的粒子
            mask = dists < sense_radius
            if not np.any(mask): continue

            # 3. 计算相对角度
            rel_angles = np.arctan2(d_vec[mask, 1], d_vec[mask, 0]) - self.phi
            rel_angles = (rel_angles + np.pi) % (2 * np.pi) - np.pi

            weights = predictor.weights[mask]

            # 4. 统计扇区
            # 前 (-45 ~ 45)
            uncertainty_sectors[0] += np.sum(weights[(rel_angles >= -np.pi / 4) & (rel_angles < np.pi / 4)])
            # 左 (45 ~ 135)
            uncertainty_sectors[1] += np.sum(weights[(rel_angles >= np.pi / 4) & (rel_angles < 3 * np.pi / 4)])
            # 后 (135 ~ -135)
            uncertainty_sectors[2] += np.sum(weights[(rel_angles >= 3 * np.pi / 4) | (rel_angles < -3 * np.pi / 4)])
            # 右 (-135 ~ -45)
            uncertainty_sectors[3] += np.sum(weights[(rel_angles >= -3 * np.pi / 4) & (rel_angles < -np.pi / 4)])

        # 归一化：为了让数值匹配神经网络输入范围，稍微缩放一下
        uncertainty_sectors = np.clip(uncertainty_sectors * 5.0, 0, 5.0)

        # 拼接
        final_obs = np.concatenate((obs, uncertainty_sectors))

        return final_obs

    def state_update(self, action, map_size, obstacles_map):
        """
        物理运动学更新 + 边界/障碍物处理
        :param action: 归一化动作 [-1, 1] -> [v_acc, w_acc] 或直接映射
        :param map_size: (M, N)
        :param obstacles_map: 障碍物矩阵
        """
        M, N = map_size

        # 1. 解析动作 (Action -> Physics)
        # 假设 action[0] 控制加速度, action[1] 控制角速度
        # 映射到物理范围: action [-1, 1] -> acc_range, w_range
        acc = action[0] * self.acc_range[1]
        w = action[1] * self.w_range[1]

        # 2. 更新速度与航向
        self.v += acc * self.step
        self.phi += w * self.step

        # 速度限幅
        self.v = np.clip(self.v, self.velocity_range[0], self.velocity_range[1])
        # 角度归一化 (-pi, pi)
        self.phi = (self.phi + np.pi) % (2 * np.pi) - np.pi

        # 3. 试探性更新位置
        dx = self.v * np.cos(self.phi) * self.step
        dy = self.v * np.sin(self.phi) * self.step

        next_x = self.pos[0] + dx
        next_y = self.pos[1] + dy

        # 4. 边界处理 (反弹)
        # 如果撞墙，不仅位置限制，角度也要反射
        if next_x < 0 or next_x >= N:
            self.phi = np.pi - self.phi  # 水平镜像反弹
            next_x = np.clip(next_x, 0, N - 0.1)

        if next_y < 0 or next_y >= M:
            self.phi = -self.phi  # 垂直镜像反弹
            next_y = np.clip(next_y, 0, M - 0.1)

        # 5. 障碍物处理
        if obstacles_map is not None:
            # 坐标转网格索引 (注意 y 轴翻转)
            c = int(np.clip(next_x, 0, N - 1))
            r = int(np.clip(M - 1 - next_y, 0, M - 1))

            if obstacles_map[r, c] == -1:
                # 撞到障碍物：位置回退到上一步，并掉头
                next_x = self.pos[0]
                next_y = self.pos[1]
                self.phi += np.pi  # 简单掉头处理

        # 确认更新
        self.pos = np.array([next_x, next_y])

    def calculate_bid(self, task):
        # 简单的拍卖计算逻辑
        task_pos = np.array(task['coords'])
        priority = task['priority']
        distance = np.linalg.norm(self.pos - task_pos)
        time_cost = distance / self.v
        bid = 0.5 * priority / 5 - 0.5 * (time_cost / 1200.0)  # 归一化处理
        return bid

    def calculate_reward(self, prev_entropy, curr_entropy, is_detected, action, map_size, obstacles_map, all_uavs):
        """
        最小化熵 + 探测保持 + 安全约束
        """
        # 1. 信息增益 (熵减)
        # 放大系数 10.0 是经验值（调节RL收敛速度），不影响物理参数
        r_info = (prev_entropy - curr_entropy) * 10.0

        # 2. 探测奖励 (Detection)
        # 探测到目标给予高额奖励
        r_detect = 10.0 if is_detected else 0.0

        # 3. 动作与安全 (保持你原有的参数不变)
        r_action = -0.05 * np.sum(action ** 2)

        r_collision = 0.0
        x, y = self.pos
        M, N = map_size

        # 边界约束 (你原来的参数)
        margin = 20.0
        if x < margin:
            r_collision -= (margin - x) / margin * 2.0
        elif x > N - margin:
            r_collision -= (x - (N - margin)) / margin * 2.0
        if y < margin:
            r_collision -= (margin - y) / margin * 2.0
        elif y > M - margin:
            r_collision -= (y - (M - margin)) / margin * 2.0
        if x < 0 or x >= N or y < 0 or y >= M: r_collision -= 10.0

        # 障碍物约束
        if obstacles_map is not None:
            c, r = int(np.clip(x, 0, N - 1)), int(np.clip(M - 1 - y, 0, M - 1))
            if obstacles_map[r, c] == -1: r_collision -= 10.0

        # 机间避碰 (你原来的参数 safe_dist=15.0)
        safe_dist = 15.0
        for other in all_uavs:
            if other is self: continue
            d = np.linalg.norm(self.pos - other.pos)
            if d < safe_dist:
                r_collision -= 5.0 * (1.0 - d / safe_dist)

        total_reward = r_info + r_detect + r_action + r_collision
        return total_reward


# 奖励归一化类部分
class RunningMeanStd:
    # 动态计算均值和方差的辅助类
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = self.update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count
        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        return new_mean, new_var, tot_count

class RewardScaler:
    def __init__(self, shape, gamma=0.99, epsilon=1e-8):
        self.shape = shape
        self.gamma = gamma
        self.epsilon = epsilon
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x):
        # x: 单个奖励值或奖励数组
        # 更新统计量
        self.running_ms.update(x)
        # 归一化：这里只除以标准差 (Scaling)，不减均值 (Centering)
        # 这样可以保留奖励的正负符号（例如撞墙仍然是负的），只缩放幅度
        x_norm = x / np.sqrt(self.running_ms.var + self.epsilon)
        return x_norm


    # def calculate_reward(self, obs, action, map_size, obstacles_map, all_uavs, is_detected, particle_std=None):
    #     """
    #     核心目标：
    #     1. 降低全局预测不确定性 (particle_std)
    #     2. 保持对目标的持续探测 (is_detected)
    #     3. 安全飞行 (不出界、不碰撞)
    #     """
    #     M, N = map_size
    #
    #     # --- 1. 全局不确定性奖励 (Global Uncertainty Reward) ---
    #     # 这是驱动无人机"去哪里"的核心动力
    #     r_uncertainty = 0.0
    #     if particle_std is not None:
    #         # 归一化 std: 假设如果不去观测，粒子最大扩散到 200m (经验值)
    #         # 逻辑：std=0 -> reward=+2.0; std=100 -> reward=0.0; std=200 -> reward=-2.0
    #         norm_std = np.clip(particle_std, 0, 200.0) / 200.0
    #         r_uncertainty = 2.0 * (1.0 - norm_std * 2.0)
    #
    #         # 动态加权：如果还没看到目标，"降低不确定性"的权重翻倍，迫使它去搜
    #         if not is_detected:
    #             r_uncertainty *= 1.5
    #
    #     # --- 2. 探测奖励 (Detection Reward) ---
    #     # 只要有一架无人机看到了目标，这就是一个高额的正反馈
    #     r_detect = 0.0
    #     if is_detected:
    #         r_detect = 2.0  # 锁定目标的持续奖励
    #
    #     # --- 3. 动作平滑与能耗 ---
    #     r_action = -0.05 * np.sum(action ** 2)
    #
    #     # --- 4. 安全约束 (Safety Constraints) ---
    #     r_collision = 0.0
    #     x, y = self.pos
    #
    #     # 4.1 强力边界约束 (防止因为没有任务点引导而乱飞出界)
    #     # 预警区：离边界 20m 开始扣分
    #     margin = 20.0
    #     if x < margin: r_collision -= (margin - x) / margin * 1.0
    #     if x > N - margin: r_collision -= (x - (N - margin)) / margin * 1.0
    #     if y < margin: r_collision -= (margin - y) / margin * 1.0
    #     if y > M - margin: r_collision -= (y - (M - margin)) / margin * 1.0
    #
    #     # 越界极刑
    #     if x < 0 or x >= N or y < 0 or y >= M:
    #         r_collision -= 10.0
    #
    #     # 4.2 障碍物约束
    #     if obstacles_map is not None:
    #         c, r = int(np.clip(x, 0, N - 1)), int(np.clip(M - 1 - y, 0, M - 1))
    #         if obstacles_map[r, c] == -1:
    #             r_collision -= 10.0
    #
    #     # 4.3 机间避碰
    #     safe_dist = 15.0
    #     for other in all_uavs:
    #         if other is self: continue
    #         d = np.linalg.norm(self.pos - other.pos)
    #         if d < safe_dist:
    #             r_collision -= 5.0 * (1.0 - d / safe_dist)
    #
    #     # === 总分汇总 ===
    #     total_reward = r_uncertainty + r_detect + r_action + r_collision
    #
    #     return total_reward

    # def calculate_reward(self, obs, action, map_size, obstacles_map, all_uavs, is_detected, particle_std=None):
    #     """
    #     增加了降低不确定性奖励
    #     :param is_detected: bool, 本时刻是否探测到了任意真实目标
    #     :param particle_std: float, 当前目标预测粒子的平均位置标准差
    #     """
    #     M, N = map_size
    #     max_dist = np.sqrt(M ** 2 + N ** 2)
    #
    #     # 从观测中恢复真实距离
    #     dist_norm = obs[6]
    #     real_dist = dist_norm * max_dist
    #
    #     # 1. 距离引导奖励
    #     r_dist = -dist_norm
    #
    #     # 2. 势能奖励
    #     if self.prev_dist == 0.0: self.prev_dist = real_dist
    #     progress = (self.prev_dist - real_dist) / max_dist
    #     r_potential = progress * 10.0
    #     self.prev_dist = real_dist
    #
    #     # 3. 到达奖励
    #     r_reach = 0.0
    #     if self.assigned_task_coords is not None:
    #         if real_dist < 30.0:
    #             r_reach = 1.0
    #
    #     # 4. 真实探测奖励
    #     r_detect = 0.0
    #     if is_detected:
    #         r_detect = 5.0
    #
    #     # 5. 区域不确定性奖励
    #     # 鼓励无人机主动降低预测的不确定性
    #     r_uncertainty = 0.0
    #     if particle_std is not None:
    #         # 归一化 std: 假设如果不去观测，粒子最大扩散到 200m (经验值)
    #         # 我们希望 std 越小越好
    #         norm_std = np.clip(particle_std, 0, 200.0) / 200.0
    #
    #         # 奖励设计：
    #         # std 很小 (0) -> +1.0 奖励
    #         # std 很大 (200) -> -1.0 惩罚
    #         r_uncertainty = 1.0 - (norm_std * 2.0)
    #
    #         # 动态权重策略：
    #         # 如果还没看到目标 (is_detected=False)，此时"降低不确定性"是第一要务
    #         # 因此放大这个奖励的权重
    #         if not is_detected:
    #             r_uncertainty *= 2.0
    #
    #             # 6. 动作平滑
    #     r_action = -0.05 * np.sum(action ** 2)
    #
    #     # === [强化] 7. 边界与避碰约束 (Safety Constraints) ===
    #     r_collision = 0.0
    #     x, y = self.pos
    #
    #     # 7.1 强化地图边界约束
    #     # 预警区：离边界 20m 以内开始扣分 (线性增加)
    #     margin = 20.0
    #     if x < margin:
    #         r_collision -= (margin - x) / margin * 2.0
    #     elif x > N - margin:
    #         r_collision -= (x - (N - margin)) / margin * 2.0
    #
    #     if y < margin:
    #         r_collision -= (margin - y) / margin * 2.0
    #     elif y > M - margin:
    #         r_collision -= (y - (M - margin)) / margin * 2.0
    #
    #     # 严禁出界：一旦出界给予巨额惩罚
    #     if x < 0 or x >= N or y < 0 or y >= M:
    #         r_collision -= 10.0  # 原来是 -5.0，建议加大
    #
    #     # 7.2 障碍物约束
    #     if obstacles_map is not None:
    #         # 增加越界保护
    #         c, r = int(np.clip(x, 0, N - 1)), int(np.clip(M - 1 - y, 0, M - 1))
    #         if obstacles_map[r, c] == -1:
    #             r_collision -= 10.0  # 撞障碍物重罚
    #
    #     # 7.3 机间避碰
    #     safe_dist = 15.0
    #     for other in all_uavs:
    #         if other is self: continue
    #         d = np.linalg.norm(self.pos - other.pos)
    #         if d < safe_dist:
    #             r_collision -= 5.0 * (1.0 - d / safe_dist)  # 加大避碰权重
    #
    #     # === 总分汇总 ===
    #     total_reward = r_dist + r_potential + r_reach + r_detect + r_action + r_collision + r_uncertainty
    #     return total_reward

    # def calculate_reward(self, obs, action, map_size, obstacles_map, all_uavs, is_detected, communication_range=60.0):
    #     """
    #     :param is_detected: bool, 本时刻是否探测到了任意真实目标
    #     """
    #     M, N = map_size
    #     max_dist = np.sqrt(M ** 2 + N ** 2)
    #
    #     # 从观测中恢复真实距离
    #     dist_norm = obs[6]
    #     real_dist = dist_norm * max_dist  # 映射
    #
    #     # --- 1. 距离引导奖励 (归一化到 -1 ~ 0) ---
    #     # 目的：引导无人机靠近分配的任务点
    #     r_dist = -dist_norm
    #
    #     # --- 2. 势能奖励 (归一化到 -0.5 ~ 0.5) ---
    #     # 目的：鼓励每一步都向目标靠近
    #     # 缩放因子：假设一步最大移动 ~50m, 地图 2000m, 比例 0.025. 放大 20 倍使其显著
    #     if self.prev_dist == 0.0: self.prev_dist = real_dist
    #     progress = (self.prev_dist - real_dist) / max_dist
    #     r_potential = progress * 10.0
    #     self.prev_dist = real_dist
    #
    #     # --- 3. 预测点到达奖励 (小奖励 +1) ---
    #     # 目的：鼓励到达预测位置进行盘旋搜索
    #     r_reach = 0.0
    #     if self.assigned_task_coords is not None:
    #         # 判定到达半径 30m
    #         if real_dist < 30.0:
    #             r_reach = 1.0
    #
    #     # --- 4. 真实探测奖励 (大奖励 +5) ---
    #     r_detect = 0.0
    #     if is_detected:
    #         r_detect = 5.0
    #
    #     # --- 5. 动作平滑 (归一化到 -0.1 ~ 0) ---
    #     # action 取值 [-1, 1], sum(sq) max = 2.
    #     # 0.05 * 2 = 0.1
    #     r_action = -0.05 * np.sum(action ** 2)
    #
    #     # --- 6. 惩罚项 (归一化到 -5 ~ 0) ---
    #     r_collision = 0.0
    #     x, y = self.pos
    #
    #     # 6.1 边界软约束
    #     if x < 10 or x > N - 10 or y < 10 or y > M - 10:
    #         r_collision -= 0.5  # 警告
    #     if x < 0 or x >= N or y < 0 or y >= M:
    #         r_collision -= 5.0  # 严重出界
    #
    #     # 6.2 障碍物
    #     if obstacles_map is not None:
    #         c, r = int(np.clip(x, 0, N - 1)), int(np.clip(M - 1 - y, 0, M - 1))
    #         if obstacles_map[r, c] == -1:
    #             r_collision -= 5.0
    #
    #     # 6.3 机间避碰
    #     safe_dist = 15.0
    #     for other in all_uavs:
    #         if other is self: continue
    #         d = np.linalg.norm(self.pos - other.pos)
    #         if d < safe_dist:
    #             # 越近惩罚越重，最大 -2.0
    #             r_collision -= 2.0 * (1.0 - d / safe_dist)
    #
    #     # --- 总分汇总 ---
    #     total_reward = r_dist + r_potential + r_reach + r_detect + r_action + r_collision
    #
    #     return total_reward

    # def calculate_reward(self, obs, action, map_size, obstacles_map, all_uavs, communication_range=60.0):
    #     M, N = map_size
    #
    #     # 1. 距离奖励 (Task Guidance)
    #     dist_norm = obs[6]
    #     dist_reward = -dist_norm * 10.0
    #
    #     # 2. 到达奖励
    #     real_dist = dist_norm * np.sqrt(M ** 2 + N ** 2)  # 映射
    #     reach_reward = 0.0
    #     if self.assigned_task_coords is not None:
    #         if real_dist < 25.0:
    #             reach_reward = 50.0  # 稍微降低一点，避免数值过大
    #
    #     # 3. 动作平滑
    #     action_penalty = -0.1 * np.sum(action ** 2)
    #
    #     # 4. 边界惩罚 (软约束)
    #     boundary_penalty = 0
    #     x, y = self.pos
    #     if x < 10 or x > N - 10 or y < 10 or y > M - 10:
    #         boundary_penalty = -10.0  # 靠近边界就惩罚
    #     if x < 0 or x >= N or y < 0 or y >= M:
    #         boundary_penalty = -50.0  # 出界重罚
    #
    #     # 5. 障碍物与避碰
    #     obstacle_penalty = 0
    #     if obstacles_map is not None:
    #         c, r = int(x), int(M - 1 - y)
    #         if 0 <= r < M and 0 <= c < N and obstacles_map[r, c] == -1:
    #             obstacle_penalty = -50.0
    #
    #     collision_penalty = 0
    #     safe_distance = 15.0
    #     for other in all_uavs:
    #         if other is self: continue
    #         d = np.linalg.norm(self.pos - other.pos)
    #         if d < safe_distance:
    #             collision_penalty += -20.0 * (1.0 - d / safe_distance)
    #
    #     total_reward = (dist_reward + reach_reward + action_penalty +
    #                     boundary_penalty + obstacle_penalty + collision_penalty)
    #
    #     # 归一化奖励到 [-10, 10] 之间，有助于训练稳定
    #     return total_reward / 5.0