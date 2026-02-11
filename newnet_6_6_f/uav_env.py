import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ===========================
# 1. 系统配置参数 (Configuration)
# ===========================
class Config:
    # 地图与栅格
    MAP_SIZE = 1000.0  # 地图边长 (米) [cite: 263]
    GRID_ROWS = 50  # 栅格行数
    GRID_COLS = 50  # 栅格列数
    GRID_WIDTH = MAP_SIZE / GRID_COLS  # 20m

    # 无人机参数 (UAV)
    N_UAV_RANGE = (3, 6)  # 无人机数量范围
    N_UAV = 6  # 默认无人机数量
    V_MIN = 10.0  # 最小速度 (m/s) [cite: 311]
    V_MAX = 30.0  # 最大速度 (m/s) [cite: 311]
    MAX_ACC = 5.0  # 最大加速度 (m/s^2) [cite: 304]
    MAX_ANG_VEL = np.pi / 4  # 最大角速度 (rad/s) [cite: 314]
    SENSOR_RANGE = 150.0  # 探测半径 (m)
    COMM_RANGE = 300.0  # 通信半径 (m) [cite: 191]
    SAFE_DIST_OBS = 30.0  # 障碍物安全距离 [cite: 317]
    SAFE_DIST_UAV = 20.0  # 无人机避碰距离 [cite: 320]

    # 目标参数 (Target)
    N_TARGET_RANGE = (1, 4)  # 目标数量范围
    N_TARGET = 1  # 默认目标数量
    TARGET_V_RANGE = [5.0, 15.0]  # 目标速度范围
    DT = 1.0  # 仿真步长 (s) [cite: 121]

    # 奖励权重 [cite: 351]
    W_COV = 0.2 / (GRID_ROWS * GRID_COLS)  # 覆盖奖励权重 (归一化)
    W_EXP = 3.0  # 探索奖励权重
    W_UNK = 1.0  # 未知区域探索奖励
    W_COL = -0.5  # 重叠/协同惩罚权重 (论文中虽然叫奖励，通常处理为负项或调整逻辑)
    W_SMOOTH = -0.1  # 平滑飞行权重
    W_OBS = -20.0  # 障碍物惩罚
    W_CRASH = -10.0  # 碰撞惩罚
    W_SHARE = 0.5  # 信息共享奖励
    W_DETECT = 2.0  # 目标检测奖励 (Target Detection Reward)
    W_TRACK = 3.0  # 目标跟踪奖励 (Target Tracking Reward)
    W_SPIN = -0.2  # 原地转圈惩罚
    W_BOUNDARY = -0.5  # 边界惩罚

    # 粒子滤波参数
    N_PARTICLES = 2000
    PF_MOTION_STD = 8.0
    PF_MEAS_STD = 30.0

    # 目标随机运动参数
    TARGET_OMEGA_STD = 0.08  # 转向随机扰动
    TARGET_OMEGA_LIMIT = 0.4  # 最大转向率
    TARGET_SPEED_STD = 0.8  # 速度随机扰动

    # SISC-PF 软约束配置 (新增)
    USE_SOFT_CONSTRAINT = False  # 【安全开关】默认关闭，保证现有训练不受影响
    PF_SOFT_K = 5.0  # 软约束硬度系数 (k)
    PF_SOFT_V_LIMIT = 30.0  # 软约束速度上限 (通常等于 V_MAX)


# ===========================
# 2. 目标与障碍物模型
# ===========================
class Target:
    """
    实现论文 3.2.1 节的目标协同转弯(CT)模型
    """

    def __init__(self, rng):
        self.rng = rng
        self.x = self.rng.uniform(0, Config.MAP_SIZE)
        self.y = self.rng.uniform(0, Config.MAP_SIZE)
        self.v = self.rng.uniform(*Config.TARGET_V_RANGE)
        self.phi = self.rng.uniform(0, 2 * np.pi)  # 航向角
        self.omega = self.rng.uniform(-0.1, 0.1)  # 转弯率

    def step(self):
        # 随机扰动转向与速度，使目标呈现随机性运动
        self.omega += self.rng.normal(0.0, Config.TARGET_OMEGA_STD)
        self.omega = np.clip(self.omega, -Config.TARGET_OMEGA_LIMIT, Config.TARGET_OMEGA_LIMIT)
        self.v += self.rng.normal(0.0, Config.TARGET_SPEED_STD)
        self.v = np.clip(self.v, Config.TARGET_V_RANGE[0], Config.TARGET_V_RANGE[1])

        # 边界反弹逻辑 (简化处理，防止目标跑出地图)
        if self.x < 0 or self.x > Config.MAP_SIZE:
            self.phi = np.pi - self.phi
        if self.y < 0 or self.y > Config.MAP_SIZE:
            self.phi = -self.phi

        # CT 模型状态转移
        # x_k+1 = x_k + (v/w)*sin(w*dt) ...
        # 为避免除零，当 omega 很小时使用匀速直线运动(CV)模型
        if abs(self.omega) < 1e-3:
            self.x += self.v * np.cos(self.phi) * Config.DT
            self.y += self.v * np.sin(self.phi) * Config.DT
        else:
            self.x += (self.v / self.omega) * (np.sin(self.phi + self.omega * Config.DT) - np.sin(self.phi))
            self.y += (self.v / self.omega) * (np.cos(self.phi) - np.cos(self.phi + self.omega * Config.DT))
            self.phi += self.omega * Config.DT

        # 限制在地图内
        self.x = np.clip(self.x, 0, Config.MAP_SIZE)
        self.y = np.clip(self.y, 0, Config.MAP_SIZE)


# ===========================
# 3. 粒子滤波器 (Particle Filter)
# ===========================
class ParticleFilter:
    def __init__(self, n_particles, rng):
        self.n_particles = n_particles
        self.rng = rng
        # 修改点1: 将粒子状态扩展为 4 维 [x, y, v, phi] 以支持软约束计算
        # 即使 USE_SOFT_CONSTRAINT=False，也初始化 4 维，但在逻辑中只更新前 2 维
        self.particles = np.zeros((n_particles, 4), dtype=np.float32)
        self.weights = np.ones(n_particles, dtype=np.float32) / n_particles
        self._init_particles()

    def _init_particles(self):
        # 初始化 x, y (全图均匀分布)
        self.particles[:, 0] = self.rng.uniform(0, Config.MAP_SIZE, self.n_particles)
        self.particles[:, 1] = self.rng.uniform(0, Config.MAP_SIZE, self.n_particles)
        # 初始化 v, phi (若开启软约束则需要合理初始化，否则置 0)
        self.particles[:, 2] = self.rng.uniform(Config.V_MIN, Config.V_MAX, self.n_particles)
        self.particles[:, 3] = self.rng.uniform(0, 2 * np.pi, self.n_particles)
        self.weights.fill(1.0 / self.n_particles)

    def predict(self):
        if not Config.USE_SOFT_CONSTRAINT:
            # === 原有逻辑 (保持训练稳定性) ===
            # 仅对 x, y (前两列) 施加高斯噪声
            noise = self.rng.normal(0.0, Config.PF_MOTION_STD, size=(self.n_particles, 2))
            self.particles[:, :2] += noise
            # 边界裁剪
            self.particles[:, 0] = np.clip(self.particles[:, 0], 0, Config.MAP_SIZE)
            self.particles[:, 1] = np.clip(self.particles[:, 1], 0, Config.MAP_SIZE)
        else:
            # === 新增逻辑: 协同转弯 (CT) 模型预测 + 软约束 ===
            # 1. 动力学传播 (CT Model)
            v = self.particles[:, 2]
            phi = self.particles[:, 3]
            dt = Config.DT

            # 简单假设 omega 扰动
            omega = self.rng.normal(0, Config.TARGET_OMEGA_STD, self.n_particles)

            # 状态更新
            # x += v * cos(phi) * dt
            self.particles[:, 0] += v * np.cos(phi) * dt
            # y += v * sin(phi) * dt
            self.particles[:, 1] += v * np.sin(phi) * dt
            # phi += omega * dt
            self.particles[:, 3] += omega * dt

            # v 施加随机扰动
            self.particles[:, 2] += self.rng.normal(0, Config.TARGET_SPEED_STD, self.n_particles)

            # 2. 应用软约束 (Soft Constraint)
            self.apply_soft_constraint()

            # 3. 边界处理 (Hard constraint for map boundary)
            self.particles[:, 0] = np.clip(self.particles[:, 0], 0, Config.MAP_SIZE)
            self.particles[:, 1] = np.clip(self.particles[:, 1], 0, Config.MAP_SIZE)

    def apply_soft_constraint(self):
        """
        实现论文 3.2.5 节: 基于 Sigmoid 的软约束
        对超出物理极限(如最大速度)的粒子进行平滑惩罚
        """
        # 提取速度 magnitude
        v = np.abs(self.particles[:, 2])
        limit = Config.PF_SOFT_V_LIMIT
        k = Config.PF_SOFT_K

        # 计算违反程度 diff (仅当 v > limit 时 > 0)
        diff = v - limit

        # Sigmoid 惩罚权重: w_soft = 1 / (1 + exp(k * (v - v_max)))
        penalty = 1.0 / (1.0 + np.exp(k * diff))

        # 更新权重
        self.weights *= penalty

        # 归一化防止数值下溢
        w_sum = np.sum(self.weights)
        if w_sum > 0:
            self.weights /= w_sum

    def update(self, _uav_states, detections):
        eps = 1e-6
        for (uav_pos, detected, meas_pos) in detections:
            if detected:
                # 修改点2: 计算距离时只切片取前两列 [:, :2]
                diff = self.particles[:, :2] - np.array(meas_pos, dtype=np.float32)
                dist_sq = np.sum(diff ** 2, axis=1)
                likelihood = np.exp(-0.5 * dist_sq / (Config.PF_MEAS_STD ** 2)) + eps
                self.weights *= likelihood
            else:
                # 修改点3: 负向更新同样切片取前两列
                diff = self.particles[:, :2] - np.array(uav_pos, dtype=np.float32)
                dist_sq = np.sum(diff ** 2, axis=1)
                in_range = dist_sq <= Config.SENSOR_RANGE ** 2
                self.weights[in_range] *= 0.1

        weight_sum = np.sum(self.weights)
        if weight_sum <= 0:
            self.weights.fill(1.0 / self.n_particles)
        else:
            self.weights /= weight_sum

    def resample(self):
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        step = 1.0 / self.n_particles
        start = self.rng.uniform(0, step)
        points = start + step * np.arange(self.n_particles)
        indexes = np.searchsorted(cumulative, points)
        indexes = np.clip(indexes, 0, self.n_particles - 1)
        # resample 会同时复制 x, y, v, phi，保持状态一致性
        self.particles = self.particles[indexes]
        self.weights.fill(1.0 / self.n_particles)

    def estimate_map(self):
        # 修改点4: histogram2d 使用前两列
        hist, _, _ = np.histogram2d(
            self.particles[:, 0],
            self.particles[:, 1],
            bins=[Config.GRID_ROWS, Config.GRID_COLS],
            range=[[0, Config.MAP_SIZE], [0, Config.MAP_SIZE]]
        )
        if hist.max() > 0:
            hist = hist / hist.max()
        return hist


# ===========================
# 4. 核心环境类 (Gymnasium Interface)
# ===========================
class UAVSwarmEnv(gym.Env):
    def __init__(self, n_uav=None, n_target=None):
        super(UAVSwarmEnv, self).__init__()

        n_uav = Config.N_UAV if n_uav is None else n_uav
        n_target = Config.N_TARGET if n_target is None else n_target

        if not (Config.N_UAV_RANGE[0] <= n_uav <= Config.N_UAV_RANGE[1]):
            raise ValueError(f"n_uav must be in range {Config.N_UAV_RANGE}, got {n_uav}")
        if not (Config.N_TARGET_RANGE[0] <= n_target <= Config.N_TARGET_RANGE[1]):
            raise ValueError(f"n_target must be in range {Config.N_TARGET_RANGE}, got {n_target}")

        # 定义动作空间: [加速度, 角速度] -> 连续空间 [cite: 304]
        # action range: [-1, 1], 会在 step 中映射到 [MAX_ACC, MAX_ANG_VEL]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 定义观测空间: 论文表 3-4 [cite: 298]
        # 包含: 自身(x,y,v,phi) + 局部障碍物 + 局部概率 + 邻居信息 + 局部覆盖
        # 这里为了简化网络输入，将 Observation 扁平化为一个向量
        obs_dim = 4 + 3 + 2 + 3 + (n_uav - 1) * 2 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.n_agents = n_uav
        self.n_targets = n_target
        self.agents = []
        self.targets = []
        self.obstacles = []  # 简单圆形障碍物 [(x, y, r), ...]
        self.uav_trajectories = []
        self.target_trajectories = []
        self.detection_points = []

        # 全局状态 (Global State) 容器
        self.global_map_prob = np.zeros((Config.GRID_ROWS, Config.GRID_COLS))  # 目标存在概率图
        self.global_map_cover = np.zeros((Config.GRID_ROWS, Config.GRID_COLS))  # 覆盖图 [cite: 289]

        self.rng = np.random.default_rng()

        # 初始化障碍物 (固定或随机)
        self._init_obstacles()
        self.particle_filters = []
        self.target_detected_by = []
        self.agent_map_prob = []
        self.last_detection = False

    def _init_obstacles(self):
        # 随机生成几个圆形障碍物 [cite: 358]
        for _ in range(5):
            x = self.rng.uniform(200, 800)
            y = self.rng.uniform(200, 800)
            r = self.rng.uniform(30, 60)
            self.obstacles.append((x, y, r))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 使用环境私有 RNG，避免算法内部随机数(如探索噪声)污染环境随机过程
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        self.obstacles = []
        self._init_obstacles()

        # 初始化无人机状态: x, y, v, phi
        self.agents = []
        for i in range(self.n_agents):
            agent = {
                'x': self.rng.uniform(0, 100),  # 起始区
                'y': self.rng.uniform(0, 100),
                'v': Config.V_MIN,
                'phi': self.rng.uniform(0, np.pi / 2),
                'id': i
            }
            self.agents.append(agent)

        # 初始化目标
        self.targets = [Target(self.rng) for _ in range(self.n_targets)]
        self.particle_filters = [ParticleFilter(Config.N_PARTICLES, self.rng) for _ in range(self.n_targets)]
        self.target_detected_by = [None for _ in range(self.n_targets)]

        # 重置地图
        target_maps = [pf.estimate_map() for pf in self.particle_filters]
        if target_maps:
            self.global_map_prob = np.mean(target_maps, axis=0)
        else:
            self.global_map_prob.fill(0.0)
        self.global_map_cover.fill(0.0)
        self.agent_map_prob = [self.global_map_prob.copy() for _ in range(self.n_agents)]

        # 初始化轨迹与检测点
        self.uav_trajectories = [[(agent['x'], agent['y'])] for agent in self.agents]
        self.target_trajectories = [[(target.x, target.y)] for target in self.targets]
        self.detection_points = []
        self.last_detection = False

        return self._get_all_obs(), {}

    def _pos_to_grid(self, x, y):
        # 笛卡尔坐标转栅格坐标 [cite: 117]
        m = int(np.floor(x / Config.GRID_WIDTH))
        n = int(np.floor(y / Config.GRID_WIDTH))
        m = np.clip(m, 0, Config.GRID_ROWS - 1)
        n = np.clip(n, 0, Config.GRID_COLS - 1)
        return m, n

    def step(self, actions):
        """
        核心物理引擎与奖励计算
        actions: list of [acc, ang_vel] for each agent
        """
        rewards = []
        reward_breakdowns = []
        obs_n = []
        infos = {}

        current_step_covered_grids = set()  # 记录本步所有无人机覆盖的网格，用于计算重叠

        # --- 1. 状态更新 (Dynamics Update) ---
        boundary_hits = [False] * self.n_agents
        for i, agent in enumerate(self.agents):
            # 解析动作并解归一化
            acc = np.clip(actions[i][0], -1, 1) * Config.MAX_ACC
            omega = np.clip(actions[i][1], -1, 1) * Config.MAX_ANG_VEL

            # 动力学更新 [cite: 121]
            agent['v'] += acc * Config.DT
            agent['v'] = np.clip(agent['v'], Config.V_MIN, Config.V_MAX)  # 速度约束 [cite: 311]

            agent['phi'] += omega * Config.DT  # 航向角
            # 归一化角度
            agent['phi'] = (agent['phi'] + np.pi) % (2 * np.pi) - np.pi

            agent['x'] += agent['v'] * np.cos(agent['phi']) * Config.DT
            agent['y'] += agent['v'] * np.sin(agent['phi']) * Config.DT

            # 边界约束 [cite: 323]
            clipped_x = np.clip(agent['x'], 0, Config.MAP_SIZE)
            clipped_y = np.clip(agent['y'], 0, Config.MAP_SIZE)
            if clipped_x != agent['x'] or clipped_y != agent['y']:
                boundary_hits[i] = True
            agent['x'] = clipped_x
            agent['y'] = clipped_y
            self.uav_trajectories[i].append((agent['x'], agent['y']))

        # 更新目标
        for idx, target in enumerate(self.targets):
            target.step()
            self.target_trajectories[idx].append((target.x, target.y))

        # 粒子滤波更新 (去除“上帝视角”作弊)
        target_maps = []
        detected_by = [None for _ in range(self.n_targets)]
        any_detection = False
        for target_idx, target in enumerate(self.targets):
            detections = []
            detected_agents = []
            for idx, agent in enumerate(self.agents):
                dist = np.sqrt((agent['x'] - target.x) ** 2 + (agent['y'] - target.y) ** 2)
                detected = dist <= Config.SENSOR_RANGE
                meas_pos = (target.x, target.y) if detected else None
                detections.append(((agent['x'], agent['y']), detected, meas_pos))
                if detected:
                    detected_agents.append((idx, dist))
                    any_detection = True
            if detected_agents:
                detected_agents.sort(key=lambda item: item[1])
                detected_by[target_idx] = detected_agents[0][0]

            pf = self.particle_filters[target_idx]
            pf.predict()
            pf.update(self.agents, detections)
            pf.resample()
            target_maps.append(pf.estimate_map())

        self.target_detected_by = detected_by
        if target_maps:
            self.global_map_prob = np.mean(target_maps, axis=0)
        else:
            self.global_map_prob.fill(0.0)
        self.agent_map_prob = []
        for i in range(self.n_agents):
            eligible_maps = []
            for t_idx, target_map in enumerate(target_maps):
                assigned_agent = detected_by[t_idx]
                if assigned_agent is None or assigned_agent == i:
                    eligible_maps.append(target_map)
            if eligible_maps:
                self.agent_map_prob.append(np.mean(eligible_maps, axis=0))
            else:
                self.agent_map_prob.append(np.zeros_like(self.global_map_prob))
        self.last_detection = any_detection

        # --- 2. 奖励计算 (Reward Calculation) [cite: 351] ---
        newly_covered_count = 0

        # 2.1 覆盖与探索奖励
        agent_grid_coverage = []  # 记录每个agent覆盖了哪些网格

        for i, agent in enumerate(self.agents):
            r_step = 0
            reward_breakdown = {
                "coverage": 0.0,
                "explore": 0.0,
                "unknown": 0.0,
                "detect": 0.0,
                "smooth": 0.0,
                "spin": 0.0,
                "obstacle": 0.0,
                "overlap": 0.0,
                "crash": 0.0,
                "share": 0.0,
                "track": 0.0,
                "boundary": 0.0,
            }

            # (1) 覆盖奖励逻辑
            # 计算当前无人机覆盖的栅格集合
            covered_indices = []
            m_center, n_center = self._pos_to_grid(agent['x'], agent['y'])
            radius_grid = int(Config.SENSOR_RANGE / Config.GRID_WIDTH)

            for r in range(-radius_grid, radius_grid + 1):
                for c in range(-radius_grid, radius_grid + 1):
                    if r ** 2 + c ** 2 <= radius_grid ** 2:
                        gm, gn = m_center + r, n_center + c
                        if 0 <= gm < Config.GRID_ROWS and 0 <= gn < Config.GRID_COLS:
                            grid_idx = (gm, gn)
                            covered_indices.append(grid_idx)

                            # 全局覆盖逻辑 [cite: 327]
                            if self.global_map_cover[gm, gn] == 0:
                                self.global_map_cover[gm, gn] = 1
                                r_step += Config.W_COV  # 发现新区域
                                reward_breakdown["coverage"] += Config.W_COV
                                newly_covered_count += 1

                                # 探索奖励: 如果该区域概率高，奖励更多 [cite: 330]
                                local_prob = self.agent_map_prob[i][gm, gn]
                                if local_prob > 0.5:
                                    r_step += Config.W_EXP
                                    reward_breakdown["explore"] += Config.W_EXP
                                else:
                                    unk_reward = Config.W_UNK * (1.0 - local_prob)
                                    r_step += unk_reward
                                    reward_breakdown["unknown"] += unk_reward

            agent_grid_coverage.append(set(covered_indices))
            current_step_covered_grids.update(covered_indices)

            # (1.5) 目标检测奖励 (连续跟踪则持续给分)
            for target in self.targets:
                dist = np.sqrt((agent['x'] - target.x) ** 2 + (agent['y'] - target.y) ** 2)
                if dist <= Config.SENSOR_RANGE:
                    r_step += Config.W_DETECT
                    reward_breakdown["detect"] += Config.W_DETECT
                    r_step += Config.W_TRACK
                    reward_breakdown["track"] += Config.W_TRACK
                    self.detection_points.append((agent['x'], agent['y']))

            # (2) 平滑飞行奖励 [cite: 343]
            acc_norm = abs(actions[i][0])
            omega_norm = abs(actions[i][1])
            smooth_reward = Config.W_SMOOTH * (acc_norm + omega_norm)
            r_step += smooth_reward
            reward_breakdown["smooth"] += smooth_reward

            # (2.5) 原地转圈惩罚：近几步净位移较小但轨迹长度较大
            if len(self.uav_trajectories[i]) >= 6:
                recent_points = np.array(self.uav_trajectories[i][-6:])
                segment_diffs = np.diff(recent_points, axis=0)
                path_len = float(np.sum(np.linalg.norm(segment_diffs, axis=1)))
                net_disp = float(np.linalg.norm(recent_points[-1] - recent_points[0]))
                if path_len > 0 and net_disp / path_len < 0.3:
                    r_step += Config.W_SPIN
                    reward_breakdown["spin"] += Config.W_SPIN

            # (2.6) 边界惩罚：触发地图边界裁剪
            if boundary_hits[i]:
                r_step += Config.W_BOUNDARY
                reward_breakdown["boundary"] += Config.W_BOUNDARY

            # (3) 障碍物惩罚 [cite: 346]
            dist_to_obs = min([np.sqrt((agent['x'] - ox) ** 2 + (agent['y'] - oy) ** 2) - oradius for ox, oy, oradius in
                               self.obstacles] or [9999])
            if dist_to_obs < Config.SAFE_DIST_OBS:
                if dist_to_obs <= 0:  # 撞击
                    obs_penalty = Config.W_OBS * 2
                    r_step += obs_penalty
                    reward_breakdown["obstacle"] += obs_penalty
                else:  # 警告区
                    obs_penalty = Config.W_OBS * (1 - dist_to_obs / Config.SAFE_DIST_OBS)
                    r_step += obs_penalty
                    reward_breakdown["obstacle"] += obs_penalty

            rewards.append(r_step)
            reward_breakdowns.append(reward_breakdown)

        # 2.2 协同/重叠奖励与避碰 [cite: 336, 349]
        for i in range(self.n_agents):
            # (1) 重叠惩罚
            # 如果两个无人机覆盖了同一个网格，给予负奖励
            # 简化计算：只看中心点距离是否过近导致视野重叠严重
            for j in range(i + 1, self.n_agents):
                dist = np.sqrt((self.agents[i]['x'] - self.agents[j]['x']) ** 2 +
                               (self.agents[i]['y'] - self.agents[j]['y']) ** 2)
                if dist < 1.5 * Config.SENSOR_RANGE:
                    rewards[i] += Config.W_COL
                    rewards[j] += Config.W_COL
                    reward_breakdowns[i]["overlap"] += Config.W_COL
                    reward_breakdowns[j]["overlap"] += Config.W_COL

            # (2) 无人机间避碰 [cite: 349]
            for j in range(i + 1, self.n_agents):
                dist = np.sqrt((self.agents[i]['x'] - self.agents[j]['x']) ** 2 +
                               (self.agents[i]['y'] - self.agents[j]['y']) ** 2)
                if dist < Config.SAFE_DIST_UAV:
                    penalty = Config.W_CRASH * (1 - dist / Config.SAFE_DIST_UAV)
                    rewards[i] += penalty
                    rewards[j] += penalty
                    reward_breakdowns[i]["crash"] += penalty
                    reward_breakdowns[j]["crash"] += penalty
                elif dist < Config.COMM_RANGE:
                    # (3) 信息共享奖励 [cite: 339]
                    rewards[i] += Config.W_SHARE
                    rewards[j] += Config.W_SHARE
                    reward_breakdowns[i]["share"] += Config.W_SHARE
                    reward_breakdowns[j]["share"] += Config.W_SHARE

        # --- 3. 生成观测 (Observation) ---
        obs_n = self._get_all_obs()

        # 检查是否结束 (这里设为永不结束，直到 max_steps)
        terminated = [False] * self.n_agents
        truncated = [False] * self.n_agents

        reward_scale = 10.0
        rewards = [reward / reward_scale for reward in rewards]
        for breakdown in reward_breakdowns:
            for key in breakdown:
                breakdown[key] /= reward_scale

        infos["detections"] = self.last_detection
        infos["reward_breakdown"] = reward_breakdowns
        return obs_n, rewards, terminated, truncated, infos

    def _get_agent_obs(self, agent_index):
        agent = self.agents[agent_index]
        cos_phi = np.cos(agent['phi'])
        sin_phi = np.sin(agent['phi'])

        # 1. 自身状态 (Normalized)
        own_state = [
            agent['x'] / Config.MAP_SIZE,
            agent['y'] / Config.MAP_SIZE,
            (agent['v'] - Config.V_MIN) / (Config.V_MAX - Config.V_MIN),
            agent['phi'] / np.pi
        ]

        # 2. 局部障碍物 (最近距离 + 相对方向) [cite: 298]
        closest_dist = Config.SENSOR_RANGE
        closest_dx = 0.0
        closest_dy = 0.0
        for ox, oy, _ in self.obstacles:
            dist = np.sqrt((agent['x'] - ox) ** 2 + (agent['y'] - oy) ** 2)
            if dist <= Config.SENSOR_RANGE and dist < closest_dist:
                closest_dist = dist
                closest_dx = ox - agent['x']
                closest_dy = oy - agent['y']

        obs_body_x = closest_dx * cos_phi + closest_dy * sin_phi
        obs_body_y = -closest_dx * sin_phi + closest_dy * cos_phi
        obs_info = [
            closest_dist / Config.SENSOR_RANGE,
            obs_body_x / Config.SENSOR_RANGE,
            obs_body_y / Config.SENSOR_RANGE
        ]

        # 3. 粒子滤波估计目标相对向量 (Body Frame)
        pf_dx = 0.0
        pf_dy = 0.0
        if self.particle_filters:
            est_positions = []
            for pf in self.particle_filters:
                if pf.particles.size > 0:
                    # 粒子状态是 [x, y, v, phi]，观测几何只使用前两维避免维度不匹配
                    est_positions.append(np.mean(pf.particles[:, :2], axis=0))
            if est_positions:
                est_positions = np.array(est_positions)
                diffs = est_positions - np.array([agent['x'], agent['y']])
                dist_sq = np.sum(diffs ** 2, axis=1)
                nearest_idx = int(np.argmin(dist_sq))
                pf_dx = diffs[nearest_idx][0]
                pf_dy = diffs[nearest_idx][1]

        pf_body_x = pf_dx * cos_phi + pf_dy * sin_phi
        pf_body_y = -pf_dx * sin_phi + pf_dy * cos_phi
        pf_info = [
            pf_body_x / Config.MAP_SIZE,
            pf_body_y / Config.MAP_SIZE
        ]

        # 4. 局部概率特征 (最大值, 均值, 梯度方向) [cite: 298]
        # 提取局部网格
        m, n = self._pos_to_grid(agent['x'], agent['y'])
        # 简化：直接取当前网格的概率
        agent_map = self.agent_map_prob[agent_index] if self.agent_map_prob else self.global_map_prob
        local_prob = [
            agent_map[m, n],
            float(np.mean(agent_map)),  # 简化的全局/局部感知
            0.0  # 梯度暂略
        ]

        # 5. 邻居信息 (相对位置) [cite: 298]
        neighbor_info = []
        for j, other in enumerate(self.agents):
            if agent_index == j:
                continue
            # 相对坐标
            dx = (other['x'] - agent['x']) / Config.MAP_SIZE
            dy = (other['y'] - agent['y']) / Config.MAP_SIZE
            neighbor_info.extend([dx, dy])

        # 6. 局部覆盖信息
        cov_val = self.global_map_cover[m, n]

        # 拼接
        obs = np.concatenate([own_state, obs_info, pf_info, local_prob, neighbor_info, [cov_val]])
        return obs.astype(np.float32)

    def _get_all_obs(self):
        return [self._get_agent_obs(i) for i in range(self.n_agents)]

    def get_global_state(self):
        """Critic 用的全局状态 [cite: 293]"""
        # 拼接所有 obs + 全局地图特征(Flatten)
        # 为节省显存，这里简化为只拼接所有 obs
        all_obs = self._get_all_obs()
        return np.concatenate(all_obs)
