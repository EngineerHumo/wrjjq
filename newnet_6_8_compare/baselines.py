import numpy as np

from uav_env import Config


def _angle_wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _compute_action(agent_state, desired_heading, desired_speed):
    heading_error = _angle_wrap(desired_heading - agent_state["phi"])
    desired_omega = np.clip(heading_error / Config.DT, -Config.MAX_ANG_VEL, Config.MAX_ANG_VEL)
    ang_action = desired_omega / Config.MAX_ANG_VEL

    acc = (desired_speed - agent_state["v"]) / Config.DT
    acc = np.clip(acc, -Config.MAX_ACC, Config.MAX_ACC)
    acc_action = acc / Config.MAX_ACC
    return np.array([acc_action, ang_action], dtype=np.float32)


class LawnmowerController:
    def __init__(self, env):
        self.env = env
        self.lane_spacing = max(2 * Config.SENSOR_RANGE * 0.9, Config.GRID_WIDTH)
        self.lane_targets = []
        self.directions = []
        self.reset()

    def reset(self):
        self.lane_targets = self._init_lanes()
        self.directions = [1 for _ in range(self.env.n_agents)]

    def _init_lanes(self):
        lanes = []
        for idx in range(self.env.n_agents):
            lane_y = (idx + 1) / (self.env.n_agents + 1) * Config.MAP_SIZE
            lanes.append(lane_y)
        return lanes

    def select_actions(self):
        actions = []
        for idx, agent in enumerate(self.env.agents):
            lane_y = self.lane_targets[idx]
            desired_speed = Config.V_MAX * 0.9
            target_x = Config.MAP_SIZE if self.directions[idx] > 0 else 0.0
            desired_heading = np.arctan2(lane_y - agent["y"], target_x - agent["x"])

            if abs(agent["y"] - lane_y) < Config.GRID_WIDTH * 0.5:
                desired_heading = 0.0 if self.directions[idx] > 0 else np.pi

            if agent["x"] <= Config.GRID_WIDTH or agent["x"] >= Config.MAP_SIZE - Config.GRID_WIDTH:
                self.directions[idx] *= -1
                lane_y = np.clip(lane_y + self.lane_spacing, 0.0, Config.MAP_SIZE)
                self.lane_targets[idx] = lane_y

            repel = self._obstacle_repulsion(agent)
            if np.linalg.norm(repel) > 0:
                desired_heading = np.arctan2(repel[1], repel[0])

            actions.append(_compute_action(agent, desired_heading, desired_speed))
        return actions

    def _obstacle_repulsion(self, agent):
        repulse = np.zeros(2, dtype=np.float32)
        for ox, oy, radius in self.env.obstacles:
            dx = agent["x"] - ox
            dy = agent["y"] - oy
            dist = np.sqrt(dx * dx + dy * dy) - radius
            if dist < Config.SAFE_DIST_OBS * 1.5:
                strength = (Config.SAFE_DIST_OBS * 1.5 - dist) / max(dist, 1e-3)
                repulse += np.array([dx, dy]) * strength
        return repulse


class GreedyAPFController:
    def __init__(self, env):
        self.env = env

    def reset(self):
        return None

    def select_actions(self):
        actions = []
        for idx, agent in enumerate(self.env.agents):
            target_vec = self._attractive_force(agent)
            repulse = self._repulsive_force(agent, idx)
            force = target_vec + repulse
            if np.linalg.norm(force) < 1e-6:
                desired_heading = agent["phi"]
            else:
                desired_heading = np.arctan2(force[1], force[0])
            desired_speed = Config.V_MAX * 0.8
            actions.append(_compute_action(agent, desired_heading, desired_speed))
        return actions

    def _attractive_force(self, agent):
        prob_map = self.env.global_map_prob
        if prob_map is None or prob_map.size == 0:
            return np.zeros(2, dtype=np.float32)
        idx = np.unravel_index(np.argmax(prob_map), prob_map.shape)
        center_x = (idx[1] + 0.5) * Config.GRID_WIDTH
        center_y = (idx[0] + 0.5) * Config.GRID_WIDTH
        dx = center_x - agent["x"]
        dy = center_y - agent["y"]
        return np.array([dx, dy], dtype=np.float32)

    def _repulsive_force(self, agent, agent_idx):
        repulse = np.zeros(2, dtype=np.float32)
        for ox, oy, radius in self.env.obstacles:
            dx = agent["x"] - ox
            dy = agent["y"] - oy
            dist = np.sqrt(dx * dx + dy * dy) - radius
            if dist < Config.SAFE_DIST_OBS * 2.0:
                strength = (Config.SAFE_DIST_OBS * 2.0 - dist) / max(dist, 1e-3)
                repulse += np.array([dx, dy]) * strength

        for j, other in enumerate(self.env.agents):
            if j == agent_idx:
                continue
            dx = agent["x"] - other["x"]
            dy = agent["y"] - other["y"]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < Config.SAFE_DIST_UAV * 2.0:
                strength = (Config.SAFE_DIST_UAV * 2.0 - dist) / max(dist, 1e-3)
                repulse += np.array([dx, dy]) * strength

        return repulse
