from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from uav_search_rl.envs.dynamics import UAVConstraints, clamp_position, step_uav
from uav_search_rl.envs.features import coverage_patch, obstacle_features, probability_features
from uav_search_rl.envs.particle_filter import ParticleFilter, ParticleFilterConfig
from uav_search_rl.envs.target_ct import CTConstraints, ct_step


@dataclass
class EnvConfig:
    grid_m: int
    grid_n: int
    obstacle_ratio: float
    num_uavs: int
    num_targets: int
    dt: float
    fov_radius: float
    comm_range: float
    safe_distance: float
    warning_distance: float
    min_speed: float
    max_speed: float
    max_accel: float
    max_yaw_rate: float
    obs_patch_size: int
    max_neighbors: int
    detect_prob: float
    pf: ParticleFilterConfig
    reward_weights: Dict[str, float]


class UAVSearchEnv:
    def __init__(self, config: EnvConfig, rng: np.random.Generator) -> None:
        self.cfg = config
        self.rng = rng
        self.obstacles = np.zeros((config.grid_m, config.grid_n), dtype=np.int8)
        self.covered = np.zeros((config.grid_m, config.grid_n), dtype=np.float32)
        self.explored = np.zeros((config.grid_m, config.grid_n), dtype=np.float32)
        self.uav_states = np.zeros((config.num_uavs, 5), dtype=np.float32)
        self.prev_actions = np.zeros((config.num_uavs, 2), dtype=np.float32)
        self.target_state = np.zeros(6, dtype=np.float32)
        self.pf = ParticleFilter(config.pf, rng)
        self.collision_count = 0
        self._init_maps()

    def _init_maps(self) -> None:
        num_cells = self.cfg.grid_m * self.cfg.grid_n
        num_obstacles = int(num_cells * self.cfg.obstacle_ratio)
        indices = self.rng.choice(num_cells, size=num_obstacles, replace=False)
        self.obstacles.fill(0)
        self.obstacles.flat[indices] = -1

    def reset(self) -> Tuple[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
        self.covered.fill(0.0)
        self.explored.fill(0.0)
        free_cells = np.argwhere(self.obstacles == 0)
        uav_indices = self.rng.choice(len(free_cells), size=self.cfg.num_uavs, replace=False)
        for idx, cell_idx in enumerate(uav_indices):
            pos = free_cells[cell_idx].astype(np.float32)
            heading = self.rng.uniform(-np.pi, np.pi)
            speed = self.rng.uniform(self.cfg.min_speed, self.cfg.max_speed)
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)
            self.uav_states[idx] = np.array([pos[0], pos[1], vx, vy, heading], dtype=np.float32)
        target_cell = free_cells[self.rng.integers(len(free_cells))].astype(np.float32)
        heading = self.rng.uniform(-np.pi, np.pi)
        speed = self.rng.uniform(self.cfg.pf.speed_range[0], self.cfg.pf.speed_range[1])
        yaw_rate = self.rng.uniform(self.cfg.pf.yaw_rate_range[0], self.cfg.pf.yaw_rate_range[1])
        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)
        self.target_state = np.array([target_cell[0], target_cell[1], vx, vy, heading, yaw_rate], dtype=np.float32)
        self.pf.initialize(target_cell)
        self.prev_actions.fill(0.0)
        self.collision_count = 0
        self._update_coverage()
        return self._get_state(), self._get_obs()

    def _update_coverage(self) -> Tuple[int, int, np.ndarray]:
        covered_before = self.covered.copy()
        covered_mask = np.zeros_like(self.covered, dtype=bool)
        for uav in self.uav_states:
            xs = np.arange(self.cfg.grid_m)[:, None]
            ys = np.arange(self.cfg.grid_n)[None, :]
            dist = np.hypot(xs - uav[0], ys - uav[1])
            covered_mask |= dist <= self.cfg.fov_radius
        covered_mask = covered_mask.astype(np.float32)
        self.covered = np.maximum(self.covered, covered_mask)
        new_explored = ((self.explored < 0.5) & (covered_mask > 0.5)).astype(np.float32)
        self.explored = np.maximum(self.explored, covered_mask)
        covered_gain = int((self.covered - covered_before).sum())
        explored_gain = int(new_explored.sum())
        return covered_gain, explored_gain, covered_mask

    def _get_state(self) -> Dict[str, np.ndarray]:
        return {
            "obstacles": self.obstacles.copy(),
            "uav_states": self.uav_states.copy(),
            "possibility": self.pf.gridify(self.cfg.grid_m, self.cfg.grid_n),
            "env_cov": np.stack([self.covered, self.explored], axis=0),
        }

    def _get_obs(self) -> List[Dict[str, np.ndarray]]:
        prob_map = self.pf.gridify(self.cfg.grid_m, self.cfg.grid_n)
        obs_list: List[Dict[str, np.ndarray]] = []
        for idx, uav in enumerate(self.uav_states):
            min_dist, density = obstacle_features(self.obstacles, uav[:2], self.cfg.fov_radius)
            max_p, avg_p, direction = probability_features(prob_map, uav[:2], self.cfg.fov_radius)
            neighbors = []
            for jdx, other in enumerate(self.uav_states):
                if jdx == idx:
                    continue
                dist = np.hypot(*(other[:2] - uav[:2]))
                if dist <= self.cfg.comm_range:
                    neighbors.append(other)
            neighbors_arr = np.zeros((self.cfg.max_neighbors, 5), dtype=np.float32)
            if neighbors:
                neighbors_arr[: min(len(neighbors), self.cfg.max_neighbors)] = np.stack(neighbors)[: self.cfg.max_neighbors]
            patch = coverage_patch(self.explored, uav[:2], self.cfg.obs_patch_size)
            obs_list.append(
                {
                    "selfstate": uav.copy(),
                    "partobs": np.array([min_dist, density], dtype=np.float32),
                    "partp": np.array([max_p, avg_p, direction], dtype=np.float32),
                    "neighbors": neighbors_arr,
                    "partcov": patch,
                }
            )
        return obs_list

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, np.ndarray], List[Dict[str, np.ndarray]], np.ndarray, bool, Dict]:
        constraints = UAVConstraints(
            min_speed=self.cfg.min_speed,
            max_speed=self.cfg.max_speed,
            max_accel=self.cfg.max_accel,
            max_yaw_rate=self.cfg.max_yaw_rate,
            dt=self.cfg.dt,
        )
        bounds = (0.0, float(self.cfg.grid_m - 1), 0.0, float(self.cfg.grid_n - 1))
        rewards = np.zeros(self.cfg.num_uavs, dtype=np.float32)
        info: Dict[str, float] = {}
        obstacle_penalty = 0.0
        warning_penalty = 0.0
        collision_penalty = 0.0
        energy_penalty = 0.0
        smooth_penalty = 0.0
        for idx in range(self.cfg.num_uavs):
            self.uav_states[idx] = step_uav(self.uav_states[idx], actions[idx], constraints)
            self.uav_states[idx] = clamp_position(self.uav_states[idx], bounds)
            pos = self.uav_states[idx][:2]
            cell_x = np.clip(int(round(pos[0])), 0, self.cfg.grid_m - 1)
            cell_y = np.clip(int(round(pos[1])), 0, self.cfg.grid_n - 1)
            cell = np.array([cell_x, cell_y])
            if self.obstacles[cell[0], cell[1]] == -1:
                obstacle_penalty += 1.0
            warning_penalty += self._warning_penalty(pos)
            speed = np.hypot(self.uav_states[idx, 2], self.uav_states[idx, 3])
            energy_penalty += speed
            smooth_penalty += np.linalg.norm(actions[idx] - self.prev_actions[idx])

        for i in range(self.cfg.num_uavs):
            for j in range(i + 1, self.cfg.num_uavs):
                dist = np.hypot(*(self.uav_states[i, :2] - self.uav_states[j, :2]))
                if dist < self.cfg.safe_distance:
                    collision_penalty += 1.0
        if collision_penalty > 0:
            self.collision_count += int(collision_penalty)

        covered_gain, explored_gain, covered_mask = self._update_coverage()

        ct_constraints = CTConstraints(
            min_speed=self.cfg.pf.speed_range[0],
            max_speed=self.cfg.pf.speed_range[1],
            min_yaw_rate=self.cfg.pf.yaw_rate_range[0],
            max_yaw_rate=self.cfg.pf.yaw_rate_range[1],
            dt=self.cfg.dt,
        )
        target_noise = self.rng.normal(0.0, 0.1, size=(2,))
        self.target_state = ct_step(self.target_state, target_noise, ct_constraints)
        self.target_state[0] = np.clip(self.target_state[0], 0.0, float(self.cfg.grid_m - 1))
        self.target_state[1] = np.clip(self.target_state[1], 0.0, float(self.cfg.grid_n - 1))

        detections = np.zeros(self.cfg.num_uavs, dtype=bool)
        measurement = None
        for idx, uav in enumerate(self.uav_states):
            dist = np.hypot(*(uav[:2] - self.target_state[:2]))
            if dist <= self.cfg.fov_radius and self.rng.random() < self.cfg.detect_prob:
                detections[idx] = True
                measurement = self.target_state[:2] + self.rng.normal(0.0, self.cfg.pf.meas_noise, size=(2,))
                break

        prev_prob = self.pf.gridify(self.cfg.grid_m, self.cfg.grid_n)
        prev_entropy = -np.sum(prev_prob * np.log(prev_prob + 1e-6))
        self.pf.predict(self.cfg.dt)
        self.pf.update(measurement, self.uav_states[:, :2], detections, self.cfg.fov_radius)
        self.pf.resample()
        prob_map = self.pf.gridify(self.cfg.grid_m, self.cfg.grid_n)
        entropy = -np.sum(prob_map * np.log(prob_map + 1e-6))
        entropy_reward = prev_entropy - entropy

        overlap = 0.0
        total_cover = covered_mask.sum() + 1e-6
        if total_cover > 0:
            stacked = np.zeros_like(self.covered)
            for uav in self.uav_states:
                xs = np.arange(self.cfg.grid_m)[:, None]
                ys = np.arange(self.cfg.grid_n)[None, :]
                dist = np.hypot(xs - uav[0], ys - uav[1])
                stacked += (dist <= self.cfg.fov_radius).astype(np.float32)
            overlap = (stacked >= 2).sum() / total_cover

        share_reward = 0.0
        for i in range(self.cfg.num_uavs):
            for j in range(i + 1, self.cfg.num_uavs):
                dist = np.hypot(*(self.uav_states[i, :2] - self.uav_states[j, :2]))
                if dist <= self.cfg.comm_range and detections.any():
                    share_reward += 1.0

        cover_reward = covered_gain / (self.cfg.grid_m * self.cfg.grid_n)
        explore_reward = explored_gain / (self.cfg.grid_m * self.cfg.grid_n)
        obstacle_penalty = obstacle_penalty / self.cfg.num_uavs
        warning_penalty = warning_penalty / self.cfg.num_uavs
        collision_penalty = collision_penalty / max(1, self.cfg.num_uavs - 1)
        smooth_penalty = smooth_penalty / self.cfg.num_uavs
        energy_penalty = energy_penalty / self.cfg.num_uavs

        info.update(
            {
                "cover_reward": cover_reward,
                "explore_reward": explore_reward,
                "cover_rate": float(self.covered.sum() / (self.cfg.grid_m * self.cfg.grid_n)),
                "explore_rate": float(self.explored.sum() / (self.cfg.grid_m * self.cfg.grid_n)),
                "overlap": overlap,
                "share_reward": share_reward,
                "smooth_penalty": smooth_penalty,
                "obstacle_penalty": obstacle_penalty,
                "collision_penalty": collision_penalty,
                "entropy_reward": entropy_reward,
                "energy_penalty": energy_penalty,
                "warning_penalty": warning_penalty,
                "entropy": entropy,
                "detection": float(detections.any()),
            }
        )

        self.prev_actions = actions.copy()
        done = False
        weights = self.cfg.reward_weights
        for idx in range(self.cfg.num_uavs):
            rewards[idx] = (
                weights["cover"] * cover_reward
                + weights["explore"] * explore_reward
                - weights["overlap"] * overlap
                + weights["share"] * share_reward
                - weights["smooth"] * smooth_penalty
                - weights["obstacle"] * obstacle_penalty
                - weights["collision"] * collision_penalty
                + weights["entropy"] * entropy_reward
                - weights["energy"] * energy_penalty
                - weights["warning"] * warning_penalty
            )

        return self._get_state(), self._get_obs(), rewards, done, info

    def _warning_penalty(self, pos: np.ndarray) -> float:
        xs = np.arange(self.cfg.grid_m)[:, None]
        ys = np.arange(self.cfg.grid_n)[None, :]
        distances = np.hypot(xs - pos[0], ys - pos[1])
        obstacle_mask = self.obstacles == -1
        if not obstacle_mask.any():
            return 0.0
        min_dist = distances[obstacle_mask].min()
        if min_dist <= self.cfg.safe_distance:
            return 1.0
        if min_dist <= self.cfg.warning_distance:
            return (self.cfg.warning_distance - min_dist) / (self.cfg.warning_distance - self.cfg.safe_distance + 1e-6)
        return 0.0
