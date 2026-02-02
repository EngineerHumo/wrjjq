from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ParticleFilterConfig:
    num_particles: int
    init_pos_sigma: float
    speed_range: Tuple[float, float]
    heading_range: Tuple[float, float]
    yaw_rate_range: Tuple[float, float]
    process_noise: float
    meas_noise: float
    neg_decay: float
    sigmoid_k: float
    soft_vmax: float
    soft_yaw_rate_max: float


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class ParticleFilter:
    def __init__(self, config: ParticleFilterConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.particles = np.zeros((config.num_particles, 6), dtype=np.float32)
        self.weights = np.ones(config.num_particles, dtype=np.float32) / config.num_particles

    def initialize(self, init_pos: np.ndarray) -> None:
        pos = self.rng.normal(loc=init_pos, scale=self.config.init_pos_sigma, size=(self.config.num_particles, 2))
        speeds = self.rng.uniform(self.config.speed_range[0], self.config.speed_range[1], size=(self.config.num_particles,))
        headings = self.rng.uniform(self.config.heading_range[0], self.config.heading_range[1], size=(self.config.num_particles,))
        yaw_rates = self.rng.uniform(
            self.config.yaw_rate_range[0], self.config.yaw_rate_range[1], size=(self.config.num_particles,)
        )
        vx = speeds * np.cos(headings)
        vy = speeds * np.sin(headings)
        self.particles = np.stack(
            [pos[:, 0], pos[:, 1], vx, vy, headings, yaw_rates], axis=1
        ).astype(np.float32)
        self.weights.fill(1.0 / self.config.num_particles)

    def predict(self, dt: float) -> None:
        noise = self.rng.normal(0.0, self.config.process_noise, size=(self.config.num_particles, 2))
        yaw_rate = self.particles[:, 5] + noise[:, 0]
        speed = np.hypot(self.particles[:, 2], self.particles[:, 3]) + noise[:, 1]
        heading = self.particles[:, 4] + yaw_rate * dt
        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)
        self.particles[:, 0] += vx * dt
        self.particles[:, 1] += vy * dt
        self.particles[:, 2] = vx
        self.particles[:, 3] = vy
        self.particles[:, 4] = heading
        self.particles[:, 5] = yaw_rate

    def update(
        self,
        measurement: np.ndarray | None,
        uav_positions: np.ndarray,
        detect_flags: np.ndarray,
        fov_radius: float,
    ) -> None:
        if measurement is not None:
            distances = np.linalg.norm(self.particles[:, :2] - measurement[None, :], axis=1)
            likelihood = np.exp(-(distances ** 2) / (2 * self.config.meas_noise ** 2))
            self.weights *= likelihood
        else:
            if uav_positions.size > 0:
                for idx, pos in enumerate(uav_positions):
                    if not detect_flags[idx]:
                        distances = np.linalg.norm(self.particles[:, :2] - pos[None, :], axis=1)
                        inside = distances < fov_radius
                        self.weights[inside] *= self.config.neg_decay

        speed = np.hypot(self.particles[:, 2], self.particles[:, 3])
        speed_penalty = sigmoid(-self.config.sigmoid_k * (speed - self.config.soft_vmax))
        yaw_penalty = sigmoid(-self.config.sigmoid_k * (np.abs(self.particles[:, 5]) - self.config.soft_yaw_rate_max))
        self.weights *= speed_penalty * yaw_penalty
        self.normalize()

    def normalize(self) -> None:
        total = np.sum(self.weights)
        if total <= 1e-12:
            self.weights.fill(1.0 / self.config.num_particles)
        else:
            self.weights /= total

    def resample(self) -> None:
        cumulative = np.cumsum(self.weights)
        step = 1.0 / self.config.num_particles
        start = self.rng.uniform(0, step)
        points = start + step * np.arange(self.config.num_particles)
        indices = np.searchsorted(cumulative, points)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.config.num_particles)

    def gridify(self, grid_m: int, grid_n: int) -> np.ndarray:
        grid = np.zeros((grid_m, grid_n), dtype=np.float32)
        xs = np.clip(self.particles[:, 0].round().astype(int), 0, grid_m - 1)
        ys = np.clip(self.particles[:, 1].round().astype(int), 0, grid_n - 1)
        for idx, (x, y) in enumerate(zip(xs, ys)):
            grid[x, y] += self.weights[idx]
        total = grid.sum()
        if total > 0:
            grid /= total
        return grid
