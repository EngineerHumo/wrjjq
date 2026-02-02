from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class UAVConstraints:
    min_speed: float
    max_speed: float
    max_accel: float
    max_yaw_rate: float
    dt: float


def step_uav(state: np.ndarray, action: np.ndarray, constraints: UAVConstraints) -> np.ndarray:
    """State: [x, y, vx, vy, heading], action: [accel, yaw_rate]."""
    x, y, vx, vy, heading = state
    accel = np.clip(action[0], -constraints.max_accel, constraints.max_accel)
    yaw_rate = np.clip(action[1], -constraints.max_yaw_rate, constraints.max_yaw_rate)

    speed = np.hypot(vx, vy)
    speed = np.clip(speed + accel * constraints.dt, constraints.min_speed, constraints.max_speed)
    heading = heading + yaw_rate * constraints.dt
    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)
    x = x + vx * constraints.dt
    y = y + vy * constraints.dt
    return np.array([x, y, vx, vy, heading], dtype=np.float32)


def clamp_position(state: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    x, y, vx, vy, heading = state
    min_pos, max_pos = bounds
    x = np.clip(x, min_pos, max_pos)
    y = np.clip(y, min_pos, max_pos)
    return np.array([x, y, vx, vy, heading], dtype=np.float32)
