from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CTConstraints:
    min_speed: float
    max_speed: float
    min_yaw_rate: float
    max_yaw_rate: float
    dt: float


def ct_step(state: np.ndarray, noise: np.ndarray, constraints: CTConstraints) -> np.ndarray:
    """State: [x, y, vx, vy, heading, yaw_rate]."""
    x, y, vx, vy, heading, yaw_rate = state
    yaw_rate = np.clip(yaw_rate + noise[0], constraints.min_yaw_rate, constraints.max_yaw_rate)
    speed = np.hypot(vx, vy)
    speed = np.clip(speed + noise[1], constraints.min_speed, constraints.max_speed)
    heading = heading + yaw_rate * constraints.dt
    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)
    x = x + vx * constraints.dt
    y = y + vy * constraints.dt
    return np.array([x, y, vx, vy, heading, yaw_rate], dtype=np.float32)
