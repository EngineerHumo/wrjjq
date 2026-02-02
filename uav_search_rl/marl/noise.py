from __future__ import annotations

import numpy as np


class OUNoise:
    def __init__(self, size: int, theta: float = 0.15, sigma: float = 0.2) -> None:
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.state = np.zeros(size, dtype=np.float32)

    def reset(self) -> None:
        self.state.fill(0.0)

    def sample(self) -> np.ndarray:
        dx = self.theta * (-self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state


class GaussianNoise:
    def __init__(self, size: int, sigma: float = 0.2) -> None:
        self.size = size
        self.sigma = sigma

    def reset(self) -> None:
        return None

    def sample(self) -> np.ndarray:
        return self.sigma * np.random.randn(self.size)
