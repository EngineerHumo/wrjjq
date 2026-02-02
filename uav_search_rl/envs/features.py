from __future__ import annotations

from typing import Tuple

import numpy as np


def obstacle_features(obstacles: np.ndarray, position: np.ndarray, radius: float) -> Tuple[float, float]:
    grid_m, grid_n = obstacles.shape
    xs = np.arange(grid_m)[:, None]
    ys = np.arange(grid_n)[None, :]
    distances = np.hypot(xs - position[0], ys - position[1])
    mask = distances <= radius
    obstacle_mask = (obstacles == -1) & mask
    density = obstacle_mask.sum() / (mask.sum() + 1e-6)
    if obstacle_mask.any():
        min_dist = distances[obstacle_mask].min()
    else:
        min_dist = radius
    return float(min_dist), float(density)


def probability_features(prob_map: np.ndarray, position: np.ndarray, radius: float) -> Tuple[float, float, float]:
    grid_m, grid_n = prob_map.shape
    xs = np.arange(grid_m)[:, None]
    ys = np.arange(grid_n)[None, :]
    distances = np.hypot(xs - position[0], ys - position[1])
    mask = distances <= radius
    local = prob_map * mask
    max_p = local.max() if local.size > 0 else 0.0
    avg_p = local.sum() / (mask.sum() + 1e-6)

    grad_x, grad_y = np.gradient(prob_map)
    gx = grad_x[int(np.clip(position[0], 0, grid_m - 1)), int(np.clip(position[1], 0, grid_n - 1))]
    gy = grad_y[int(np.clip(position[0], 0, grid_m - 1)), int(np.clip(position[1], 0, grid_n - 1))]
    direction = np.arctan2(gy, gx)
    return float(max_p), float(avg_p), float(direction)


def coverage_patch(cov_map: np.ndarray, position: np.ndarray, patch_size: int) -> np.ndarray:
    half = patch_size // 2
    grid_m, grid_n = cov_map.shape
    x = int(np.clip(position[0], 0, grid_m - 1))
    y = int(np.clip(position[1], 0, grid_n - 1))
    x_min = max(0, x - half)
    x_max = min(grid_m, x + half + 1)
    y_min = max(0, y - half)
    y_max = min(grid_n, y + half + 1)
    patch = np.zeros((patch_size, patch_size), dtype=np.float32)
    patch_x_min = half - (x - x_min)
    patch_y_min = half - (y - y_min)
    patch[patch_x_min:patch_x_min + (x_max - x_min), patch_y_min:patch_y_min + (y_max - y_min)] = cov_map[
        x_min:x_max, y_min:y_max
    ]
    return patch
