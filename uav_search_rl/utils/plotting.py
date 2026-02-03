from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectories(
    output_dir: Path,
    obstacles: np.ndarray,
    uav_trajectories: List[np.ndarray],
    target_trajectory: np.ndarray,
    episode: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(obstacles.T == -1, cmap="gray_r", origin="lower")
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    for idx, traj in enumerate(uav_trajectories):
        if traj.size == 0:
            continue
        ax.plot(traj[:, 0], traj[:, 1], color=colors[idx % len(colors)], label=f"UAV {idx + 1}")
        ax.scatter(traj[0, 0], traj[0, 1], color=colors[idx % len(colors)], marker="o")
    if target_trajectory.size > 0:
        ax.plot(target_trajectory[:, 0], target_trajectory[:, 1], color="black", linestyle="--", label="Target")
        ax.scatter(target_trajectory[0, 0], target_trajectory[0, 1], color="black", marker="x")
    ax.set_title(f"Episode {episode + 1} Trajectory")
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / f"trajectory_ep{episode + 1}.png")
    plt.close(fig)


def plot_reward_curve(
    output_dir: Path,
    reward_history: Iterable[float],
    window_sizes: Iterable[int],
    episode: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rewards = np.array(list(reward_history), dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rewards, label="Avg Reward")
    for window in window_sizes:
        if len(rewards) >= window:
            ma_values = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(
                range(window - 1, window - 1 + len(ma_values)),
                ma_values,
                label=f"MA{window}",
            )
    ax.set_title(f"Reward Curve (Episode {episode + 1})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"reward_curve_ep{episode + 1}.png")
    plt.close(fig)
