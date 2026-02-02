from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class EvalMetrics:
    cover_rate: float
    explore_rate: float
    overlap_rate: float
    detection_rate: float
    first_detection_time: float
    collision_rate: float
    entropy_drop: float
    smoothness: float
    energy: float


def aggregate_episode(info_history: List[Dict[str, float]], total_cells: int, max_steps: int) -> EvalMetrics:
    cover_rate = info_history[-1].get("cover_rate", 0.0)
    explore_rate = info_history[-1].get("explore_rate", 0.0)
    overlap_rate = np.mean([info["overlap"] for info in info_history])
    detection_rate = np.mean([info["detection"] for info in info_history])
    first_detection_time = max_steps
    for step, info in enumerate(info_history):
        if info["detection"] > 0.5:
            first_detection_time = step
            break
    collision_rate = np.mean([info["collision_penalty"] for info in info_history])
    entropy_drop = info_history[0]["entropy"] - info_history[-1]["entropy"]
    smoothness = np.mean([info["smooth_penalty"] for info in info_history])
    energy = np.mean([info["energy_penalty"] for info in info_history])
    return EvalMetrics(
        cover_rate=cover_rate,
        explore_rate=explore_rate,
        overlap_rate=overlap_rate,
        detection_rate=detection_rate,
        first_detection_time=float(first_detection_time),
        collision_rate=collision_rate,
        entropy_drop=entropy_drop,
        smoothness=smoothness,
        energy=energy,
    )


def summarize_metrics(metrics: List[EvalMetrics]) -> Dict[str, float]:
    keys = metrics[0].__dict__.keys()
    summary = {}
    for key in keys:
        summary[key] = float(np.mean([getattr(m, key) for m in metrics]))
    return summary


def compute_score(summary: Dict[str, float]) -> float:
    return (
        summary["cover_rate"]
        + summary["explore_rate"]
        + summary["detection_rate"]
        + summary["entropy_drop"]
        - summary["collision_rate"]
        - summary["overlap_rate"]
    )
