from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict


@dataclass
class MetricLogger:
    window: int = 100
    sums: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    history: Dict[str, Deque[float]] = field(default_factory=dict)

    def update(self, key: str, value: float) -> None:
        self.sums[key] = self.sums.get(key, 0.0) + value
        self.counts[key] = self.counts.get(key, 0) + 1
        if key not in self.history:
            self.history[key] = deque(maxlen=self.window)
        self.history[key].append(value)

    def mean(self, key: str) -> float:
        values = self.history.get(key)
        if not values:
            return 0.0
        return sum(values) / len(values)

    def reset(self) -> None:
        self.sums.clear()
        self.counts.clear()
        self.history.clear()
