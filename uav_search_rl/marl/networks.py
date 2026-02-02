from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, output_activation: nn.Module | None = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_activation = output_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out


class Actor(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.model = MLP(input_dim, action_dim, hidden_dim, output_activation=nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)


class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.model = MLP(input_dim, 1, hidden_dim)

    def forward(self, state_action: torch.Tensor) -> torch.Tensor:
        return self.model(state_action)
