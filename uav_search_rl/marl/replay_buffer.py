from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayBatch:
    states: np.ndarray
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, num_agents: int, state_dim: int, obs_dim: int, action_dim: int) -> None:
        self.capacity = capacity
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, num_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, num_agents), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_state: np.ndarray,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        idx = self.ptr
        self.states[idx] = state
        self.obs[idx] = obs
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_states[idx] = next_state
        self.next_obs[idx] = next_obs
        self.dones[idx] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        indices = np.random.randint(0, self.size, size=batch_size)
        return ReplayBatch(
            states=self.states[indices],
            obs=self.obs[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_states=self.next_states[indices],
            next_obs=self.next_obs[indices],
            dones=self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size
