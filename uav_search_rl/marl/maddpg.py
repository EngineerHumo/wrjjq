from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from uav_search_rl.marl.networks import Actor, Critic


@dataclass
class MADDPGConfig:
    num_agents: int
    obs_dim: int
    state_dim: int
    action_dim: int
    action_scale: np.ndarray
    hidden_dim: int
    actor_lr: float
    critic_lr: float
    gamma: float
    tau: float
    grad_clip: float
    device: str


class MADDPG:
    def __init__(self, config: MADDPGConfig) -> None:
        self.cfg = config
        self.device = torch.device(config.device)
        self.action_scale = torch.as_tensor(config.action_scale, dtype=torch.float32, device=self.device)
        self.actors = nn.ModuleList([
            Actor(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
            for _ in range(config.num_agents)
        ])
        self.critics = nn.ModuleList([
            Critic(config.state_dim + config.num_agents * config.action_dim, config.hidden_dim).to(self.device)
            for _ in range(config.num_agents)
        ])
        self.target_actors = nn.ModuleList([
            Actor(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
            for _ in range(config.num_agents)
        ])
        self.target_critics = nn.ModuleList([
            Critic(config.state_dim + config.num_agents * config.action_dim, config.hidden_dim).to(self.device)
            for _ in range(config.num_agents)
        ])
        self.actor_opts = [torch.optim.Adam(actor.parameters(), lr=config.actor_lr) for actor in self.actors]
        self.critic_opts = [torch.optim.Adam(critic.parameters(), lr=config.critic_lr) for critic in self.critics]
        self._hard_update()

    def _hard_update(self) -> None:
        for tgt, src in zip(self.target_actors, self.actors):
            tgt.load_state_dict(src.state_dict())
        for tgt, src in zip(self.target_critics, self.critics):
            tgt.load_state_dict(src.state_dict())

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions = []
        for idx, actor in enumerate(self.actors):
            action = actor(obs_t[idx]).detach().cpu().numpy()
            actions.append(action)
        return np.stack(actions)

    def update(self, batch) -> dict:
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)

        metrics = {}
        next_actions = []
        for idx, target_actor in enumerate(self.target_actors):
            scaled = target_actor(next_obs[:, idx]) * self.action_scale
            next_actions.append(scaled)
        next_actions_cat = torch.cat(next_actions, dim=-1)
        actions_cat = actions.view(actions.shape[0], -1)

        for i in range(self.cfg.num_agents):
            critic = self.critics[i]
            target_critic = self.target_critics[i]
            critic_opt = self.critic_opts[i]

            target_q = target_critic(torch.cat([next_states, next_actions_cat], dim=-1)).squeeze(-1)
            y = rewards[:, i] + self.cfg.gamma * (1.0 - dones.squeeze(-1)) * target_q
            q = critic(torch.cat([states, actions_cat], dim=-1)).squeeze(-1)
            critic_loss = nn.MSELoss()(q, y.detach())
            critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), self.cfg.grad_clip)
            critic_opt.step()

            actor = self.actors[i]
            actor_opt = self.actor_opts[i]
            cur_actions = actions.clone()
            cur_actions[:, i] = actor(obs[:, i]) * self.action_scale
            cur_actions_cat = cur_actions.view(cur_actions.shape[0], -1)
            actor_loss = -critic(torch.cat([states, cur_actions_cat], dim=-1)).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), self.cfg.grad_clip)
            actor_opt.step()

            metrics[f"critic_loss_{i}"] = float(critic_loss.item())
            metrics[f"actor_loss_{i}"] = float(actor_loss.item())

        self.soft_update()
        return metrics

    def soft_update(self) -> None:
        for target, source in zip(self.target_actors, self.actors):
            for tgt_param, src_param in zip(target.parameters(), source.parameters()):
                tgt_param.data.copy_(self.cfg.tau * src_param.data + (1.0 - self.cfg.tau) * tgt_param.data)
        for target, source in zip(self.target_critics, self.critics):
            for tgt_param, src_param in zip(target.parameters(), source.parameters()):
                tgt_param.data.copy_(self.cfg.tau * src_param.data + (1.0 - self.cfg.tau) * tgt_param.data)

    def state_dict(self) -> dict:
        return {
            "actors": [actor.state_dict() for actor in self.actors],
            "critics": [critic.state_dict() for critic in self.critics],
            "target_actors": [actor.state_dict() for actor in self.target_actors],
            "target_critics": [critic.state_dict() for critic in self.target_critics],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        for actor, sd in zip(self.actors, state_dict["actors"]):
            actor.load_state_dict(sd)
        for critic, sd in zip(self.critics, state_dict["critics"]):
            critic.load_state_dict(sd)
        for actor, sd in zip(self.target_actors, state_dict["target_actors"]):
            actor.load_state_dict(sd)
        for critic, sd in zip(self.target_critics, state_dict["target_critics"]):
            critic.load_state_dict(sd)
