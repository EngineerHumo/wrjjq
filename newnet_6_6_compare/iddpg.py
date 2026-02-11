import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class IDDPGConfig:
    LR_ACTOR = 1e-4
    LR_CRITIC = 1e-3
    GAMMA = 0.95
    TAU = 0.01
    BATCH_SIZE = 128
    BUFFER_SIZE = 100000
    HIDDEN_DIM = 64


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, IDDPGConfig.HIDDEN_DIM)
        self.fc2 = nn.Linear(IDDPGConfig.HIDDEN_DIM, IDDPGConfig.HIDDEN_DIM)
        self.fc3 = nn.Linear(IDDPGConfig.HIDDEN_DIM, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, IDDPGConfig.HIDDEN_DIM * 2)
        self.fc2 = nn.Linear(IDDPGConfig.HIDDEN_DIM * 2, IDDPGConfig.HIDDEN_DIM * 2)
        self.fc3 = nn.Linear(IDDPGConfig.HIDDEN_DIM * 2, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        self.pointer = 0
        self.size = 0
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, obs, act, rew, next_obs, done):
        idx = self.pointer
        self.obs_buf[idx] = obs
        self.act_buf[idx] = act
        self.rew_buf[idx] = rew
        self.next_obs_buf[idx] = next_obs
        self.done_buf[idx] = done
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.FloatTensor(self.obs_buf[idxs]).to(device),
            torch.FloatTensor(self.act_buf[idxs]).to(device),
            torch.FloatTensor(self.rew_buf[idxs]).to(device),
            torch.FloatTensor(self.next_obs_buf[idxs]).to(device),
            torch.FloatTensor(self.done_buf[idxs]).to(device),
        )


class IDDPGAgent:
    def __init__(self, obs_dim, act_dim):
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.target_actor = Actor(obs_dim, act_dim).to(device)
        self.target_critic = Critic(obs_dim, act_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=IDDPGConfig.LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=IDDPGConfig.LR_CRITIC)
        self.memory = ReplayBuffer(IDDPGConfig.BUFFER_SIZE, obs_dim, act_dim)

    def get_action(self, obs, noise_std=0.0):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()[0]
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        return action

    def update(self):
        if self.memory.size < IDDPGConfig.BATCH_SIZE:
            return None
        obs, act, rew, next_obs, done = self.memory.sample(IDDPGConfig.BATCH_SIZE)

        with torch.no_grad():
            next_action = self.target_actor(next_obs)
            target_q = self.target_critic(next_obs, next_action)
            y = rew + IDDPGConfig.GAMMA * target_q * (1 - done)

        current_q = self.critic(obs, act)
        critic_loss = F.mse_loss(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(obs, self.actor(obs)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1 - IDDPGConfig.TAU) + param.data * IDDPGConfig.TAU)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1 - IDDPGConfig.TAU) + param.data * IDDPGConfig.TAU)

        return critic_loss.item(), actor_loss.item()

    def save(self, path_prefix):
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pth")

    def load(self, path_prefix):
        self.actor.load_state_dict(torch.load(f"{path_prefix}_actor.pth", map_location=device))
        self.critic.load_state_dict(torch.load(f"{path_prefix}_critic.pth", map_location=device))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())


class IDDPG:
    def __init__(self, n_agents, obs_dim, act_dim):
        self.agents = [IDDPGAgent(obs_dim, act_dim) for _ in range(n_agents)]

    def select_actions(self, obs_n, noise_std=0.0):
        return [agent.get_action(obs, noise_std) for agent, obs in zip(self.agents, obs_n)]

    def update(self):
        losses = []
        for agent in self.agents:
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
        return losses

    def save(self, path):
        for idx, agent in enumerate(self.agents):
            agent.save(f"{path}/agent_{idx}")

    def load(self, path):
        for idx, agent in enumerate(self.agents):
            agent.load(f"{path}/agent_{idx}")
