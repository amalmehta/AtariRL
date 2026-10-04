"""The deep Q-learning agent: epsilon-greedy acting and the Q-learning update.

The 2013 paper has no separate target network: the target
    y = r + gamma * max_a' Q(s', a'; theta)
is computed with the current weights, held fixed (no gradient) for the update.
The loss is the squared error (y - Q(s, a; theta))^2.
"""
from collections import deque

import numpy as np
import torch
from torch import nn

from .model import QNetwork


class FrameHistory:
    """The agent's view of the last `history` frames; padded with the first frame after a reset."""

    def __init__(self, history: int = 4):
        self.frames = deque(maxlen=history)

    def reset(self, frame):
        for _ in range(self.frames.maxlen):
            self.frames.append(frame)

    def push(self, frame):
        self.frames.append(frame)

    def state(self) -> np.ndarray:
        return np.stack(self.frames)


class DQNAgent:
    def __init__(self, n_actions: int, cfg, seed: int = 0):
        torch.manual_seed(seed)
        self.n_actions = n_actions
        self.cfg = cfg
        self.net = QNetwork(n_actions, cfg.history)
        self.optimizer = torch.optim.RMSprop(
            self.net.parameters(), lr=cfg.lr, alpha=cfg.rms_alpha, eps=cfg.rms_eps, centered=True
        )
        self.rng = np.random.default_rng(seed)

    def epsilon(self, step: int) -> float:
        """Linear anneal from eps_start to eps_end, then constant."""
        frac = min(step / self.cfg.eps_anneal_steps, 1.0)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    @torch.no_grad()
    def q_values(self, state: np.ndarray) -> np.ndarray:
        return self.net(torch.from_numpy(state).unsqueeze(0))[0].numpy()

    def act(self, state: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(self.q_values(state).argmax())

    def td_targets(self, rewards, next_states, dones) -> torch.Tensor:
        with torch.no_grad():
            next_max = self.net(next_states).max(dim=1).values
        return rewards + self.cfg.gamma * next_max * (1.0 - dones)

    def learn(self, batch) -> float:
        states, actions, rewards, next_states, dones = (torch.from_numpy(x) for x in batch)
        targets = self.td_targets(rewards, next_states, dones.float())
        q = self.net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.functional.mse_loss(q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def mean_max_q(self, states: np.ndarray) -> float:
        """Figure 2 (right) of the paper: average max Q over a fixed set of states."""
        return self.net(torch.from_numpy(states)).max(dim=1).values.mean().item()

    def save(self, path):
        torch.save({"net": self.net.state_dict(), "n_actions": self.n_actions}, path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location="cpu")["net"])
