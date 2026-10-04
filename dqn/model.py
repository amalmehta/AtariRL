"""The Q-network from Section 5 of the paper.

Input: 84x84x4 stack of preprocessed frames.
  conv 16 filters 8x8 stride 4 -> ReLU    (16 x 20 x 20)
  conv 32 filters 4x4 stride 2 -> ReLU    (32 x 9 x 9)
  fully connected 256 -> ReLU
  fully connected, one output per action (the Q-value of that action)
"""
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, n_actions: int, history: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(history, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 9 * 9, 256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, frames_uint8: torch.Tensor) -> torch.Tensor:
        x = frames_uint8.float() / 255.0
        return self.head(self.features(x))
