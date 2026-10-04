"""Experience replay memory (Section 4 of the paper).

Each 84x84 frame is stored once. A state is the stack of the 4 most recent
frames, rebuilt at sampling time, so 1M transitions cost ~7 GB instead of ~28 GB.

Slot i holds: the newest frame of state s_i, the action a_i taken from it,
the clipped reward r_i, and whether the episode ended after that action.
The next state s_{i+1} ends with the frame in slot i+1.
"""
import numpy as np


class ReplayMemory:
    def __init__(self, capacity: int, history: int = 4, frame_shape=(84, 84), seed: int = 0):
        self.capacity = capacity
        self.history = history
        self.frames = np.zeros((capacity, *frame_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.cursor = 0      # next slot to write
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def add(self, frame, action, reward, done):
        i = self.cursor
        self.frames[i] = frame
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.cursor = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _stack_indices(self, idx: np.ndarray) -> np.ndarray:
        """Slot indices of the frames making up the state that ends at `idx`.

        Frames from a previous episode are replaced by the episode's first frame,
        matching how the agent pads its own history after a reset.
        """
        h = self.history
        out = np.empty((len(idx), h), dtype=np.int64)
        out[:, -1] = idx
        valid = np.ones(len(idx), dtype=bool)
        for p in range(h - 2, -1, -1):
            back = idx - (h - 1 - p)
            if self.size < self.capacity:
                valid &= back >= 0
            back %= self.capacity
            valid &= ~self.dones[back]
            out[:, p] = np.where(valid, back, out[:, p + 1])
        return out

    def _valid(self, idx: np.ndarray) -> np.ndarray:
        newest = (self.cursor - 1) % self.capacity
        ok = idx != newest  # its next frame has not been observed yet
        if self.size == self.capacity:
            # The oldest slots' histories have been overwritten by the newest data.
            age = (idx - self.cursor) % self.capacity
            ok &= age >= self.history - 1
        return ok

    def sample(self, batch_size: int):
        """Uniformly sample transitions. Returns uint8 stacks (B, 4, 84, 84)."""
        assert self.size > self.history, "not enough experience to sample"
        idx = np.empty(0, dtype=np.int64)
        while len(idx) < batch_size:
            cand = self.rng.integers(0, self.size, size=batch_size * 2)
            idx = np.concatenate([idx, cand[self._valid(cand)]])
        idx = idx[:batch_size]
        nxt = (idx + 1) % self.capacity
        states = self.frames[self._stack_indices(idx)]
        next_states = self.frames[self._stack_indices(nxt)]
        return states, self.actions[idx], self.rewards[idx], next_states, self.dones[idx]
