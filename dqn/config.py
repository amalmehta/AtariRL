"""Hyperparameters, with the values from Mnih et al. 2013 as defaults.

Where the 2013 paper is silent (discount, learning rate, replay start size),
the value comes from the 2015 Nature follow-up and is marked "not in 2013".
All step counts are *agent steps*: one agent step = `frame_skip` emulator frames.
"""
from dataclasses import dataclass, asdict


@dataclass
class Config:
    game: str = "pong"
    seed: int = 0

    # Environment (Section 4.1 / 5 of the paper)
    frame_skip: int = 4            # paper: k=4 for every game except Space Invaders (k=3)
    history: int = 4               # last 4 preprocessed frames form the network input
    clip_rewards: bool = True      # positive -> +1, negative -> -1, zero -> 0

    # Learning (Section 5)
    total_steps: int = 10_000_000  # paper: 10 million frames
    replay_capacity: int = 1_000_000  # paper: one million most recent frames
    batch_size: int = 32
    gamma: float = 0.99            # not in 2013 (Nature value)
    lr: float = 2.5e-4             # not in 2013 (Nature value); optimiser is RMSProp
    rms_alpha: float = 0.95        # RMSProp squared-gradient momentum (Nature)
    rms_eps: float = 0.01          # RMSProp min squared gradient (Nature)
    update_every: int = 1          # Algorithm 1 does one minibatch update per step
    replay_start: int = 50_000     # not in 2013 (Nature value): random play before learning
    target_update: int = 0         # not in 2013: steps between target-network refreshes (0 = none, as in 2013)

    # Exploration (Section 5)
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_anneal_steps: int = 1_000_000  # paper: annealed over the first million frames
    eval_eps: float = 0.05         # paper evaluates with epsilon = 0.05

    # Bookkeeping (Figure 2: an "epoch" is 50,000 minibatch updates)
    eval_every: int = 50_000
    eval_steps: int = 10_000
    heldout_states: int = 500      # fixed states for the average-max-Q curve
    threads: int = 4

    def to_dict(self):
        return asdict(self)


def frame_skip_for(game: str) -> int:
    """Space Invaders uses k=3 so the blinking lasers stay visible (Section 5)."""
    return 3 if game == "space_invaders" else 4
