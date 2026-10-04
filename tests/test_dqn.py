import numpy as np
import torch

from dqn.agent import DQNAgent, FrameHistory
from dqn.config import Config, frame_skip_for
from dqn.env import AtariEnv, preprocess
from dqn.model import QNetwork
from dqn.replay import ReplayMemory
from dqn.train import train


def frame(value):
    return np.full((84, 84), value, dtype=np.uint8)


# --- preprocessing / environment -------------------------------------------

def test_preprocess_shape_and_dtype():
    out = preprocess(np.random.randint(0, 255, (210, 160), dtype=np.uint8))
    assert out.shape == (84, 84) and out.dtype == np.uint8


def test_env_frame_skip_and_minimal_actions():
    env = AtariEnv("pong", frame_skip=4)
    assert env.n_actions == 6
    f = env.reset()
    assert f.shape == (84, 84)
    before = env.ale.getEpisodeFrameNumber()
    env.step(0)
    assert env.ale.getEpisodeFrameNumber() - before == 4


def test_space_invaders_uses_k3():
    assert frame_skip_for("space_invaders") == 3 and frame_skip_for("pong") == 4


# --- replay memory -----------------------------------------------------------

def test_replay_stacks_frames_in_order():
    m = ReplayMemory(100)
    for i in range(10):
        m.add(frame(i), i, 0.0, False)
    stacks = m.frames[m._stack_indices(np.array([5]))][0]
    assert [s[0, 0] for s in stacks] == [2, 3, 4, 5]


def test_replay_pads_with_first_frame_of_episode():
    m = ReplayMemory(100)
    for i in range(10):
        m.add(frame(i), 0, 0.0, i == 3)   # episode ends after slot 3; slot 4 starts a new one
    stacks = m.frames[m._stack_indices(np.array([5]))][0]
    assert [s[0, 0] for s in stacks] == [4, 4, 4, 5]


def test_replay_start_of_buffer_pads_without_wrapping():
    m = ReplayMemory(100)
    for i in range(10):
        m.add(frame(i), 0, 0.0, False)
    stacks = m.frames[m._stack_indices(np.array([1]))][0]
    assert [s[0, 0] for s in stacks] == [0, 0, 0, 1]


def test_replay_next_state_and_never_samples_newest():
    m = ReplayMemory(50, seed=1)
    for i in range(20):
        m.add(frame(i), i, float(i), False)
    s, a, r, s2, d = m.sample(256)
    assert s.shape == (256, 4, 84, 84)
    assert np.all(a != 19)                               # newest slot has no next frame yet
    assert np.all(s2[:, -1, 0, 0] == s[:, -1, 0, 0] + 1)  # next state ends one frame later
    assert np.all(r == a)


def test_replay_wraps_and_skips_overwritten_histories():
    m = ReplayMemory(10, seed=2)
    for i in range(25):
        m.add(frame(i), i, 0.0, False)
    s, a, *_ = m.sample(500)
    # Oldest surviving slots (actions 15, 16, 17) have histories that were overwritten.
    assert not set(a.tolist()) & {15, 16, 17, 24}
    # Every stack is four consecutive frames.
    assert np.all(np.diff(s[:, :, 0, 0].astype(int), axis=1) == 1)


# --- network and agent -------------------------------------------------------

def test_network_shapes_match_paper():
    net = QNetwork(n_actions=4)
    q = net(torch.zeros(2, 4, 84, 84, dtype=torch.uint8))
    assert q.shape == (2, 4)
    convs = [m for m in net.features if isinstance(m, torch.nn.Conv2d)]
    assert [(c.out_channels, c.kernel_size, c.stride) for c in convs] == [(16, (8, 8), (4, 4)),
                                                                          (32, (4, 4), (2, 2))]
    assert net.head[0].out_features == 256


def test_epsilon_schedule():
    agent = DQNAgent(4, Config(eps_anneal_steps=100))
    assert agent.epsilon(0) == 1.0
    assert abs(agent.epsilon(50) - 0.55) < 1e-9
    assert abs(agent.epsilon(10_000) - 0.1) < 1e-9


def test_td_target_masks_terminal_states():
    agent = DQNAgent(3, Config(gamma=0.5))
    with torch.no_grad():
        for p in agent.net.parameters():
            p.zero_()
        agent.net.head[2].bias.copy_(torch.tensor([1.0, 4.0, 2.0]))  # Q = [1, 4, 2] everywhere
    states = torch.zeros(2, 4, 84, 84, dtype=torch.uint8)
    y = agent.td_targets(torch.tensor([1.0, 1.0]), states, torch.tensor([0.0, 1.0]))
    assert torch.allclose(y, torch.tensor([1.0 + 0.5 * 4.0, 1.0]))


def test_learning_reduces_loss_on_fixed_batch():
    agent = DQNAgent(3, Config(gamma=0.0, lr=1e-3))
    rng = np.random.default_rng(0)
    batch = (rng.integers(0, 255, (32, 4, 84, 84), dtype=np.uint8), rng.integers(0, 3, 32),
             rng.choice([-1.0, 0.0, 1.0], 32).astype(np.float32),
             rng.integers(0, 255, (32, 4, 84, 84), dtype=np.uint8), np.zeros(32, dtype=bool))
    first = agent.learn(batch)
    for _ in range(100):
        last = agent.learn(batch)
    assert last < first * 0.5


def test_frame_history_pads_after_reset():
    h = FrameHistory(4)
    h.reset(frame(7))
    h.push(frame(8))
    assert [f[0, 0] for f in h.state()] == [7, 7, 7, 8]


# --- end to end ----------------------------------------------------------------

def test_training_smoke(tmp_path):
    cfg = Config(total_steps=600, replay_start=200, eval_every=300, eval_steps=100,
                 heldout_states=20, threads=1, update_every=4)
    train(cfg, tmp_path, log_every=10**9)
    assert (tmp_path / "latest.pt").exists() and (tmp_path / "best.pt").exists()
    evals = (tmp_path / "evals.csv").read_text().strip().splitlines()
    assert len(evals) == 3  # header + 2 epochs
