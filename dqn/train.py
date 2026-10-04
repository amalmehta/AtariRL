"""Algorithm 1 of the paper: deep Q-learning with experience replay.

    python -m dqn.train --game pong --steps 1000000 --out runs/pong

Writes to the output folder:
    config.json      the hyperparameters used
    episodes.csv     one row per training episode (raw, unclipped score)
    evals.csv        one row per epoch: eval score with eps=0.05 and average max Q
    heldout.npy      the fixed states used for the average-max-Q curve
    latest.pt        latest weights; best.pt: weights with the best eval score
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .agent import DQNAgent, FrameHistory
from .config import Config, frame_skip_for
from .env import AtariEnv
from .replay import ReplayMemory


def collect_heldout_states(env: AtariEnv, n: int, history: int, rng) -> np.ndarray:
    """Run a random policy and keep n states (the paper does this before training)."""
    states, hist = [], FrameHistory(history)
    hist.reset(env.reset())
    steps = 0
    while len(states) < n:
        frame, _, done = env.step(int(rng.integers(env.n_actions)))
        steps += 1
        if done:
            hist.reset(env.reset())
            continue
        hist.push(frame)
        if steps % 5 == 0:  # spread them out instead of keeping consecutive frames
            states.append(hist.state())
    return np.stack(states)


def evaluate(agent: DQNAgent, env: AtariEnv, n_steps: int, eps: float):
    """Play n_steps with an eps-greedy policy. Returns mean score of completed episodes."""
    hist = FrameHistory(agent.cfg.history)
    hist.reset(env.reset())
    scores, score = [], 0.0
    for _ in range(n_steps):
        frame, reward, done = env.step(agent.act(hist.state(), eps))
        score += reward
        if done:
            scores.append(score)
            score = 0.0
            hist.reset(env.reset())
        else:
            hist.push(frame)
    if not scores:  # one long unfinished episode: report its score so far
        scores.append(score)
    return float(np.mean(scores)), len(scores)


def train(cfg: Config, out: Path, log_every: int = 10_000):
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    torch.set_num_threads(cfg.threads)
    rng = np.random.default_rng(cfg.seed)

    env = AtariEnv(cfg.game, cfg.frame_skip, cfg.seed)
    eval_env = AtariEnv(cfg.game, cfg.frame_skip, cfg.seed + 1)
    agent = DQNAgent(env.n_actions, cfg, cfg.seed)
    memory = ReplayMemory(min(cfg.replay_capacity, cfg.total_steps), cfg.history, seed=cfg.seed)

    heldout = collect_heldout_states(env, cfg.heldout_states, cfg.history, rng)
    np.save(out / "heldout.npy", heldout)

    ep_file = open(out / "episodes.csv", "w", newline="")
    ep_log = csv.writer(ep_file)
    ep_log.writerow(["step", "episode", "score", "length", "epsilon", "seconds"])
    ev_file = open(out / "evals.csv", "w", newline="")
    ev_log = csv.writer(ev_file)
    ev_log.writerow(["step", "epoch", "eval_score", "eval_episodes", "avg_max_q", "seconds"])

    hist = FrameHistory(cfg.history)
    frame = env.reset()
    hist.reset(frame)
    episode, score, length, best = 0, 0.0, 0, -np.inf
    losses, start = [], time.time()

    for step in range(1, cfg.total_steps + 1):
        eps = agent.epsilon(step)
        action = agent.act(hist.state(), eps)
        next_frame, reward, done = env.step(action)
        clipped = float(np.sign(reward)) if cfg.clip_rewards else reward
        memory.add(frame, action, clipped, done)
        score += reward
        length += 1

        if done:
            episode += 1
            ep_log.writerow([step, episode, score, length, round(eps, 4), round(time.time() - start, 1)])
            ep_file.flush()
            score, length = 0.0, 0
            frame = env.reset()
            hist.reset(frame)
        else:
            frame = next_frame
            hist.push(frame)

        if step >= cfg.replay_start and step % cfg.update_every == 0:
            losses.append(agent.learn(memory.sample(cfg.batch_size)))

        if cfg.target_update and step % cfg.target_update == 0:
            agent.sync_target()

        if step % log_every == 0:
            rate = step / (time.time() - start)
            loss = np.mean(losses) if losses else float("nan")
            print(f"step {step:>9,}  episodes {episode:>5}  eps {eps:.3f}  "
                  f"loss {loss:.4f}  {rate:.0f} steps/s", flush=True)
            losses = []

        if step % cfg.eval_every == 0:
            eval_score, n_eps = evaluate(agent, eval_env, cfg.eval_steps, cfg.eval_eps)
            q = agent.mean_max_q(heldout)
            ev_log.writerow([step, step // cfg.eval_every, eval_score, n_eps, round(q, 5),
                             round(time.time() - start, 1)])
            ev_file.flush()
            agent.save(out / "latest.pt")
            if eval_score > best:
                best = eval_score
                agent.save(out / "best.pt")
            print(f"== epoch {step // cfg.eval_every}: eval score {eval_score:.1f} "
                  f"over {n_eps} episodes, avg max Q {q:.3f}", flush=True)

    agent.save(out / "latest.pt")
    ep_file.close()
    ev_file.close()


def main():
    p = argparse.ArgumentParser(description="Train a 2013-style DQN agent on an Atari game.")
    p.add_argument("--game", default="pong", help="ALE ROM name, e.g. pong, breakout, seaquest")
    p.add_argument("--steps", type=int, default=Config.total_steps, help="agent steps to train for")
    p.add_argument("--out", type=Path, default=None, help="output folder (default runs/<game>)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=Config.threads)
    p.add_argument("--replay-capacity", type=int, default=Config.replay_capacity)
    p.add_argument("--replay-start", type=int, default=Config.replay_start)
    p.add_argument("--eps-anneal-steps", type=int, default=Config.eps_anneal_steps)
    p.add_argument("--eval-every", type=int, default=Config.eval_every)
    p.add_argument("--eval-steps", type=int, default=Config.eval_steps)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--update-every", type=int, default=Config.update_every,
                   help="agent steps between minibatch updates (paper: 1)")
    p.add_argument("--target-update", type=int, default=Config.target_update,
                   help="steps between target-network refreshes; 0 = none (paper: 0)")
    a = p.parse_args()
    cfg = Config(game=a.game, seed=a.seed, total_steps=a.steps, threads=a.threads,
                 frame_skip=frame_skip_for(a.game), replay_capacity=a.replay_capacity,
                 replay_start=a.replay_start, eps_anneal_steps=a.eps_anneal_steps,
                 eval_every=a.eval_every, eval_steps=a.eval_steps, lr=a.lr,
                 update_every=a.update_every, target_update=a.target_update)
    train(cfg, a.out or Path("runs") / a.game)


if __name__ == "__main__":
    main()
