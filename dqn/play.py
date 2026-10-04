"""Watch a trained agent play, and record what it sees and what it predicts.

    python -m dqn.play --run runs/pong --out docs/assets/pong

Writes:
    <out>.mp4        the episode at real speed (one video frame per agent step)
    <out>.gif        the first few seconds, for the README
    <out>_q.json     Q-value of every action at every step, plus rewards
                     (the data behind the paper's Figure 3 value-function plot)
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from .agent import DQNAgent, FrameHistory
from .config import Config
from .env import AtariEnv


def play_episode(agent: DQNAgent, env: AtariEnv, eps: float, max_steps: int):
    hist = FrameHistory(agent.cfg.history)
    hist.reset(env.reset())
    screens, qs, actions, rewards = [env.rgb_screen()], [], [], []
    for _ in range(max_steps):
        q = agent.q_values(hist.state())
        action = agent.act(hist.state(), eps)
        frame, reward, done = env.step(action)
        qs.append([round(float(x), 3) for x in q])
        actions.append(action)
        rewards.append(reward)
        screens.append(env.rgb_screen())
        if done:
            break
        hist.push(frame)
    return screens, qs, actions, rewards


def main():
    p = argparse.ArgumentParser(description="Record a trained agent playing one episode.")
    p.add_argument("--run", type=Path, required=True, help="training output folder")
    p.add_argument("--weights", default="best.pt")
    p.add_argument("--out", type=Path, required=True, help="output path prefix")
    p.add_argument("--eps", type=float, default=0.05, help="exploration while playing (paper: 0.05)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--gif-seconds", type=float, default=8)
    a = p.parse_args()

    cfg = Config(**json.loads((a.run / "config.json").read_text()))
    env = AtariEnv(cfg.game, cfg.frame_skip, seed=a.seed)
    agent = DQNAgent(env.n_actions, cfg, seed=a.seed)
    agent.load(a.run / a.weights)

    screens, qs, actions, rewards = play_episode(agent, env, a.eps, a.max_steps)
    fps = round(60 / cfg.frame_skip)  # the Atari runs at 60 Hz; we see every k-th frame
    a.out.parent.mkdir(parents=True, exist_ok=True)
    big = [np.repeat(np.repeat(s, 2, axis=0), 2, axis=1) for s in screens]  # 320x420, crisp pixels
    imageio.mimwrite(f"{a.out}.mp4", big, fps=fps, quality=7, macro_block_size=4)
    gif_frames = screens[: int(a.gif_seconds * fps): 2]
    imageio.mimwrite(f"{a.out}.gif", gif_frames, duration=2000 / fps, loop=0)

    names = [act.name for act in env.actions]
    data = {"game": cfg.game, "fps": fps, "action_names": names, "q": qs,
            "actions": actions, "rewards": rewards, "score": float(sum(rewards))}
    Path(f"{a.out}_q.json").write_text(json.dumps(data, separators=(",", ":")))
    print(f"{cfg.game}: score {data['score']:.0f} in {len(actions)} steps -> {a.out}.mp4")


if __name__ == "__main__":
    main()
