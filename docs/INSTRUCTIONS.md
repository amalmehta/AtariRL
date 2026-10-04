# Instructions

## Setup

Needs Python 3.11 (3.9–3.12 work) and about 8 GB of free RAM per training run at the paper's 1M replay size.

```bash
git clone https://github.com/amalmehta/AtariRL && cd AtariRL
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The Atari ROMs ship inside `ale-py`; nothing else to download. `torch==2.2.2` is pinned because it's the last release with Intel-Mac wheels; on Apple Silicon or Linux any newer PyTorch also works.

Check everything is wired up:

```bash
.venv/bin/python -m pytest tests
```

## Train

```bash
.venv/bin/python -m dqn.train --game pong --steps 1500000 --update-every 4 --eps-anneal-steps 250000
```

This is the setting used for the published Pong run. For the exact paper settings (update every step, 10M steps), drop the flags:

```bash
.venv/bin/python -m dqn.train --game breakout
```

Games from the paper: `beam_rider`, `breakout`, `enduro`, `pong`, `qbert`, `seaquest`, `space_invaders` (any ALE ROM name works).

Useful flags (all optional):

| Flag | Default | Meaning |
|---|---|---|
| `--steps` | 10,000,000 | agent steps to train (1 step = 4 frames) |
| `--out` | `runs/<game>` | output folder |
| `--update-every` | 1 | agent steps between gradient updates |
| `--eps-anneal-steps` | 1,000,000 | steps for ε to fall from 1 to 0.1 |
| `--replay-capacity` | 1,000,000 | transitions kept in memory |
| `--replay-start` | 50,000 | random steps before learning starts |
| `--eval-every` / `--eval-steps` | 50,000 / 10,000 | evaluation schedule |
| `--threads` | 4 | PyTorch CPU threads |
| `--lr`, `--seed` | 2.5e-4, 0 | |

Progress prints every 10,000 steps, plus an evaluation line every epoch. Logs go to `runs/<game>/episodes.csv` and `evals.csv`; weights to `best.pt` and `latest.pt`. Stop any time with Ctrl-C; the last epoch's weights are kept.

## Watch a trained agent

```bash
.venv/bin/python -m dqn.play --run runs/pong --out pong
```

Writes `pong.mp4`, `pong.gif` and `pong_q.json` (the Q-value of every action at every step). `--weights latest.pt` uses the newest weights instead of the best; `--eps 0` plays fully greedy.

## Rebuild the website

```bash
for g in pong breakout; do .venv/bin/python -m dqn.play --run runs/$g --out docs/assets/$g; done
.venv/bin/python scripts/build_site_data.py
.venv/bin/python -m http.server 8765 -d docs
```

Then open http://localhost:8765. The page loads its data with `fetch`, so open it through a server rather than as a file. GitHub Pages serves the `docs/` folder of `main`.
