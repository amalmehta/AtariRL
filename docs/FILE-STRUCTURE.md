# File Structure

```
AtariRL/
├── README.md                  what it is, a chart, links
├── requirements.txt           pinned dependencies
├── dqn/                       the agent (python -m dqn.train / dqn.play)
│   ├── config.py              every hyperparameter, paper values as defaults
│   ├── env.py                 Atari emulator + preprocessing φ + frame skip
│   ├── model.py               the Q-network
│   ├── replay.py              experience replay memory
│   ├── agent.py               ε-greedy acting, Q-learning update, frame history
│   ├── train.py               Algorithm 1 training loop, logging, evaluation
│   └── play.py                record a trained agent (video + Q-values)
├── scripts/
│   └── build_site_data.py     run logs → website data, images, README chart
├── tests/
│   └── test_dqn.py            unit + end-to-end tests (pytest)
├── docs/                      served by GitHub Pages
│   ├── index.html             the website
│   ├── assets/                videos, Q-value traces, results.json, images
│   ├── INSTRUCTIONS.md        setup, train, watch, rebuild the site
│   ├── SYSTEM-DESIGN.md       architecture, flows, decisions, limits
│   └── FILE-STRUCTURE.md      this file
└── runs/                      (not in git) training output per game
    └── <game>/                config.json, episodes.csv, evals.csv,
                               heldout.npy, best.pt, latest.pt
```

## Where to start reading

1. `dqn/train.py`, the loop that ties everything together.
2. `dqn/agent.py` → `learn()`, the Q-learning update.
3. `dqn/replay.py`, the trickiest part (rebuilding frame stacks).
