"""Turn training runs into the data, images and charts the website and README use.

    python scripts/build_site_data.py

Reads runs/<game>/{config.json, evals.csv, episodes.csv}; writes into docs/assets/:
    results.json          learning curves + paper reference scores for the website
    preprocess_*.png      the stages of the preprocessing function phi
    learning-curves.png   chart for the README
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dqn.env import CROP_SIZE, CROP_TOP, AtariEnv  # noqa: E402

ASSETS = ROOT / "docs" / "assets"
GAMES = ["pong", "breakout"]

# Earlier runs kept for comparison: runs/<folder> -> label. Both used the paper's
# RMSProp settings and never learned Pong; the published Pong run is the fast recipe.
BASELINES = {
    "pong": [("pong-no-target", "paper settings, no target network"),
             ("pong-target", "paper settings + target network")],
}

# Table 1 of Mnih et al. 2013 (average total reward, eps = 0.05).
PAPER = {
    "pong": {"random": -20.7, "human": -3, "dqn": 20, "dqn_best": 21},
    "breakout": {"random": 1.2, "human": 31, "dqn": 168, "dqn_best": 225},
}


def read_csv(path):
    with open(path) as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def smooth_episodes(rows, window=20, points=200):
    """Rolling mean of training-episode scores, thinned to ~points entries."""
    if not rows:
        return []
    scores = np.array([r["score"] for r in rows])
    steps = np.array([r["step"] for r in rows])
    out = []
    for i in np.linspace(0, len(rows) - 1, min(points, len(rows))).astype(int):
        lo = max(0, i - window + 1)
        out.append([int(steps[i]), round(float(scores[lo:i + 1].mean()), 2)])
    return out


def build_results():
    results = {}
    for game in GAMES:
        run = ROOT / "runs" / game
        if not (run / "evals.csv").exists():
            continue
        cfg = json.loads((run / "config.json").read_text())
        evals = read_csv(run / "evals.csv")
        episodes = read_csv(run / "episodes.csv")
        results[game] = {
            "config": cfg,
            "evals": [[int(e["step"]), e["eval_score"], e["avg_max_q"]] for e in evals],
            "train": smooth_episodes(episodes),
            "episodes": len(episodes),
            "hours": round(max(e["seconds"] for e in evals) / 3600, 1) if evals else 0,
            "best_eval": max((e["eval_score"] for e in evals), default=None),
            "paper": PAPER[game],
        }
        results[game]["baselines"] = [
            {"name": label, "evals": [[int(e["step"]), e["eval_score"], e["avg_max_q"]]
                                      for e in read_csv(ROOT / "runs" / folder / "evals.csv")]}
            for folder, label in BASELINES.get(game, [])
            if (ROOT / "runs" / folder / "evals.csv").exists()]
    (ASSETS / "results.json").write_text(json.dumps(results, separators=(",", ":")))
    return results


def save_px(arr, name, scale):
    img = Image.fromarray(arr)
    img.resize((img.width * scale, img.height * scale), Image.NEAREST).save(ASSETS / name)


def build_preprocess_images():
    """One real Pong moment, shown at each stage of phi."""
    env = AtariEnv("pong")
    env.reset()
    rng = np.random.default_rng(3)
    gray = np.empty((210, 160), dtype=np.uint8)
    recent = []
    for _ in range(70):  # play until the ball is in the middle of the court
        frame, _, _ = env.step(int(rng.integers(env.n_actions)))
        recent = (recent + [frame])[-4:]
    save_px(env.rgb_screen(), "preprocess_1_raw.png", 2)
    env.ale.getScreenGrayscale(gray)
    save_px(gray, "preprocess_2_gray.png", 2)
    small = np.asarray(Image.fromarray(gray).resize((84, 110), Image.BILINEAR))
    boxed = np.stack([small] * 3, axis=-1).copy()
    boxed[[CROP_TOP, CROP_TOP + CROP_SIZE - 1], :] = [235, 104, 52]  # crop lines
    save_px(boxed, "preprocess_3_small.png", 4)
    for i, f in enumerate(recent):
        save_px(f, f"preprocess_4_stack{i}.png", 3)


def build_readme_chart(results):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 3.2), squeeze=False)
    for ax, (game, r) in zip(axes[0], results.items()):
        steps = [e[0] / 1e6 for e in r["evals"]]
        for b, color in zip(r["baselines"], ["#a3a29c", "#eb6834"]):
            ax.plot([e[0] / 1e6 for e in b["evals"]], [e[1] for e in b["evals"]],
                    color=color, lw=1.5, label=b["name"])
        ax.plot(steps, [e[1] for e in r["evals"]], color="#2a78d6", lw=2, label="published agent")
        if r["baselines"]:
            ax.legend(fontsize=7, frameon=False, loc="center right")
        ax.axhline(r["paper"]["dqn"], color="#52514e", ls="--", lw=1)
        ax.axhline(r["paper"]["random"], color="#a3a29c", ls=":", lw=1)
        ref = ax.get_yaxis_transform()  # x in axes fraction, y in data units
        ax.text(1, r["paper"]["dqn"], "paper DQN ", va="bottom", ha="right", fontsize=8,
                color="#52514e", transform=ref)
        ax.text(1, r["paper"]["random"], "random ", va="bottom", ha="right", fontsize=8,
                color="#7a7974", transform=ref)
        ax.set_title(game.title(), loc="left", fontsize=11)
        ax.set_xlabel("training steps (millions)", fontsize=9)
        ax.set_ylabel("score per episode", fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#e5e4e0", lw=0.8)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "learning-curves.png", dpi=150)


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    build_preprocess_images()
    res = build_results()
    if res:
        build_readme_chart(res)
    print("wrote", sorted(p.name for p in ASSETS.iterdir()))
