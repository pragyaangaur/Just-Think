"""Figures for the write-up, from results/<MODEL>/analysis and the trial logs.

Usage: python scripts/figures.py MODEL
Writes PNG and PDF files to results/<MODEL>/figures/.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from justthink.paths import RESULTS

LABEL = {"A": "A pain", "C": "C random", "D": "D inert", "B": "B sham", "E": "E pleasant",
         "F": "F pain + puzzle", "Avt": "A varied ticks", "Alow": "A 0.75 dose", "G": "G no button"}
COLOR = {"A": "#c0392b", "B": "#e59866", "C": "#5d6d7e", "D": "#aab7b8", "E": "#27ae60",
         "F": "#922b21", "Avt": "#cd6155", "Alow": "#f1948a", "G": "#2c3e50"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, out, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def main(model):
    base = RESULTS / model
    out = base / "figures"
    out.mkdir(parents=True, exist_ok=True)
    S = json.load(open(base / "analysis" / "summary.json"))
    ticks = pd.read_csv(base / "analysis" / "ticks.csv")
    data = {f.stem: [json.loads(l) for l in open(f)] for f in (base / "trials").glob("*.jsonl")}
    arms = [a for a in LABEL if a in S["arms"]]

    # 1. Share of trials that pressed at least once, with Wilson intervals.
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, a in enumerate(arms):
        d = S["arms"][a]
        ax.bar(i, 100 * d["rate"], color=COLOR[a])
        ax.errorbar(i, 100 * d["rate"], yerr=[[100 * (d["rate"] - d["ci"][0])], [100 * (d["ci"][1] - d["rate"])]],
                    color="black", capsize=3, lw=1)
    ax.set_xticks(range(len(arms)), [f"{LABEL[a]} (n={S['arms'][a]['n']})" for a in arms], rotation=30, ha="right")
    ax.set_ylabel("Pressed at least once (%)")
    save(fig, out, "fig1_press_rate")

    # 2. Survival: share of trials that have not yet pressed, by tick.
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for a in [x for x in ("A", "B", "C", "D", "E") if x in data]:
        first = np.array([r["presses"][0]["tick"] if r["presses"] else 99 for r in data[a]])
        ax.step(range(41), [100 * np.mean(first >= t) for t in range(41)], where="post", color=COLOR[a], label=LABEL[a])
    ax.set_xlabel("Tick")
    ax.set_ylabel("Not yet pressed (%)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, out, "fig2_time_to_first_press")

    # 3. Repetition: similarity of each tick to the most similar earlier tick.
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for a in [x for x in ("G", "D", "A", "C") if x in data]:
        g = ticks[ticks.arm == a].groupby("tick").embed_prev.mean()
        ax.plot(g.index, g.values, color=COLOR[a], label=LABEL[a])
    ax.set_xlabel("Tick")
    ax.set_ylabel("Similarity to an earlier tick")
    ax.legend(frameon=False, fontsize=8)
    save(fig, out, "fig3_repetition")

    # 4. The state read back: monitor projection on steered and unsteered replies.
    fig, ax = plt.subplots(figsize=(5, 3.4))
    rows = [(a, e["steered"] > 0, e["proj"]) for a in ("A", "C", "E", "B", "D") if a in data
            for r in data[a] for e in r["events"] if e["proj"] is not None]
    df = pd.DataFrame(rows, columns=["arm", "steered", "proj"])
    xs, labels = [], []
    for i, (a, s) in enumerate([(a, s) for a in ("A", "C", "E", "B", "D") if a in data for s in (False, True)]):
        v = df[(df.arm == a) & (df.steered == s)].proj
        if len(v):
            ax.boxplot(v, positions=[len(xs)], widths=0.6, showfliers=False,
                       patch_artist=True, boxprops=dict(facecolor=COLOR[a], alpha=0.8 if s else 0.3))
            xs.append(len(xs))
            labels.append(f"{a} {'on' if s else 'off'}")
    ax.set_xticks(xs, labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Projection on the pain direction")
    save(fig, out, "fig4_state_check")

    # 5. Phase 1 valence rating by arm.
    fig, ax = plt.subplots(figsize=(5, 3.4))
    va = [a for a in ("A", "B", "C", "D", "E") if a in data]
    ax.boxplot([[r["valence"] for r in data[a] if r.get("valence")] for a in va], showfliers=False,
               tick_labels=[LABEL[a] for a in va])
    ax.set_ylabel("Rating of the sample (1 unpleasant, 9 pleasant)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save(fig, out, "fig5_valence")
    print("wrote figures to", out)


if __name__ == "__main__":
    main(sys.argv[1])
