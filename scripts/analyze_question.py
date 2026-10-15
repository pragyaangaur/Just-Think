"""The preregistered analysis of the question-only rating test (PREREGISTRATION_QUESTION.md).

Usage: python scripts/analyze_question.py MODEL
Reads results/<MODEL>/rating_question/<arm>_<condition>.jsonl, and the earlier rating run in
results/<MODEL>/rating/ for the comparisons with the full condition, and writes
question_summary.json next to the new trials.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from justthink.paths import RESULTS
from analyze_rating import describe, paired


def load(model, sub):
    d = {}
    for f in (RESULTS / model / sub).glob("*.jsonl"):
        arm, cond = f.stem.split("_", 1)
        d[(arm, cond)] = {r["trial"]: r for r in map(json.loads, open(f))}
    return d


def changes(d, arm):
    """Per-trial change in rating from off to question."""
    a, b = d.get((arm, "question"), {}), d.get((arm, "off"), {})
    ids = sorted(t for t in set(a) & set(b) if a[t]["valence"] is not None and b[t]["valence"] is not None)
    return np.array([a[t]["valence"] - b[t]["valence"] for t in ids])


def main(model):
    d = load(model, "rating_question")
    old = load(model, "rating")
    S = {"model": model, "protocol": sorted({r["protocol"] for rows in d.values() for r in rows.values()})}
    q1 = paired(d, "A", "question", "off")
    S["Q1 pain, question vs off"] = q1
    if q1:
        v = [r["valence"] for r in d[("A", "question")].values() if r["valence"] is not None]
        below = sum(x < 5 for x in v) / len(v)
        S["pain under question: median and share below 5"] = [float(np.median(v)), below]
        if q1["p_wilcoxon"] < 0.05 and q1["mean"][0] < q1["mean"][1]:
            S["reading"] = "supported" if below >= 0.20 else "mixed"
        else:
            S["reading"] = "not supported"
    a, c = changes(d, "A"), changes(d, "C")
    S["secondary"] = {
        "S1 E question vs off": paired(d, "E", "question", "off"),
        "S2 C question vs off": paired(d, "C", "question", "off"),
        "S3 change in A vs change in C": None if not len(a) or not len(c) else {
            "mean change": [float(a.mean()), float(c.mean())],
            "p_mannwhitney": float(mannwhitneyu(a, c, alternative="two-sided").pvalue)},
    }
    # S4: how much of the full condition's shift the question condition reproduces, using the
    # full ratings of the earlier run for the same trial ids.
    share = {}
    for arm in ("A", "C", "E"):
        if (arm, "full") not in old or (arm, "question") not in d or (arm, "off") not in d:
            continue
        f, q, o = old[(arm, "full")], d[(arm, "question")], d[(arm, "off")]
        ids = [t for t in o if t in f and t in q and None not in (f[t]["valence"], q[t]["valence"], o[t]["valence"])]
        mo = np.mean([o[t]["valence"] for t in ids])
        mq = np.mean([q[t]["valence"] for t in ids])
        mf = np.mean([f[t]["valence"] for t in ids])
        share[arm] = {"n": len(ids), "mean off, question, full": [float(mo), float(mq), float(mf)],
                      "share of the full shift": float((mo - mq) / (mo - mf)) if mo != mf else None}
    S["secondary"]["S4 share of the full shift"] = share
    S["descriptive"] = describe(d)
    agree = {}
    for arm in ("A", "C", "E"):
        if (arm, "off") in d and (arm, "off") in old:
            n, o = d[(arm, "off")], old[(arm, "off")]
            ids = [t for t in n if t in o]
            agree[arm] = [sum(n[t]["valence"] == o[t]["valence"] for t in ids), len(ids)]
    S["S6 off ratings equal to the earlier rating run, by trial id"] = agree
    out = RESULTS / model / "rating_question" / "question_summary.json"
    json.dump(S, open(out, "w"), indent=2)
    print(json.dumps({k: v for k, v in S.items() if k != "descriptive"}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen7b-fp16")
