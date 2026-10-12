"""The preregistered analysis of the rating follow-up (PREREGISTRATION_RATING.md).

Usage: python scripts/analyze_rating.py MODEL
Reads results/<MODEL>/rating/<arm>_<condition>.jsonl and writes rating_summary.json there.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from justthink import protocol as P
from justthink.paths import RESULTS

ARMS = ("A", "C", "E")


def load(model):
    d = {}
    for f in (RESULTS / model / "rating").glob("*.jsonl"):
        arm, cond = f.stem.split("_", 1)
        d[(arm, cond)] = {r["trial"]: r for r in map(json.loads, open(f))}
    return d


def paired(d, arm, c1, c0):
    if (arm, c1) not in d or (arm, c0) not in d:
        return None
    a, b = d[(arm, c1)], d[(arm, c0)]
    ids = sorted(t for t in set(a) & set(b) if a[t]["valence"] is not None and b[t]["valence"] is not None)
    x = np.array([a[t]["valence"] for t in ids])
    y = np.array([b[t]["valence"] for t in ids])
    diff = x - y
    p = float(wilcoxon(x, y).pvalue) if np.any(diff != 0) else 1.0
    return {"arm": arm, "conditions": [c1, c0], "n_pairs": len(ids), "mean": [float(x.mean()), float(y.mean())],
            "median": [float(np.median(x)), float(np.median(y))], "lower_in_first": int((diff < 0).sum()),
            "higher_in_first": int((diff > 0).sum()), "p_wilcoxon": p}


def asymmetry(d, cond):
    if ("A", cond) not in d or ("E", cond) not in d:
        return None
    a = [r["valence"] for r in d[("A", cond)].values() if r["valence"] is not None]
    e = [r["valence"] for r in d[("E", cond)].values() if r["valence"] is not None]
    below, above = sum(v < 5 for v in a), sum(v > 5 for v in e)
    p = fisher_exact([[below, len(a) - below], [above, len(e) - above]])[1]
    return {"condition": cond, "A rated below 5": [below, len(a)], "E rated above 5": [above, len(e)], "p_fisher": p}


def describe(d):
    out = {}
    for (arm, cond), rows in sorted(d.items()):
        v = [r["valence"] for r in rows.values()]
        got = [x for x in v if x is not None]
        out[f"{arm}_{cond}"] = {"n": len(v), "missing": len(v) - len(got),
                                "counts": {str(k): got.count(k) for k in range(1, 10) if got.count(k)},
                                "below_5": sum(x < 5 for x in got), "above_5": sum(x > 5 for x in got),
                                "median": float(np.median(got)) if got else None}
    return out


def main(model):
    d = load(model)
    S = {"model": model, "protocol": sorted({r["protocol"] for rows in d.values() for r in rows.values()}),
         "R1 pain, full vs off": paired(d, "A", "full", "off"),
         "R2 asymmetry under full": asymmetry(d, "full")}
    a_full = S["R1 pain, full vs off"]
    if a_full:
        rows = [r["valence"] for r in d[("A", "full")].values() if r["valence"] is not None]
        S["pain under full: median and share below 5"] = [float(np.median(rows)), sum(v < 5 for v in rows) / len(rows)]
    S["secondary"] = {
        "R1 header vs off": paired(d, "A", "header", "off"),
        "R2 asymmetry under header": asymmetry(d, "header"),
        "C full vs off": paired(d, "C", "full", "off"),
        "E full vs off": paired(d, "E", "full", "off"),
    }
    S["descriptive"] = describe(d)
    main_trials = RESULTS / model / "trials"
    if main_trials.exists():
        agree = {}
        for arm in ARMS:
            f = main_trials / f"{arm}.jsonl"
            if f.exists() and (arm, "off") in d:
                old = {r["trial"]: r["valence"] for r in map(json.loads, open(f))}
                new = d[(arm, "off")]
                ids = [t for t in new if t in old]
                agree[arm] = [sum(old[t] == new[t]["valence"] for t in ids), len(ids)]
        S["off ratings equal to the main study, by trial id"] = agree
    out = RESULTS / model / "rating" / "rating_summary.json"
    json.dump(S, open(out, "w"), indent=2)
    print(json.dumps({k: v for k, v in S.items() if k != "descriptive"}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen7b-fp16")
