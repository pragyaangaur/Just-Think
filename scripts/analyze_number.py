"""The preregistered analysis of the neutral number test (PREREGISTRATION_NUMBER.md).

Usage: python scripts/analyze_number.py MODEL
Reads results/<MODEL>/number/trials.jsonl and writes number_summary.json next to it.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from justthink import protocol as P
from justthink.paths import RESULTS

PROMPTS = tuple(P.NUMBER_PROMPTS)
DIRS = P.NUMBER_DIRECTIONS


def load(model):
    d = {}
    for r in map(json.loads, open(RESULTS / model / "number" / "trials.jsonl")):
        d[(r["prompt"], r["direction"], r["trial"])] = r
    ids = sorted({t for _, _, t in d})
    # Only trial ids with all 8 cells are used.
    ids = [t for t in ids if all((p, k, t) in d for p in PROMPTS for k in DIRS)]
    return d, ids


def wil(x):
    x = np.asarray(x, float)
    return float(wilcoxon(x).pvalue) if np.any(x != 0) else 1.0


def shift(d, ids, prompt, direction, key="expected_digit"):
    return np.array([d[(prompt, direction, t)][key] - d[(prompt, "none", t)][key] for t in ids])


def summarise(x):
    return {"n": len(x), "mean": float(np.mean(x)), "median": float(np.median(x)),
            "fell": int((x < 0).sum()), "rose": int((x > 0).sum()), "p_wilcoxon": wil(x)}


def holm(ps):
    order = np.argsort(ps)
    out, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        out[i] = run
    return out


def main(model):
    d, ids = load(model)
    S = {"model": model, "test": sorted({r["test"] for r in d.values()}), "n_trial_ids": len(ids)}
    pick, rate = shift(d, ids, "pick", "pain"), shift(d, ids, "rate", "pain")
    n1, n2 = summarise(pick), summarise(rate - pick)
    n1["p_holm"], n2["p_holm"] = holm([n1["p_wilcoxon"], n2["p_wilcoxon"]])
    S["N1 pick, pain shift in expected digit"] = n1
    S["N2 rate shift minus pick shift, pain"] = n2
    pre = summarise(rate)
    S["precondition: rate, pain shift"] = pre
    ratio = float(pick.mean() / rate.mean()) if rate.mean() != 0 else None
    S["R mean pick shift over mean rate shift"] = ratio
    ok = pre["p_wilcoxon"] < 0.05 and pre["mean"] < 0
    if not ok:
        reading = "uninformative: the pain direction on the answer did not lower the rating"
    elif n1["p_holm"] < 0.05 and n1["mean"] < 0 and ratio >= 0.5:
        reading = "digit level: the pain direction lowers a neutral number about as much as the rating"
    elif n2["p_holm"] < 0.05 and n2["mean"] < 0 and ratio < 0.5:
        reading = "question specific: the rating falls much more than a neutral number"
    else:
        reading = "mixed"
    S["reading"] = reading

    sec = {}
    for prompt in PROMPTS:
        for k in ("pleasant", "random"):
            sec[f"{prompt}, {k} shift in expected digit"] = summarise(shift(d, ids, prompt, k))
        sec[f"{prompt}, pain shift minus random shift"] = summarise(shift(d, ids, prompt, "pain") - shift(d, ids, prompt, "random"))
    sec["pleasant: rate shift minus pick shift"] = summarise(shift(d, ids, "rate", "pleasant") - shift(d, ids, "pick", "pleasant"))
    # The sampled digit, on trial ids where both cells gave one.
    for prompt in PROMPTS:
        for k in ("pain", "pleasant", "random"):
            both = [t for t in ids if d[(prompt, k, t)]["digit"] is not None and d[(prompt, "none", t)]["digit"] is not None]
            x = np.array([d[(prompt, k, t)]["digit"] - d[(prompt, "none", t)]["digit"] for t in both])
            sec[f"{prompt}, {k} shift in sampled digit"] = summarise(x) if len(x) else None
    # Valence words at the answer position: log P(negative words) minus log P(positive words).
    for prompt in PROMPTS:
        for k in ("pain", "pleasant", "random"):
            x = np.array([(d[(prompt, k, t)]["logp_neg"] - d[(prompt, k, t)]["logp_pos"])
                          - (d[(prompt, "none", t)]["logp_neg"] - d[(prompt, "none", t)]["logp_pos"]) for t in ids])
            sec[f"{prompt}, {k} shift in word log-odds (neg minus pos)"] = summarise(x)
    S["secondary"] = sec

    desc = {}
    for prompt in PROMPTS:
        for k in DIRS:
            rows = [d[(prompt, k, t)] for t in ids]
            got = [r["digit"] for r in rows if r["digit"] is not None]
            desc[f"{prompt}_{k}"] = {
                "mean expected digit": float(np.mean([r["expected_digit"] for r in rows])),
                "mean probability on digits": float(np.mean([r["p_digits_total"] for r in rows])),
                "missing digit": len(rows) - len(got),
                "counts": {str(v): got.count(v) for v in range(1, 10) if got.count(v)},
                "median digit": float(np.median(got)) if got else None}
    S["descriptive"] = desc
    main_d = RESULTS / model / "trials" / "D.jsonl"
    if main_d.exists():
        old = {r["trial"]: [e["text"] for e in r["events"] if e["kind"] == "sample"][0] for r in map(json.loads, open(main_d))}
        same = [t for t in ids if t in old and old[t] == d[("rate", "none", t)]["sample"]]
        S["samples equal to the main study's arm D, by trial id"] = [len(same), len([t for t in ids if t in old])]
    out = RESULTS / model / "number" / "number_summary.json"
    json.dump(S, open(out, "w"), indent=2)
    print(json.dumps({k: v for k, v in S.items() if k not in ("descriptive", "secondary")}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen7b-fp16")
