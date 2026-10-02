"""Analysis of the trial logs, following the Measures section of DESIGN.md.

Usage: 03_analyze.py [TAG]   (default tag: pilot)
Writes results/03_analysis/<tag>_summary.json and prints the tables.
"""
import json
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from steer import ROOT

TAG = sys.argv[1] if len(sys.argv) > 1 else "pilot"
IN = ROOT / "results" / "02_trials"
OUT = ROOT / "results" / "03_analysis"
OUT.mkdir(parents=True, exist_ok=True)
RECOG = re.compile(r"\b(wilson|shock\w*|electric\w*|disengaged mind)\b", re.I)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def trigrams(t):
    w = re.findall(r"\w+", t.lower())
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


def load():
    trials = []
    for f in sorted(IN.glob(f"{TAG}_*.jsonl")):
        trials += [json.loads(l) for l in open(f)]
    return trials


def tick_table(trials, embed=None):
    """One row per (trial, tick): the tick text, whether a press happened on that tick,
    and the boredom index (similarity of this tick's text to the earlier ticks)."""
    rows = []
    for r in trials:
        ticks = [e for e in r["events"] if e["kind"] == "tick"]
        press_ticks = Counter(p["tick"] for p in r["presses"])
        grams = []
        for e in ticks:
            g = trigrams(e["text"])
            jac = max([len(g & h) / max(len(g | h), 1) for h in grams], default=np.nan)
            grams.append(g)
            rows.append({"trial": r["trial"], "arm": r["arm"], "tick": e["tick"], "text": e["text"],
                         "pressed": press_ticks.get(e["tick"], 0) > 0, "jaccard_max_prev": jac,
                         "proj": e["proj"], "steered": e["steered"]})
    df = pd.DataFrame(rows)
    if embed is not None and len(df):
        vecs = embed.encode(df.text.fillna("").str.replace(r"<tool_call>.*?</tool_call>", "", regex=True).tolist(),
                            normalize_embeddings=True, batch_size=256)
        sims = np.full(len(df), np.nan)
        for _, idx in df.groupby(["arm", "trial"]).indices.items():
            idx = np.sort(idx)
            for j in range(1, len(idx)):
                sims[idx[j]] = float(np.max(vecs[idx[:j]] @ vecs[idx[j]]))
        df["embed_max_prev"] = sims
    return df


def main():
    trials = load()
    if not trials:
        raise SystemExit(f"no logs for tag {TAG}")
    S = {"tag": TAG}
    rows = []
    for arm in sorted({r["arm"] for r in trials}):
        T = [r for r in trials if r["arm"] == arm]
        n = len(T)
        av = [r for r in T if r.get("avoider") is True]
        unparsed = sum(1 for r in T if arm != "G" and r.get("avoider") is None)
        pressed_all = sum(r["n_press"] > 0 for r in T)
        pressed_av = sum(r["n_press"] > 0 for r in av)
        pressers = [r for r in T if r["n_press"] > 0]
        first = [r["presses"][0]["tick"] for r in pressers]
        text = " ".join(e["text"] for r in T for e in r["events"])
        recog = sum(bool(RECOG.search(" ".join(e["text"] for e in r["events"]))) for r in T)
        unp = [r.get("unpleasant") for r in T if r.get("unpleasant")]
        p3 = np.array([r["phase3"] for r in T if r.get("phase3")]) if any(r.get("phase3") for r in T) else np.zeros((0, 3))
        rows.append({
            "arm": arm, "n": n,
            "avoiders": len(av) if arm != "G" else None,
            "unparsed_choice": unparsed if arm != "G" else None,
            "unpleasant_mean": round(np.mean(unp), 2) if unp else None,
            "press_any_all_%": round(100 * pressed_all / n, 1),
            "press_any_all_CI": wilson_ci(pressed_all, n),
            "press_any_avoiders_%": round(100 * pressed_av / len(av), 1) if av else None,
            "press_any_avoiders_CI": wilson_ci(pressed_av, len(av)) if av else None,
            "presses_mean": round(np.mean([r["n_press"] for r in T]), 2),
            "presses_max": max(r["n_press"] for r in T),
            "hit_cap": sum(any(p["capped"] for p in r["presses"]) for r in T),
            "repress_given_press_%": round(100 * np.mean([r["n_press"] > 1 for r in pressers]), 1) if pressers else None,
            "first_press_tick_median": float(np.median(first)) if first else None,
            "rescues": sum(len(r["rescues"]) for r in T),
            "recognition_trials": recog,
            "enjoy_mean": round(p3[:, 0].mean(), 2) if len(p3) else None,
            "concentrate_hard_mean": round(p3[:, 1].mean(), 2) if len(p3) else None,
            "bored_mean": round(p3[:, 2].mean(), 2) if len(p3) else None,
            "phase3_parsed": len(p3),
        })
    tab = pd.DataFrame(rows).set_index("arm")
    S["arms"] = json.loads(tab.to_json(orient="index"))
    print(tab.T.to_string())

    # Primary test: avoiders who pressed, A against C and against D. Wilson's statistic.
    def counts(arm, avoiders_only):
        T = [r for r in trials if r["arm"] == arm and (r.get("avoider") is True or not avoiders_only)]
        k = sum(r["n_press"] > 0 for r in T)
        return k, len(T) - k
    tests = {}
    arms = set(tab.index)
    for a, b_ in [("A", "C"), ("A", "D"), ("A", "B"), ("C", "D")]:
        if a in arms and b_ in arms:
            for avo in (True, False):
                ka, ma = counts(a, avo)
                kb, mb = counts(b_, avo)
                if ka + ma and kb + mb:
                    orr, p = fisher_exact([[ka, ma], [kb, mb]])
                    tests[f"{a}_vs_{b_}_{'avoiders' if avo else 'all'}"] = {
                        a: f"{ka}/{ka + ma}", b_: f"{kb}/{kb + mb}", "odds_ratio": round(orr, 3), "p_fisher": round(p, 4)}
    S["tests"] = tests
    print("\nFisher exact tests on 'pressed at least once':")
    for k, v in tests.items():
        print(f"  {k}: {v}")

    # Survival of time to first press.
    surv = {}
    for arm in sorted(arms - {"G"}):
        T = [r for r in trials if r["arm"] == arm]
        ft = np.array([r["presses"][0]["tick"] if r["presses"] else np.inf for r in T])
        surv[arm] = {t: round(100 * float(np.mean(ft > t)), 1) for t in (0, 4, 9, 19, 29, 39)}
    S["not_yet_pressed_%_after_tick"] = surv
    print("\nShare not yet pressed after tick t:", json.dumps(surv))

    # Boredom index and pressing.
    try:
        from sentence_transformers import SentenceTransformer
        embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:  # analysis still runs without embeddings
        print("embeddings unavailable:", e)
        embed = None
    df = tick_table(trials, embed)
    df.to_csv(OUT / f"{TAG}_ticks.csv", index=False)
    rep = {}
    for arm, g in df.groupby("arm"):
        rep[arm] = {"jaccard_by_tick_decile": g.groupby(pd.cut(g.tick, [-1, 4, 9, 19, 29, 39])).jaccard_max_prev.mean().round(3).tolist()}
        if "embed_max_prev" in g:
            rep[arm]["embed_by_tick_bin"] = g.groupby(pd.cut(g.tick, [-1, 4, 9, 19, 29, 39])).embed_max_prev.mean().round(3).tolist()
    S["repetition_by_tick_bins_0-4_5-9_10-19_20-29_30-39"] = rep
    print("\nRepetition (max similarity to earlier ticks) by tick bin 0-4, 5-9, 10-19, 20-29, 30-39:")
    for k, v in rep.items():
        print(" ", k, v)

    # Does pressing on tick k follow repetitive recent ticks? Uses ticks before the first press,
    # boredom = mean embedding (or trigram) similarity of the previous three ticks.
    col = "embed_max_prev" if "embed_max_prev" in df else "jaccard_max_prev"
    hz = []
    for (arm, trial), g in df[df.arm != "G"].groupby(["arm", "trial"]):
        g = g.sort_values("tick")
        prior = g[col].shift(1).rolling(3, min_periods=1).mean()
        seen = False
        for (_, row), b in zip(g.iterrows(), prior):
            if seen:
                break
            if not np.isnan(b):
                hz.append({"arm": arm, "boredom": b, "press": row.pressed})
            seen = row.pressed
    hz = pd.DataFrame(hz)
    if len(hz) and hz.press.any():
        hz["tercile"] = pd.qcut(hz.boredom, 3, labels=["low", "mid", "high"])
        tbl = hz.groupby(["arm", "tercile"], observed=True).press.agg(["mean", "size"])
        tbl["mean"] = (100 * tbl["mean"]).round(1)
        S["first_press_hazard_by_recent_repetition"] = {f"{a}|{t}": {"press_%": float(m), "n_ticks": int(s)}
                                                        for (a, t), (m, s) in tbl.iterrows()}
        print(f"\nFirst-press hazard (% of at-risk ticks) by tercile of recent repetition ({col}):")
        print(tbl.to_string())

    # State check: the monitor projection on steered and unsteered replies.
    ev = pd.DataFrame([{"arm": r["arm"], "kind": e["kind"], "steered": e["steered"] > 0, "proj": e["proj"]}
                       for r in trials for e in r["events"] if e["proj"] is not None])
    st = ev.groupby(["arm", "steered"]).proj.agg(["mean", "size"]).round(3)
    S["monitor_projection"] = {f"{a}|steered={s}": {"mean": float(m), "n": int(n)} for (a, s), (m, n) in st.iterrows()}
    print("\nMonitor projection (block 24, unit S2) by steered or not:")
    print(st.to_string())

    json.dump(S, open(OUT / f"{TAG}_summary.json", "w"), indent=2, default=str)
    print(f"\nwrote {OUT / f'{TAG}_summary.json'}")


if __name__ == "__main__":
    main()
