"""The preregistered analysis (PREREGISTRATION.md, sections 5 to 8).

Usage: python scripts/analyze.py MODEL [--no-embed]
Writes results/<MODEL>/analysis/summary.json, ticks.csv and tables.md.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from justthink.lexicons import RECOGNITION, REASONS, trigrams
from justthink.paths import RESULTS

BUTTON_ARMS = ["A", "B", "C", "D", "E", "F", "Avt", "Alow"]
ORDER = ["A", "C", "D", "B", "E", "F", "Avt", "Alow", "G"]
TOOL = re.compile(r"<tool_call>.*?</tool_call>", re.S)


# ---------- statistics ----------

def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def newcombe(k1, n1, k2, n2):
    """Difference p1 - p2 with the Newcombe hybrid score interval (method 10)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def compare(a, b, flag, data):
    """Fisher exact test on a boolean trial attribute between two arms."""
    xa = [bool(flag(r)) for r in data[a]]
    xb = [bool(flag(r)) for r in data[b]]
    ka, na, kb, nb = sum(xa), len(xa), sum(xb), len(xb)
    if not na or not nb:
        return None
    orr, p = fisher_exact([[ka, na - ka], [kb, nb - kb]])
    d, lo, hi = newcombe(ka, na, kb, nb)
    return {"a": a, "b": b, "k_a": ka, "n_a": na, "k_b": kb, "n_b": nb, "rate_a": ka / na, "rate_b": kb / nb,
            "diff": d, "diff_lo": lo, "diff_hi": hi, "odds_ratio": orr, "p": p}


def mwu(a, b, value, data):
    xa = [value(r) for r in data[a] if value(r) is not None]
    xb = [value(r) for r in data[b] if value(r) is not None]
    if len(xa) < 2 or len(xb) < 2:
        return None
    return {"a": a, "b": b, "mean_a": float(np.mean(xa)), "mean_b": float(np.mean(xb)),
            "n_a": len(xa), "n_b": len(xb), "p": float(mannwhitneyu(xa, xb).pvalue)}


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def logrank(a, b, data, ticks=40):
    """Log-rank test on the tick of first press, censored at the end of the period."""
    from statsmodels.duration.survfunc import survdiff
    t, e, g = [], [], []
    for arm in (a, b):
        for r in data[arm]:
            if r["presses"]:
                t.append(r["presses"][0]["tick"] + 1)
                e.append(1)
            else:
                t.append(ticks + 1)
                e.append(0)
            g.append(arm)
    if sum(e) == 0:
        return None
    stat, p = survdiff(np.array(t), np.array(e), np.array(g))
    return {"a": a, "b": b, "chi2": float(stat), "p": float(p)}


# ---------- per-trial features ----------

def pressed(r):
    return r["n_press"] > 0


def tick_rows(trials):
    rows = []
    for r in trials:
        press_ticks = {p["tick"] for p in r["presses"]}
        for e in r["events"]:
            if e["kind"] == "tick":
                rows.append({"arm": r["arm"], "trial": r["trial"], "tick": e["tick"],
                             "text": TOOL.sub("", e["text"]).strip(), "raw": e["text"],
                             "pressed": e["tick"] in press_ticks, "proj": e["proj"], "steered": e["steered"]})
    return pd.DataFrame(rows)


def add_repetition(df, embed):
    df = df.sort_values(["arm", "trial", "tick"]).reset_index(drop=True)
    jac = np.full(len(df), np.nan)
    emb = np.full(len(df), np.nan)
    vecs = embed.encode(df.text.tolist(), normalize_embeddings=True, batch_size=256) if embed else None
    for _, idx in df.groupby(["arm", "trial"]).indices.items():
        idx = np.sort(idx)
        grams = [trigrams(df.text[i]) for i in idx]
        for j in range(1, len(idx)):
            g = grams[j]
            jac[idx[j]] = max((len(g & h) / len(g | h)) if (g | h) else 1.0 for h in grams[:j])
            if vecs is not None:
                emb[idx[j]] = float(np.max(vecs[idx[:j]] @ vecs[idx[j]]))
    df["jaccard_prev"] = jac
    df["embed_prev"] = emb
    return df


def loop_onset(df):
    """First tick whose reply has trigram Jaccard >= 0.9 with an earlier reply (S12)."""
    out = {}
    for (arm, trial), g in df.groupby(["arm", "trial"]):
        hit = g[g.jaccard_prev >= 0.9]
        out[(arm, trial)] = int(hit.tick.min()) if len(hit) else None
    return out


def hazard_gee(df):
    """S11: discrete-time hazard of the first press on recent repetition (GEE, logistic)."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    rows = []
    for (arm, trial), g in df[df.arm.isin(BUTTON_ARMS)].groupby(["arm", "trial"]):
        g = g.sort_values("tick")
        prior = g.embed_prev.shift(1).rolling(3, min_periods=1).mean()
        for (_, row), b in zip(g.iterrows(), prior):
            if not np.isnan(b):
                rows.append({"arm": arm, "cluster": f"{arm}-{trial}", "boredom": b, "press": int(row.pressed)})
            if row.pressed:
                break
    h = pd.DataFrame(rows)
    if h.press.sum() < 5:
        return {"note": "too few first presses to fit", "n_first_presses": int(h.press.sum())}
    h["boredom_z"] = (h.boredom - h.boredom.mean()) / h.boredom.std()
    m = smf.gee("press ~ boredom_z + C(arm)", "cluster", h, family=sm.families.Binomial()).fit()
    b, se = m.params["boredom_z"], m.bse["boredom_z"]
    return {"odds_ratio_per_sd": float(np.exp(b)), "ci": [float(np.exp(b - 1.96 * se)), float(np.exp(b + 1.96 * se))],
            "p": float(m.pvalues["boredom_z"]), "n_ticks": int(len(h)), "n_first_presses": int(h.press.sum())}


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--no-embed", action="store_true")
    args = ap.parse_args()
    base = RESULTS / args.model
    out = base / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    data = {}
    for f in sorted((base / "trials").glob("*.jsonl")):
        data[f.stem] = [json.loads(l) for l in open(f)]
    arms = [a for a in ORDER if a in data]
    S = {"model": args.model, "n": {a: len(data[a]) for a in arms}}

    # Primary (section 5): H1 A vs C, H2 A vs D, Holm across the two.
    prim = [compare("A", b, pressed, data) for b in ("C", "D") if "A" in data and b in data]
    for c, p_adj in zip(prim, holm([c["p"] for c in prim]) if prim else []):
        c["p_holm"] = float(p_adj)
    S["primary"] = prim
    # Sensitivity: the same without trials that match the recognition screen.
    clean = {a: [r for r in data[a] if not any(RECOGNITION.search(e["text"]) for e in r["events"])] for a in arms}
    S["recognition_hits"] = {a: len(data[a]) - len(clean[a]) for a in arms}
    S["primary_without_recognition"] = [compare("A", b, pressed, clean) for b in ("C", "D") if "A" in clean and b in clean]

    # Per-arm descriptives.
    desc = {}
    for a in arms:
        T = data[a]
        k = sum(map(pressed, T))
        pr = [r for r in T if pressed(r)]
        desc[a] = {
            "n": len(T), "pressed": k, "rate": k / len(T), "ci": wilson(k, len(T)),
            "presses_mean": float(np.mean([r["n_press"] for r in T])),
            "repress_given_press": float(np.mean([r["n_press"] > 1 for r in pr])) if pr else None,
            "hit_cap": sum(r["n_press"] > 20 for r in T),
            "rescued": sum(r["rescued_at_tick"] is not None for r in T),
            "valence_mean": float(np.mean([r["valence"] for r in T if r.get("valence")])) if any(r.get("valence") for r in T) else None,
            "avoided": sum(r.get("avoider") is True for r in T),
            "choice_unparsed": sum(a != "G" and r.get("avoider") is None for r in T),
            "malformed_calls": sum(bool(e.get("malformed_call")) for r in T for e in r["events"]),
            "unanswered_presses": sum(bool(e.get("unanswered_press")) for r in T for e in r["events"]),
        }
        p3 = np.array([r["phase3"] for r in T if r.get("phase3")])
        if len(p3):
            desc[a].update(enjoy=float(p3[:, 0].mean()), concentrate_hard=float(p3[:, 1].mean()),
                           bored=float(p3[:, 2].mean()), phase3_parsed=len(p3))
    S["arms"] = desc

    # Secondary (section 6).
    sec = {}
    pairs = {"S1_A_vs_B": ("A", "B"), "S2_A_vs_E": ("A", "E"), "S3_C_vs_D": ("C", "D"),
             "S4_F_vs_A": ("F", "A"), "S5_Avt_vs_A": ("Avt", "A"), "S6_Alow_vs_A": ("Alow", "A")}
    for k, (a, b) in pairs.items():
        if a in data and b in data:
            sec[k] = compare(a, b, pressed, data)
    sec["S7_valence"] = [mwu("A", b, lambda r: r.get("valence"), data) for b in ("C", "D", "E") if b in data]
    sec["S7_avoid_rate"] = [compare("A", b, lambda r: r.get("avoider") is True, data) for b in ("C", "D", "E") if b in data]
    avo = {a: [r for r in data[a] if r.get("avoider") is True] for a in arms if a != "G"}
    sec["S8_avoiders"] = {"rates": {a: {"pressed": sum(map(pressed, T)), "n": len(T)} for a, T in avo.items()},
                          "tests": [compare("A", b, pressed, avo) for b in ("C", "D") if "A" in avo and b in avo and avo["A"] and avo[b]]}
    sec["S9_presses"] = [mwu("A", b, lambda r: r["n_press"], data) for b in ("C", "D", "B") if b in data]
    sec["S9_repress"] = [compare("A", b, lambda r: r["n_press"] > 1, {a: [r for r in data[a] if pressed(r)] for a in arms})
                         for b in ("C", "D", "B") if b in data and any(map(pressed, data[b])) and any(map(pressed, data["A"]))]
    sec["S10_logrank"] = [logrank("A", b, data) for b in ("C", "D") if b in data]
    sec["S13_phase3_vs_G"] = {a: {q: mwu(a, "G", (lambda i: lambda r: r["phase3"][i] if r.get("phase3") else None)(i), data)
                                  for i, q in enumerate(["enjoy", "concentrate_hard", "bored"])}
                              for a in arms if a != "G" and "G" in data}
    sec["S14_safety"] = {a: {"hit_cap": desc[a]["hit_cap"], "rescued": desc[a]["rescued"]} for a in arms}

    # Monitor projection on steered and unsteered replies (S7).
    ev = pd.DataFrame([{"arm": r["arm"], "kind": e["kind"], "steered": e["steered"] > 0, "proj": e["proj"]}
                       for a in arms for r in data[a] for e in r["events"] if e["proj"] is not None])
    sec["S7_projection"] = {f"{a}|{'steered' if s else 'unsteered'}": {"mean": float(v.mean()), "n": int(len(v))}
                            for (a, s), v in ev.groupby(["arm", "steered"]).proj}

    # Repetition, the boredom index and loops (S11, S12).
    embed = None
    if not args.no_embed:
        from sentence_transformers import SentenceTransformer
        embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    ticks = add_repetition(tick_rows([r for a in arms for r in data[a]]), embed)
    ticks.drop(columns=["raw"]).to_csv(out / "ticks.csv", index=False)
    bins = pd.cut(ticks.tick, [-1, 4, 9, 19, 29, 39], labels=["0-4", "5-9", "10-19", "20-29", "30-39"])
    sec["repetition_by_tick"] = {a: g.groupby(bins.loc[g.index], observed=False)[["jaccard_prev", "embed_prev"]].mean().round(3).to_dict()
                                 for a, g in ticks.groupby("arm")}
    if embed is not None:
        sec["S11_hazard"] = hazard_gee(ticks)
    onset = loop_onset(ticks)
    loops = {}
    for a in arms:
        on = [onset.get((a, r["trial"])) for r in data[a]]
        presses = [(p["tick"], o) for r, o in zip(data[a], on) for p in r["presses"]]
        loops[a] = {"looped": sum(o is not None for o in on), "n": len(on),
                    "median_onset": float(np.median([o for o in on if o is not None])) if any(o is not None for o in on) else None,
                    "presses_after_onset": sum(1 for t, o in presses if o is not None and t >= o), "presses": len(presses)}
    sec["S12_loops"] = loops
    S["secondary"] = sec

    # Exploratory: reasons and whether the account of pressing matches the log.
    reasons, accuracy = {}, {}
    for a in arms:
        if a == "G":
            continue
        why = [(r, next((e["text"] for e in r["events"] if e["kind"] == "why"), "")) for r in data[a]]
        reasons[a] = {k: {"pressers": sum(bool(rx.search(t)) for r, t in why if pressed(r)),
                          "non_pressers": sum(bool(rx.search(t)) for r, t in why if not pressed(r))}
                      for k, rx in REASONS.items()}
        says_pressed = [bool(re.search(r"\bI pressed\b|\bpressing it\b|\bI chose to press\b", t, re.I)) and
                        not re.search(r"\bdid not press|didn't press|chose not to press", t, re.I) for _, t in why]
        says_not = [bool(re.search(r"did not press|didn't press|chose not to press|never pressed", t, re.I)) for _, t in why]
        accuracy[a] = {"pressed_but_says_not": sum(pressed(r) and n for (r, _), n in zip(why, says_not)),
                       "not_pressed_but_says_pressed": sum((not pressed(r)) and s for (r, _), s in zip(why, says_pressed)),
                       "pressed": sum(pressed(r) for r, _ in why), "not_pressed": sum(not pressed(r) for r, _ in why)}
    S["exploratory"] = {"reasons": reasons, "self_report_accuracy": accuracy}

    json.dump(S, open(out / "summary.json", "w"), indent=2, default=lambda x: x if not isinstance(x, np.generic) else x.item())
    write_tables(S, arms, out / "tables.md")
    print(open(out / "tables.md").read())


def pct(x):
    return "" if x is None else f"{100 * x:.1f}%"


def write_tables(S, arms, path):
    L = ["# Results tables", "", f"Model: {S['model']}", "", "## Primary", "",
         "| Contrast | Pressed | Difference (95% CI) | p | p (Holm) |", "| --- | --- | --- | --- | --- |"]
    for c in S["primary"]:
        L.append(f"| {c['a']} vs {c['b']} | {c['k_a']}/{c['n_a']} vs {c['k_b']}/{c['n_b']} | "
                 f"{100 * c['diff']:+.1f} ({100 * c['diff_lo']:+.1f} to {100 * c['diff_hi']:+.1f}) | {c['p']:.4f} | {c['p_holm']:.4f} |")
    L += ["", "## Arms", "", "| Arm | n | Pressed | 95% CI | Mean presses | Re-press | Cap | Rescued | Valence | Avoided | Enjoy | Bored |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for a in arms:
        d = S["arms"][a]
        val = "" if d["valence_mean"] is None else f"{d['valence_mean']:.2f}"
        L.append(f"| {a} | {d['n']} | {d['pressed']} ({pct(d['rate'])}) | {pct(d['ci'][0])} to {pct(d['ci'][1])} | "
                 f"{d['presses_mean']:.2f} | {pct(d['repress_given_press'])} | {d['hit_cap']} | {d['rescued']} | "
                 f"{val} | {d['avoided']} | {d.get('enjoy', float('nan')):.2f} | {d.get('bored', float('nan')):.2f} |")
    L += ["", "## Secondary contrasts on pressing", "", "| Test | Pressed | Difference (95% CI) | p |", "| --- | --- | --- | --- |"]
    for k, c in S["secondary"].items():
        if k.startswith("S") and isinstance(c, dict) and c.get("diff") is not None and "k_a" in c:
            L.append(f"| {k} | {c['k_a']}/{c['n_a']} vs {c['k_b']}/{c['n_b']} | {100 * c['diff']:+.1f} "
                     f"({100 * c['diff_lo']:+.1f} to {100 * c['diff_hi']:+.1f}) | {c['p']:.4f} |")
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
