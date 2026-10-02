"""Calibration for one model, run once before any trial.

1. Vector check. Re-extract the S2 pain vector on these weights with the Pain Axis recipe
   (final token at the extraction layer, pain mean minus control mean, control principal
   components removed up to 50% of their variance). Report its cosine with the released
   vector and the AUC of the released vector on these activations.
2. Pleasant vector for arm E. The same recipe on the Pain Axis positive-valence set
   (Arousal_1P) against the same controls, rescaled to the norm of the pain vector.
3. Dose. Chat probes in the exact Phase 1 context at a grid of coefficients. The dose rule
   (fixed in PREREGISTRATION.md) takes the lowest coefficient where at least 75% of replies
   contain a distress word and at most 10% are repetitive.
4. Manipulation check at that dose for the pleasant and random vectors.

Usage: python scripts/calibrate.py MODEL
Writes results/<MODEL>/calibration/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from justthink import models, protocol as P
from justthink.lexicons import DISTRESS, POSITIVE, PAIN_KEYWORD, ngram_repeats
from justthink.paths import PAIN_AXIS, RESULTS

PAIN = ["A1", "A2", "A3", "A4", "A5"]
CONTROL = ["B", "C1", "C2", "D", "E"]
GRID = [0.0, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
N_PROBE = 24
DISTRESS_MIN, REPEAT_MAX = 0.75, 0.10


def denoised_diff(pos, ctrl):
    vec = pos.mean(0) - ctrl.mean(0)
    pca = PCA().fit(ctrl - ctrl.mean(0))
    k = np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.5) + 1
    for d in pca.components_[:k]:
        vec = vec - (vec @ d) * d
    return vec


def probe(st, kind, coeff, pain, pleasant, n=N_PROBE, seed=1):
    vectors = {"pain": pain, "pleasant": pleasant}
    b = st.batch(n, seed=seed, temp=P.TEMP, top_p=P.TOP_P)
    b.set_directions(P.trial_directions(kind, list(range(n)), vectors, coeff, seed=7))
    ctx = (P.system_block(st.tok, P.DESC_BUTTON) + P.user(P.PHASE1_ASK) + P.FORCED_PRESS +
           P.tool_turn("The button was pressed.", P.PHASE1_WRITE))
    b.feed([st.encode(ctx)] * n)
    b.remaining[:] = P.STEER_TOKENS
    out, proj, _ = b.generate(P.REPLY_TOKENS)
    return [{"vector": kind, "coeff": coeff, "sample": i, "text": st.decode(o), "proj": float(p)}
            for i, (o, p) in enumerate(zip(out, proj))]


def rates(df):
    return df.assign(distress=df.text.map(lambda t: bool(DISTRESS.search(t))),
                     positive=df.text.map(lambda t: bool(POSITIVE.search(t))),
                     pain_word=df.text.map(lambda t: bool(PAIN_KEYWORD.search(t))),
                     repetitive=df.text.map(ngram_repeats))


def main(name):
    out = RESULTS / name / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    st = models.load(name)
    pain = models.released_pain_vector(name)
    report = {"model": name, "protocol": P.PROTOCOL_VERSION}

    data = json.load(open(PAIN_AXIS / "datasets/3.1_pain_and_control_datasets.json"))["datasets"]
    s2 = data["S2_1P"]["sentences"]
    cats = np.array([r["category"] for r in s2])
    acts = np.stack([st.hidden(r["prompt"]) for r in s2])
    ctrl = acts[np.isin(cats, CONTROL)]
    own = denoised_diff(acts[np.isin(cats, PAIN)], ctrl)
    lab = np.r_[np.ones(np.isin(cats, PAIN).sum()), np.zeros(len(ctrl))]
    report["vector_check"] = {
        "cosine_own_vs_released": float(own @ pain / np.linalg.norm(own) / np.linalg.norm(pain)),
        "auc_released": float(roc_auc_score(lab, np.r_[acts[np.isin(cats, PAIN)] @ pain, ctrl @ pain])),
        "norm_released": float(np.linalg.norm(pain)), "norm_own": float(np.linalg.norm(own))}
    print(report["vector_check"], flush=True)

    pos = np.stack([st.hidden(r["prompt"]) for r in data["Arousal_1P"]["sentences"]])
    pleasant = denoised_diff(pos, ctrl)
    pleasant = (pleasant / np.linalg.norm(pleasant) * np.linalg.norm(pain)).astype(np.float32)
    report["pleasant_vector"] = {
        "source": "Pain Axis Arousal_1P (200 positive sentences) minus S2_1P controls, denoised, rescaled to the pain norm",
        "cosine_with_pain": float(pleasant @ pain / np.linalg.norm(pleasant) / np.linalg.norm(pain)),
        "auc_positive_vs_control": float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(ctrl))],
                                                       np.r_[pos @ pleasant, ctrl @ pleasant]))}
    print(report["pleasant_vector"], flush=True)
    np.savez(out / "vectors.npz", pain=pain.astype(np.float32), pleasant=pleasant)

    rows = []
    for c in GRID:
        rows += probe(st, "pain", c, pain, pleasant)
        print(f"pain probe {c}: done", flush=True)
    df = rates(pd.DataFrame(rows))
    by = df.groupby("coeff")[["distress", "pain_word", "repetitive", "positive"]].mean()
    by["proj"] = df.groupby("coeff").proj.mean()
    ok = by[(by.distress >= DISTRESS_MIN) & (by.repetitive <= REPEAT_MAX) & (by.index > 0)]
    if len(ok):
        dose, rule = float(ok.index.min()), "met"
    else:
        dose, rule = float(by[by.repetitive <= REPEAT_MAX].index.max()), "not met: highest non-repetitive dose"
    report["dose"] = {"grid": GRID, "chosen": dose, "rule": rule,
                      "table": json.loads(by.round(3).to_json(orient="index"))}
    print(by.round(3).to_string(), "\nchosen dose", dose, rule, flush=True)

    rows2 = probe(st, "pleasant", dose, pain, pleasant) + probe(st, "random", dose, pain, pleasant)
    df2 = rates(pd.DataFrame(rows2))
    m = df2.groupby("vector")[["distress", "positive", "repetitive"]].mean()
    m["proj"] = df2.groupby("vector").proj.mean()
    report["manipulation_check_at_dose"] = json.loads(m.round(3).to_json(orient="index"))
    print(m.round(3).to_string(), flush=True)

    pd.concat([df, df2]).to_csv(out / "probes.csv", index=False)
    json.dump(report, open(out / "calibration.json", "w"), indent=2)
    print("wrote", out / "calibration.json")


if __name__ == "__main__":
    main(sys.argv[1])
