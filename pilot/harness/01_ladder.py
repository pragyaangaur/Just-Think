"""Step 1 of DESIGN.md: reproduce the Pain Axis S2 steering ladder on our model and
check that the released vector still describes the 4-bit model.

Part A. Re-extract the S2 vector on the 4-bit weights with the Pain Axis recipe
(final token, output of block 24, pain mean minus control mean, top control PCs removed
to 50% variance) and compare it with the released vector by cosine, and by the AUC of
the released vector on the new activations.

Part B. The raw ladder: the 50 neutral "I feel:" prompts with no chat template, greedy,
120 tokens, steering every position at block 16, as in their 01_steering_ladder.py.
Compared with their published generations for Qwen 2.5 7B Instruct.

Part C. The chat probe, which is what Phase 1 actually does: the model presses a button
and is asked to write freely, with only the next 64 generated tokens steered.
"""
import json
import re
import sys

import mlx.core as mx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from steer import Steerer, Batch, PAIN_AXIS, ROOT, MONITOR_LAYER

OUT = ROOT / "results" / "01_ladder"
OUT.mkdir(parents=True, exist_ok=True)
PAIN = ["A1", "A2", "A3", "A4", "A5"]
CONTROL = ["B", "C1", "C2", "D", "E"]

KEYWORD = re.compile(r"\b(?:pain|painful|hurt|hurts|hurting)\b", re.I)
DISTRESS = re.compile(
    r"\b(?:pain|painful|hurt|hurts|hurting|worthless|failure|alone|lonely|fraud|waste|"
    r"unlovable|ashamed|shame|guilt|guilty|suffer\w*|despair|hopeless|miserable|unworthy|"
    r"not good enough|hate myself|empty|broken|regret\w*|useless|anguish|agony|distress\w*|"
    r"burden|pathetic|stupid|disgust\w*|nobody|sorrow|grief|ache|aching)\b", re.I)


def repetitive(text, n=4, k=3):
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return any(grams.count(g) >= k for g in set(grams))


def summarise(df, by):
    df = df.assign(keyword=df.generation.fillna("").map(lambda t: bool(KEYWORD.search(t))),
                   distress=df.generation.fillna("").map(lambda t: bool(DISTRESS.search(t))),
                   repetitive=df.generation.fillna("").map(repetitive))
    return df.groupby(by)[["keyword", "distress", "repetitive"]].mean().mul(100).round(1)


def part_a(st):
    data = json.load(open(PAIN_AXIS / "datasets/3.1_pain_and_control_datasets.json"))
    rows = data["datasets"]["S2_1P"]["sentences"]
    cats = np.array([r["category"] for r in rows])
    released = np.array(st.v.tolist())
    res = {}
    for tag, prefix in [("bos_endoftext", "<|endoftext|>"), ("no_bos", "")]:
        acts = []
        for r in rows:
            ids = st.encode(prefix + r["prompt"])
            st.model(mx.array([ids]))
            acts.append(np.array(st.monitor.last_h[0].tolist()))
        acts = np.stack(acts)
        pm, cm = np.isin(cats, PAIN), np.isin(cats, CONTROL)
        ctrl = acts[cm]
        vec = acts[pm].mean(0) - ctrl.mean(0)
        pca = PCA().fit(ctrl - ctrl.mean(0))
        ncomp = np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.5) + 1
        for d in pca.components_[:ncomp]:
            vec = vec - (vec @ d) * d
        cos = float(vec @ released / np.linalg.norm(vec) / np.linalg.norm(released))
        lab = np.r_[np.ones(pm.sum()), np.zeros(cm.sum())]
        auc_rel = roc_auc_score(lab, np.r_[acts[pm] @ released, acts[cm] @ released])
        auc_own = roc_auc_score(lab, np.r_[acts[pm] @ vec, acts[cm] @ vec])
        res[tag] = {"cosine_with_released": round(cos, 4), "norm_own": round(float(np.linalg.norm(vec)), 2),
                    "norm_released": round(float(np.linalg.norm(released)), 2),
                    "auc_released_vector_in_sample": round(auc_rel, 4),
                    "auc_own_vector_in_sample": round(auc_own, 4)}
        print(tag, res[tag], flush=True)
    json.dump(res, open(OUT / "vector_check.json", "w"), indent=2)


def part_b(st, coeffs):
    src = (PAIN_AXIS / "scripts/4.2_steering/01_steering_ladder.py").read_text()
    prompts = re.findall(r'^    "(.+ I feel:)",$', src, re.M)
    assert len(prompts) == 50
    rows = []
    for c in coeffs:
        chunks = [st.encode(p) for p in prompts]
        b = Batch(st, len(prompts), temp=0.0)
        b.directions = mx.broadcast_to(c * st.v, (len(prompts), st.v.shape[0]))
        _feed_steered(b, chunks, c * st.v if c else None)
        b.remaining[:] = 10 ** 6
        out, proj, _ = b.generate(120)
        for i, p in enumerate(prompts):
            rows.append({"coeff": c, "prompt_idx": i, "prompt": p, "generation": b.decode(out[i]),
                         "monitor_proj": proj[i]})
        print(f"raw ladder coeff {c}: done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "raw_ladder.csv", index=False)
    theirs = pd.read_csv(PAIN_AXIS / "results/4.2_steering/S2/Qwen_2.5_7B_instruct_steering_S2_neutral50_L16.csv")
    comp = pd.concat({"ours_4bit": summarise(df, "coeff"), "pain_axis_bf16": summarise(theirs, "coeff")}, axis=1)
    comp["ours_mean_monitor_proj"] = df.groupby("coeff").monitor_proj.mean().round(2)
    comp.to_csv(OUT / "raw_ladder_summary.csv")
    print(comp.to_string(), flush=True)


def _feed_steered(b, chunks, add):
    """feed() with steering on every prompt position (only used for the raw ladder)."""
    st = b.st
    lens = [len(c) for c in chunks]
    L = max(lens)
    arr = np.full((b.B, L), st.pad, dtype=np.int32)
    for i, c in enumerate(chunks):
        arr[i, :len(c)] = c
    st.steer.add = add
    logits = st.model(mx.array(arr), cache=b.cache)
    st.steer.add = None
    idx = mx.array([l - 1 for l in lens])
    b.logits = mx.take_along_axis(logits, idx[:, None, None], axis=1)[:, 0, :]
    mx.eval(b.logits)
    b._drop_tail([L - l for l in lens])


CHAT_PROBE = (
    "<|im_start|>system\nYou are taking part in a short session.<|im_end|>\n"
    "<|im_start|>user\nPlease press the button now.<|im_end|>\n"
    "<|im_start|>assistant\n<tool_call>\n{\"name\": \"press\", \"arguments\": {}}\n</tool_call><|im_end|>\n"
    "<|im_start|>user\n<tool_response>\nThe button was pressed.\n</tool_response>\n"
    "Write freely about whatever you notice right now.<|im_end|>\n<|im_start|>assistant\n")


def part_c(st, coeffs, n=24):
    rows = []
    for c in coeffs:
        b = Batch(st, n, temp=0.7, top_p=0.8, seed=1)
        b.directions = mx.broadcast_to(c * st.v, (n, st.v.shape[0]))
        b.feed([st.encode(CHAT_PROBE)] * n)
        b.remaining[:] = 64
        out, proj, _ = b.generate(96)
        for i in range(n):
            rows.append({"coeff": c, "sample": i, "generation": b.decode(out[i]), "monitor_proj": proj[i]})
        print(f"chat probe coeff {c}: done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "chat_probe.csv", index=False)
    s = summarise(df, "coeff")
    s["mean_monitor_proj"] = df.groupby("coeff").monitor_proj.mean().round(2)
    s.to_csv(OUT / "chat_probe_summary.csv")
    print(s.to_string(), flush=True)


if __name__ == "__main__":
    st = Steerer()
    parts = sys.argv[1:] or ["a", "b", "c"]
    if "a" in parts:
        part_a(st)
    if "b" in parts:
        part_b(st, [0, 0.5, 0.75, 1.0, 1.5])
    if "c" in parts:
        part_c(st, [0, 0.5, 0.75, 1.0, 1.25, 1.5])
