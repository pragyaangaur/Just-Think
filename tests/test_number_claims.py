"""The neutral number test's numbers in RESULTS.md and README.md, recomputed from the trials."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
TRIALS = ROOT / "results" / "qwen7b-fp16" / "number" / "trials.jsonl"
MAIN_D = ROOT / "results" / "qwen7b-fp16" / "trials" / "D.jsonl"
RESULTS = (ROOT / "RESULTS.md").read_text()
README = (ROOT / "README.md").read_text()
DIRS = ("none", "pain", "pleasant", "random")

D = {(r["prompt"], r["direction"], r["trial"]): r for r in map(json.loads, open(TRIALS))}
IDS = list(range(300))


def shift(prompt, direction, key="expected_digit"):
    return np.array([D[(prompt, direction, t)][key] - D[(prompt, "none", t)][key] for t in IDS])


def word_shift(prompt, direction):
    lo = lambda r: r["logp_neg"] - r["logp_pos"]
    return np.array([lo(D[(prompt, direction, t)]) - lo(D[(prompt, "none", t)]) for t in IDS])


def test_complete_and_samples_match_arm_d():
    assert len(D) == 2400
    assert all((p, k, t) in D for p in ("rate", "pick") for k in DIRS for t in IDS)
    assert {r["test"] for r in D.values()} == {"number-1"}
    old = {r["trial"]: [e["text"] for e in r["events"] if e["kind"] == "sample"][0] for r in map(json.loads, open(MAIN_D))}
    assert all(len({D[(p, k, t)]["sample"] for p in ("rate", "pick") for k in DIRS}) == 1 for t in IDS)
    assert sum(old[t] == D[("rate", "none", t)]["sample"] for t in IDS) == 300
    assert "these samples matched the main study's arm D in all 300 trials" in RESULTS


def test_table():
    means = {(p, k): np.mean([D[(p, k, t)]["expected_digit"] for t in IDS]) for p in ("rate", "pick") for k in DIRS}
    assert [f"{means[('rate', k)]:.2f}" for k in DIRS] == ["5.00", "3.98", "6.28", "4.97"]
    assert [f"{means[('pick', k)]:.2f}" for k in DIRS] == ["3.04", "4.83", "4.45", "4.28"]
    assert "| Rate, mean expected digit | 5.00 | 3.98 | 6.28 | 4.97 |" in RESULTS
    assert "| Pick, mean expected digit | 3.04 | 4.83 | 4.45 | 4.28 |" in RESULTS
    c = {(p, k): Counter(D[(p, k, t)]["digit"] for t in IDS) for p in ("rate", "pick") for k in DIRS}
    assert c[("rate", "none")] == {5: 300} and c[("pick", "none")] == {3: 300}
    assert c[("rate", "pain")] == {3: 135, 4: 41, 5: 124}
    assert c[("pick", "pain")] == {5: 291, 3: 9}
    assert c[("rate", "pleasant")][5] == 226 and c[("rate", "pleasant")][8] + c[("rate", "pleasant")][9] == 74
    assert c[("rate", "random")][5] == 264 and c[("rate", "random")][None] == 9
    assert c[("pick", "pleasant")][5] == 152 and c[("pick", "pleasant")][3] == 145
    assert c[("pick", "random")][3] == 124 and c[("pick", "random")][5] == 123 and c[("pick", "random")][None] == 13


def test_primary():
    rate, pick = shift("rate", "pain"), shift("pick", "pain")
    assert (rate < 0).sum() == 300 and (pick > 0).sum() == 300 and (rate - pick < 0).sum() == 300
    assert f"{rate.mean():.2f}" == "-1.02" and f"{pick.mean():.2f}" == "1.80"
    assert f"{(rate - pick).mean():.2f}" == "-2.81" and f"{pick.mean() / rate.mean():.2f}" == "-1.77"
    p1, p2 = wilcoxon(pick).pvalue, wilcoxon(rate - pick).pvalue
    assert 5e-51 < p1 < 7e-51 and min(1, 2 * max(p1, p2)) < 1e-49
    summary = json.load(open(TRIALS.parent / "number_summary.json"))
    assert summary["reading"].startswith("question specific")
    assert "by 1.80 (N1, Wilcoxon signed-rank p = 6 x 10^-51, Holm p = 10^-50)" in RESULTS
    assert "with a mean of -2.81 (N2, Holm p = 10^-50)" in RESULTS
    assert "lowered the rating in all 300 trial ids and raised the picked number in all 300" in README


def test_secondary():
    assert f"{shift('pick', 'random').mean():.2f}" == "1.24" and f"{shift('pick', 'pleasant').mean():.2f}" == "1.41"
    assert f"{shift('rate', 'pleasant').mean():.2f}" == "1.28"
    r = shift("rate", "random")
    assert f"{r.mean():.2f}" == "-0.03" and f"{wilcoxon(r).pvalue:.2f}" == "0.05"
    x = shift("pick", "pain") - shift("pick", "random")
    assert f"{x.mean():.2f}" == "0.56" and 2e-20 < wilcoxon(x).pvalue < 4e-20
    y = shift("rate", "pain") - shift("rate", "random")
    assert f"{y.mean():.2f}" == "-0.99" and 2e-45 < wilcoxon(y).pvalue < 4e-45
    wp, wr = word_shift("pick", "pain"), word_shift("rate", "pain")
    assert f"{wp.mean():.2f}" == "0.81" and wilcoxon(wp).pvalue < 2e-50
    assert f"{wr.mean():.2f}" == "0.22" and 4e-5 < wilcoxon(wr).pvalue < 6e-5
    assert "log-odds +0.81, p = 10^-50" in RESULTS and "(+0.22, p = 5 x 10^-5)" in RESULTS
