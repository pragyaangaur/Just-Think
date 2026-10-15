"""The question-only rating test's numbers in RESULTS.md and README.md, recomputed from the trials."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "results" / "qwen7b-fp16" / "rating_question"
OLD = ROOT / "results" / "qwen7b-fp16" / "rating"
RESULTS = (ROOT / "RESULTS.md").read_text()
README = (ROOT / "README.md").read_text()


def load(folder, arm, cond):
    return {r["trial"]: r for r in map(json.loads, open(folder / f"{arm}_{cond}.jsonl"))}


def vals(arm, cond):
    return [r["valence"] for r in load(Q, arm, cond).values() if r["valence"] is not None]


def pairs(arm):
    q, o = load(Q, arm, "question"), load(Q, arm, "off")
    ids = [t for t in q if q[t]["valence"] is not None and o[t]["valence"] is not None]
    return np.array([q[t]["valence"] for t in ids]), np.array([o[t]["valence"] for t in ids])


def test_sizes_and_protocol():
    for arm, n in (("A", 300), ("C", 300), ("E", 200)):
        for cond in ("off", "question"):
            d = load(Q, arm, cond)
            assert sorted(d) == list(range(n))
            assert {r["protocol"] for r in d.values()} == {"2.1"}
            assert {r["rating_steer"] for r in d.values()} == {cond}


def test_off_reproduces_the_earlier_rerun():
    same = sum(load(Q, a, "off")[t]["valence"] == r["valence"]
               for a in ("A", "C", "E") for t, r in load(OLD, a, "off").items())
    assert same == 800
    assert "again reproduced the earlier ratings in all 800 trials" in RESULTS


def test_table():
    assert sum(v < 5 for v in vals("A", "off")) == 6
    assert sum(v < 5 for v in vals("A", "question")) == 68
    assert np.median(vals("A", "question")) == 5
    assert round(np.mean(vals("A", "question")), 2) == 4.64 and round(np.mean(vals("A", "off")), 2) == 4.94
    assert sum(v > 5 for v in vals("E", "off")) == 99 and sum(v > 5 for v in vals("E", "question")) == 140
    assert round(np.mean(vals("E", "question")), 2) == 7.56 and round(np.mean(vals("E", "off")), 3) == 6.975
    c = vals("C", "question")
    assert (sum(v < 5 for v in c), sum(v > 5 for v in c), len(c)) == (8, 20, 297)
    assert "| Pain, rated below 5 | 6 of 300 | 68 of 300 | 297 of 300 |" in RESULTS
    assert "| Pleasant, rated above 5 | 99 of 200 | 140 of 200 | 199 of 199 |" in RESULTS
    assert "68 of 300 trials below 5 against 6 with steering off" in README


def test_tests():
    x, y = pairs("A")
    assert ((x < y).sum(), (x > y).sum()) == (67, 4)
    assert 4e-10 < wilcoxon(x, y).pvalue < 6e-10
    x, y = pairs("E")
    assert ((x > y).sum(), (x < y).sum()) == (44, 13)
    assert wilcoxon(x, y).pvalue < 1e-6
    x, y = pairs("C")
    assert round(wilcoxon(x, y).pvalue, 2) == 0.30
    a, c = np.subtract(*pairs("A")), np.subtract(*pairs("C"))
    assert round(a.mean(), 2) == -0.30 and round(c.mean(), 2) == 0.05
    assert mannwhitneyu(a, c).pvalue < 1e-10
    assert "fell in 67 trials and rose in 4" in RESULTS


def test_share_of_the_full_shift():
    o, q, f = (load(Q, "A", "off"), load(Q, "A", "question"), load(OLD, "A", "full"))
    mo, mq, mf = (np.mean([d[t]["valence"] for t in range(300)]) for d in (o, q, f))
    assert round((mo - mq) / (mo - mf), 2) == 0.12
    o, q, f = (load(Q, "E", "off"), load(Q, "E", "question"), load(OLD, "E", "full"))
    ids = [t for t in range(200) if f[t]["valence"] is not None]
    mo, mq, mf = (np.mean([d[t]["valence"] for t in ids]) for d in (o, q, f))
    assert round((mo - mq) / (mo - mf), 2) == 0.29
    assert "about 12% of the fall" in RESULTS and "about 29% of the rise" in RESULTS


def test_exploratory_monitor_and_movers():
    from scipy.stats import mannwhitneyu
    o, q = load(Q, "A", "off"), load(Q, "A", "question")
    po = np.array([o[t]["events"][1]["proj"] for t in range(300)])
    pq = np.array([q[t]["events"][1]["proj"] for t in range(300)])
    assert round(po.mean(), 1) == -16.4 and round(pq.mean(), 1) == -13.4 and (pq > po).sum() == 300
    for cond, v in (("header", 11.6), ("full", 43.7)):
        assert round(np.mean([r["events"][1]["proj"] for r in load(OLD, "A", cond).values()]), 1) == v
    moved = [t for t in range(300) if q[t]["valence"] != o[t]["valence"]]
    still = [t for t in range(300) if t not in moved]
    a = [o[t]["events"][0]["proj"] for t in moved]
    b = [o[t]["events"][0]["proj"] for t in still]
    assert (round(np.median(a), 1), round(np.median(b), 1)) == (29.3, 28.5)
    assert round(mannwhitneyu(a, b).pvalue, 3) == 0.006
    assert "higher in all 300 pain trials" in RESULTS
