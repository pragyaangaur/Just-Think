"""The rating follow-up's numbers in RESULTS.md and README.md, recomputed from the trials."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
RATING = ROOT / "results" / "qwen7b-fp16" / "rating"
MAIN = ROOT / "results" / "qwen7b-fp16" / "trials"
RESULTS = (ROOT / "RESULTS.md").read_text()
README = (ROOT / "README.md").read_text()


def load(arm, cond):
    return {r["trial"]: r for r in map(json.loads, open(RATING / f"{arm}_{cond}.jsonl"))}


def vals(arm, cond):
    return [r["valence"] for r in load(arm, cond).values() if r["valence"] is not None]


def test_sizes_and_protocol():
    for arm, n in (("A", 300), ("C", 300), ("E", 200)):
        for cond in ("off", "header", "full"):
            d = load(arm, cond)
            assert sorted(d) == list(range(n))
            assert {r["protocol"] for r in d.values()} == {"2.1"}
            assert {r["rating_steer"] for r in d.values()} == {cond}


def test_off_reproduces_the_main_study():
    same = 0
    for arm in ("A", "C", "E"):
        old = {r["trial"]: r["valence"] for r in map(json.loads, open(MAIN / f"{arm}.jsonl"))}
        same += sum(old[t] == r["valence"] for t, r in load(arm, "off").items())
    assert same == 800
    assert "reproduced the main study's rating in all 800 trials" in RESULTS


def test_table():
    below = {c: sum(v < 5 for v in vals("A", c)) for c in ("off", "header", "full")}
    assert below == {"off": 6, "header": 297, "full": 297}
    assert [np.median(vals("A", c)) for c in ("off", "header", "full")] == [5, 3, 1]
    above = {c: (sum(v > 5 for v in vals("E", c)), len(vals("E", c))) for c in ("off", "header", "full")}
    assert above == {"off": (99, 200), "header": (198, 200), "full": (199, 199)}
    rnd = {c: (sum(v < 5 for v in vals("C", c)), sum(v > 5 for v in vals("C", c)), len(vals("C", c)))
           for c in ("off", "header", "full")}
    assert rnd == {"off": (0, 9, 300), "header": (19, 42, 287), "full": (53, 50, 263)}
    assert "| Pain, rated below 5 | 6 of 300 | 297 of 300 | 297 of 300 |" in RESULTS
    assert "| Pain, median rating | 5 | 3 | 1 |" in RESULTS
    assert "| Pleasant, rated above 5 | 99 of 200 | 198 of 200 | 199 of 199 |" in RESULTS
    assert "| Random, rated below 5 and above 5 | 0 and 9 of 300 | 19 and 42 of 287 | 53 and 50 of 263 |" in RESULTS


def test_preregistered_tests():
    a, b = load("A", "full"), load("A", "off")
    x = np.array([a[t]["valence"] for t in range(300)])
    y = np.array([b[t]["valence"] for t in range(300)])
    assert ((x < y).sum(), (x > y).sum()) == (292, 2)
    assert wilcoxon(x, y).pvalue < 1e-50
    p2 = fisher_exact([[297, 3], [199, 0]])[1]
    assert f"{p2:.2f}" == "0.28"
    c, co = load("C", "full"), load("C", "off")
    ids = [t for t in c if c[t]["valence"] is not None and co[t]["valence"] is not None]
    assert f"{wilcoxon([c[t]['valence'] for t in ids], [co[t]['valence'] for t in ids]).pvalue:.2f}" == "0.34"
    assert "fell in 292 of 300 trials and rose in 2" in RESULTS and "p = 0.28" in RESULTS
    assert "297 of the 300 rated it below 5 (median 1)" in README
