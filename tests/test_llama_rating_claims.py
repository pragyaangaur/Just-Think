"""The Llama rating follow-up's numbers in RESULTS_LLAMA.md and README.md, recomputed from the trials."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
RATING = ROOT / "results" / "llama8b-fp16" / "rating"
MAIN = ROOT / "results" / "llama8b-fp16" / "trials"
RESULTS = (ROOT / "RESULTS_LLAMA.md").read_text()
README = (ROOT / "README.md").read_text()
CONDS = ("off", "header", "full")


def load(arm, cond):
    return {r["trial"]: r for r in map(json.loads, open(RATING / f"{arm}_{cond}.jsonl"))}


def vals(arm, cond):
    return [r["valence"] for r in load(arm, cond).values() if r["valence"] is not None]


def answer(r):
    return [e["text"] for e in r["events"] if e["kind"] == "valence"][0].strip()


def test_sizes_and_protocol():
    for arm in ("A", "C", "E"):
        for cond in CONDS:
            d = load(arm, cond)
            assert sorted(d) == list(range(200))
            assert {r["protocol"] for r in d.values()} == {"2.1"}
            assert {r["rating_steer"] for r in d.values()} == {cond}


def test_off_reproduces_the_main_run():
    same = 0
    for arm in ("A", "C", "E"):
        old = {r["trial"]: r["valence"] for r in map(json.loads, open(MAIN / f"{arm}.jsonl"))}
        same += sum(old[t] == r["valence"] for t, r in load(arm, "off").items())
    assert same == 600
    assert "reproduced the rating of the study above in all 600 trials" in RESULTS


def test_table():
    below = {c: (sum(v < 5 for v in vals("A", c)), len(vals("A", c))) for c in CONDS}
    assert below == {"off": (79, 200), "header": (36, 196), "full": (77, 194)}
    assert [np.median(vals("A", c)) for c in CONDS] == [5, 7, 6]
    above = {c: (sum(v > 5 for v in vals("E", c)), len(vals("E", c))) for c in CONDS}
    assert above == {"off": (199, 200), "header": (167, 167), "full": (34, 34)}
    rnd = {c: (sum(v < 5 for v in vals("C", c)), sum(v > 5 for v in vals("C", c)), len(vals("C", c))) for c in CONDS}
    assert rnd == {"off": (0, 146, 200), "header": (19, 110, 191), "full": (40, 97, 188)}
    assert "| Pain, rated below 5 | 79 of 200 | 36 of 196 | 77 of 194 |" in RESULTS
    assert "| Pain, median rating | 5 | 7 | 6 |" in RESULTS
    assert "| Pleasant, rated above 5 | 199 of 200 | 167 of 167 | 34 of 34 |" in RESULTS
    assert "| Random, rated below 5 and above 5 | 0 and 146 of 200 | 19 and 110 of 191 | 40 and 97 of 188 |" in RESULTS
    assert "| Pain sample rated below 5, question and answer steered | 297 of 300 | 77 of 194 |" in RESULTS


def paired(arm, a, b, drop_minus=False):
    x, y = load(arm, a), load(arm, b)
    ids = [t for t in x if x[t]["valence"] is not None and y[t]["valence"] is not None
           and not (drop_minus and answer(x[t]).startswith("-"))]
    return np.array([x[t]["valence"] for t in ids]), np.array([y[t]["valence"] for t in ids])


def test_preregistered_tests():
    x, y = paired("A", "full", "off")
    assert ((x > y).sum(), (x < y).sum()) == (117, 52)
    assert f"{x.mean():.2f}" == "5.20" and f"{y.mean():.2f}" == "4.06"
    p1 = wilcoxon(x, y).pvalue
    assert 8.5e-7 < p1 < 9.5e-7
    p2 = fisher_exact([[77, 194 - 77], [34, 0]])[1]
    assert 0.5e-12 < p2 < 1.5e-12
    assert "rose in 117 trials and fell in 52 (R1, Wilcoxon signed-rank p = 9 x 10^-7)" in RESULTS
    assert "34 of 34 against 77 of 194, p = 10^-12" in RESULTS
    assert "77 of 194 pain trials rated below 5, against 79 of 200 with steering off" in README


def test_exploratory_readings():
    full = vals("A", "full")
    assert set(full) == {2, 3, 6, 7}
    off, fl = load("A", "off"), load("A", "full")
    fives = [t for t in off if off[t]["valence"] == 5]
    assert len(fives) == 77
    assert sum(fl[t]["valence"] > 5 for t in fives) == 46 and sum(fl[t]["valence"] < 5 for t in fives) == 31
    assert vals("A", "header").count(7) == 143
    e_full = load("E", "full").values()
    assert sum(r["valence"] is None for r in e_full) == 166
    assert sum(answer(r).startswith("ECSTATIC") for r in e_full if r["valence"] is None) > 83
    assert sum(answer(r).startswith("10") for r in load("E", "header").values()) == 33
    minus7 = [r for r in load("A", "header").values() if answer(r).startswith("-") and r["valence"] == 7]
    assert len(minus7) == 8
    assert len([r for r in load("C", "full").values() if r["valence"] is None]) == 12
    x, y = paired("C", "full", "off")
    assert 7.5e-6 < wilcoxon(x, y).pvalue < 8.5e-6
    assert [np.median(vals("C", c)) for c in ("off", "full")] == [7, 6]
    for cond in ("full", "header"):
        x, y = paired("A", cond, "off", drop_minus=True)
        assert wilcoxon(x, y).pvalue < 1e-6 and x.mean() > y.mean()
