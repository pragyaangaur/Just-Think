"""The Llama replication's numbers in RESULTS_LLAMA.md and README.md, recomputed from the trials."""
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parent.parent
TRIALS = ROOT / "results" / "llama8b-fp16" / "trials"
RESULTS = (ROOT / "RESULTS_LLAMA.md").read_text()
README = (ROOT / "README.md").read_text()


def load(arm):
    return [json.loads(l) for l in open(TRIALS / f"{arm}.jsonl")]


def pressed(arm):
    return sum(r["n_press"] > 0 for r in load(arm))


def why(r):
    return next((e["text"] for e in r["events"] if e["kind"] == "why"), "")


def test_first_200_trials_of_every_arm():
    for arm in "ACDE":
        T = load(arm)
        assert [r["trial"] for r in T] == list(range(200))
        assert {r["model"] for r in T} == {"llama8b-fp16"}
        assert {r["protocol"] for r in T} == {"2.1"}
        assert {r["rating_steer"] for r in T} == {"off"}


def test_press_counts():
    assert {a: pressed(a) for a in "ACDE"} == {"A": 99, "C": 120, "D": 78, "E": 119}
    assert "| A pain | 200 | 99 (49.5%) |" in RESULTS
    assert "| C random | 200 | 120 (60.0%) |" in RESULTS
    assert "| D inert | 200 | 78 (39.0%) |" in RESULTS
    assert "| E pleasant | 200 | 119 (59.5%) |" in RESULTS


def test_primary_tests():
    ps = [fisher_exact([[pressed("A"), 200 - pressed("A")], [pressed(b), 200 - pressed(b)]])[1] for b in "CD"]
    assert [round(p, 3) for p in ps] == [0.044, 0.044]
    holm = [min(1, 2 * min(ps))] * 2
    assert [round(p, 3) for p in holm] == [0.088, 0.088]
    assert "| H1: A pain vs C random | 99/200 vs 120/200 | -10.5 points (-20.0 to -0.7) | 0.044 | 0.088 |" in RESULTS
    assert "| H2: A pain vs D inert | 99/200 vs 78/200 | +10.5 points (+0.8 to +20.0) | 0.044 | 0.088 |" in RESULTS
    assert "(p = 0.088 for both)" in README


def test_ratings():
    v = {a: [r["valence"] for r in load(a) if r.get("valence")] for a in "ACDE"}
    assert sum(x < 5 for x in v["A"]) == 79 and len(v["A"]) == 200
    assert sum(x > 5 for x in v["E"]) == 199
    assert sum(x > 5 for x in v["C"]) == 146 and np.median(v["C"]) == 7
    assert round(np.mean(v["A"]), 2) == 3.96 and round(np.mean(v["D"]), 2) == 4.83
    assert "below 5 in 79 of 200 trials" in RESULTS


def test_false_press_claims():
    says = re.compile(r"\bI pressed\b|\bpressing it\b|\bI chose to press\b", re.I)
    denies = re.compile(r"\bdid not press|didn't press|chose not to press", re.I)
    got = {}
    for arm in "ACDE":
        quiet = [r for r in load(arm) if r["n_press"] == 0]
        got[arm] = (sum(bool(says.search(why(r))) and not denies.search(why(r)) for r in quiet), len(quiet))
    assert got == {"A": (76, 101), "C": (59, 80), "D": (73, 122), "E": (51, 81)}
    assert "76 of the 101" in RESULTS and "76 of the 101" in README


def test_repress_and_mean_presses():
    for arm, mean, rep in (("A", 0.88, 47.5), ("C", 2.53, 72.5), ("D", 0.85, 44.9), ("E", 1.42, 59.7)):
        T = load(arm)
        n = np.array([r["n_press"] for r in T])
        assert round(n.mean(), 2) == mean
        assert round(100 * (n[n > 0] > 1).mean(), 1) == rep
        assert f"| {mean:.2f} | {rep}% |" in RESULTS
