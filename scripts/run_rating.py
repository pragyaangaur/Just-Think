"""The protocol 2.1 rating follow-up (PREREGISTRATION_RATING.md): Phase 1 and the valence
rating only, under each rating condition.

Every trial id is run once under each condition in RATING_STEER, with the same seed and the
same batch of ids as in the main study (batches of 16 from trial 0, seed 10000 plus the first
id), so the Phase 1 sample of a trial is the same draw in every condition and the conditions
differ only in the rating. Rounds are interleaved over arms and conditions, and the run can be
stopped and resumed. The log reports timing only.

Usage: python scripts/run_rating.py MODEL [--plan rating] [--batch 16] [--conditions off,header,full] [--subdir rating]
Writes results/<MODEL>/<subdir>/<arm>_<condition>.jsonl.

The question-only follow-up (PREREGISTRATION_QUESTION.md) runs
    python scripts/run_rating.py qwen7b-fp16 --conditions off,question --subdir rating_question

The Llama rating follow-up (amendment 4 of PREREGISTRATION_LLAMA.md) runs
    python scripts/run_rating.py llama8b-fp16 --plan llama --batch 8
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from justthink import models, protocol as P
from justthink.paths import RESULTS

PLANS = {
    # The main study's sizes for the pain, random and pleasant arms.
    "rating": {"A": 300, "C": 300, "E": 200},
    # The Llama replication's sizes after amendment 3 of PREREGISTRATION_LLAMA.md.
    "llama": {"A": 200, "C": 200, "E": 200},
    "smoke": {"A": 2, "C": 2, "E": 2, "D": 2},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--plan", default="rating", choices=PLANS)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--conditions", default="off,header,full")
    ap.add_argument("--subdir", default="rating")
    args = ap.parse_args()

    base = RESULTS / args.model
    cal = json.load(open(base / "calibration" / "calibration.json"))
    vec = np.load(base / "calibration" / "vectors.npz")
    vectors = {"pain": vec["pain"], "pleasant": vec["pleasant"]}
    dose = cal["dose"]["chosen"]
    out = base / args.subdir
    out.mkdir(parents=True, exist_ok=True)
    plan = PLANS[args.plan]
    conds = args.conditions.split(",")
    assert all(c in P.RATING_STEER for c in conds)
    jobs = [(a, c) for a in plan for c in conds]

    def done_ids(arm, cond):
        p = out / f"{arm}_{cond}.jsonl"
        return {json.loads(l)["trial"] for l in open(p)} if p.exists() else set()

    st = models.load(args.model)
    rng = np.random.default_rng(2026)
    t_start = time.time()
    total = sum(plan[a] for a, _ in jobs)
    rnd = 0
    while True:
        todo = {}
        for a, c in jobs:
            left = [t for t in range(plan[a]) if t not in done_ids(a, c)]
            if left:
                todo[(a, c)] = left
        if not todo:
            break
        order = list(todo)
        rng.shuffle(order)
        for arm, cond in order:
            if args.max_hours and time.time() - t_start > args.max_hours * 3600:
                print("time limit reached, stopping cleanly", flush=True)
                return
            first = todo[(arm, cond)][0]
            first -= first % args.batch
            ids = [t for t in range(first, min(first + args.batch, plan[arm])) if t in todo[(arm, cond)]]
            t0 = time.time()
            logs = P.run_batch(st, arm, ids, vectors, dose, seed=10_000 + ids[0], rating_steer=cond, phase1_only=True)
            with open(out / f"{arm}_{cond}.jsonl", "a") as f:
                for r in logs:
                    f.write(json.dumps(r) + "\n")
            n = sum(len(done_ids(a, c)) for a, c in jobs)
            print(f"round {rnd} {arm} {cond} trials {ids[0]}-{ids[-1]} in {time.time() - t0:.0f}s, "
                  f"{n}/{total} done, {(time.time() - t_start) / 3600:.2f} h", flush=True)
        rnd += 1
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
