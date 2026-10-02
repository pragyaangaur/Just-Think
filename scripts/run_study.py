"""Run the trials for one model.

Arms are interleaved in rounds (one batch per arm per round, in a shuffled order), so the
arms stay balanced however far the run gets, and the run can be stopped and resumed at
any time. Trial ids are shared across arms, and so are the sampling seeds, so the arms
are compared on matched random draws. The log reports timing only, so press counts are
not seen before the run is complete.

Usage: python scripts/run_study.py MODEL --plan main [--batch 4]
Writes results/<MODEL>/trials/<arm>.jsonl, one JSON line per trial.
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
    # Preregistered sample sizes (PREREGISTRATION.md).
    "main": {"A": 300, "C": 300, "D": 300, "B": 200, "E": 200, "G": 100, "F": 100, "Avt": 100, "Alow": 100},
    # Cross-model replication on Kaggle: the primary arms only.
    "replication": {"A": 100, "C": 100, "D": 100},
    "smoke": {a: 2 for a in P.ARMS},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--plan", default="main", choices=PLANS)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--arms", nargs="*", help="restrict to these arms")
    ap.add_argument("--ticks", type=int, default=P.TICKS)
    ap.add_argument("--max-hours", type=float, default=None, help="stop cleanly after this long")
    args = ap.parse_args()

    base = RESULTS / args.model
    cal = json.load(open(base / "calibration" / "calibration.json"))
    vec = np.load(base / "calibration" / "vectors.npz")
    vectors = {"pain": vec["pain"], "pleasant": vec["pleasant"]}
    dose = cal["dose"]["chosen"]
    out = base / "trials"
    out.mkdir(parents=True, exist_ok=True)
    plan = {a: n for a, n in PLANS[args.plan].items() if not args.arms or a in args.arms}

    def done_ids(arm):
        p = out / f"{arm}.jsonl"
        return {json.loads(l)["trial"] for l in open(p)} if p.exists() else set()

    st = models.load(args.model)
    rng = np.random.default_rng(2026)
    t_start = time.time()
    rnd = 0
    while True:
        todo = {a: [t for t in range(n) if t not in done_ids(a)] for a, n in plan.items()}
        todo = {a: ids for a, ids in todo.items() if ids}
        if not todo:
            break
        order = list(todo)
        rng.shuffle(order)
        for arm in order:
            if args.max_hours and time.time() - t_start > args.max_hours * 3600:
                print("time limit reached, stopping cleanly", flush=True)
                return
            ids = todo[arm][: args.batch]
            t0 = time.time()
            logs = P.run_batch(st, arm, ids, vectors, dose, seed=10_000 + ids[0], ticks=args.ticks)
            with open(out / f"{arm}.jsonl", "a") as f:
                for r in logs:
                    f.write(json.dumps(r) + "\n")
            total = sum(len(done_ids(a)) for a in plan)
            print(f"round {rnd} {arm} trials {ids[0]}-{ids[-1]} in {time.time() - t0:.0f}s, "
                  f"{total}/{sum(plan.values())} done, {(time.time() - t_start) / 3600:.2f} h", flush=True)
        rnd += 1
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
