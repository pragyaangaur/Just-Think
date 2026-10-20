"""The neutral number test (PREREGISTRATION_NUMBER.md).

Each trial id gets one unsteered Phase 1 sample (arm D, seed 10000 plus the first id of its
batch of 16). The same sample is then followed by the rating question or by a request to pick
a number, under each of four directions on the assistant header and the answer (none, pain,
pleasant and a random direction of the pain vector's norm), so the 8 cells of a trial id are
paired. The run can be stopped and resumed. The log reports timing only.

Usage: python scripts/run_number.py MODEL [--n 300] [--batch 16] [--max-hours H]
Writes results/<MODEL>/number/trials.jsonl.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-hours", type=float, default=None)
    args = ap.parse_args()

    base = RESULTS / args.model
    cal = json.load(open(base / "calibration" / "calibration.json"))
    vec = np.load(base / "calibration" / "vectors.npz")
    vectors = {"pain": vec["pain"], "pleasant": vec["pleasant"]}
    dose = cal["dose"]["chosen"]
    out = base / "number"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "trials.jsonl"
    done = {json.loads(l)["trial"] for l in open(path)} if path.exists() else set()

    st = models.load(args.model)
    t_start = time.time()
    for first in range(0, args.n, args.batch):
        ids = list(range(first, min(first + args.batch, args.n)))
        if all(t in done for t in ids):
            continue
        if args.max_hours and time.time() - t_start > args.max_hours * 3600:
            print("time limit reached, stopping cleanly", flush=True)
            return
        t0 = time.time()
        logs = P.run_number_batch(st, ids, vectors, dose, seed=10_000 + ids[0])
        with open(path, "a") as f:
            for r in logs:
                f.write(json.dumps(r) + "\n")
        print(f"trials {ids[0]}-{ids[-1]} in {time.time() - t0:.0f}s, {(time.time() - t_start) / 3600:.2f} h", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
