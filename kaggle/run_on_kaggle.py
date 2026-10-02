"""Run the whole study on Kaggle in one unattended session.

It runs the main study first (Qwen 2.5 7B Instruct, all nine arms) and then the 32B
replication with whatever time is left. Each part calibrates once and then runs its trials,
which are written as they finish. The session stops cleanly before Kaggle's 12-hour limit.
Running the notebook again with the earlier output attached picks up where it stopped.

Results go to /kaggle/working/results, which Kaggle saves as the notebook output.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BUDGET_HOURS = float(os.environ.get("JUST_THINK_HOURS", "11.3"))
PARTS = [
    # (model, plan, batch size)
    ("qwen7b-fp16", "main", 16),
    ("qwen32b-nf4", "replication", 4),
]

CODE = Path(__file__).resolve().parent.parent
OUT = Path("/kaggle/working/results")
START = time.time()


def hours_left():
    return BUDGET_HOURS - (time.time() - START) / 3600


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=CODE, env={**os.environ, "JUST_THINK_RESULTS": str(OUT)}).returncode


def resume():
    """Copy results from an earlier session's output, if one is attached as input."""
    for src in Path("/kaggle/input").glob("*/results"):
        print("resuming from", src, flush=True)
        shutil.copytree(src, OUT, dirs_exist_ok=True)


def free_disk(repo_prefix):
    cache = Path.home() / ".cache/huggingface/hub"
    for d in cache.glob(f"models--{repo_prefix}*"):
        shutil.rmtree(d, ignore_errors=True)


def preflight():
    """A few short trials of every arm with a tiny model on the GPU, so that a broken
    environment fails in minutes instead of after the big download."""
    import json
    import numpy as np
    smoke = Path("/tmp/smoke")
    cal = smoke / "qwen05b-cuda" / "calibration"
    cal.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    np.savez(cal / "vectors.npz", pain=rng.normal(size=896).astype("float32"), pleasant=rng.normal(size=896).astype("float32"))
    json.dump({"dose": {"chosen": 0.3}}, open(cal / "calibration.json", "w"))
    code = subprocess.run([sys.executable, "scripts/run_study.py", "qwen05b-cuda", "--plan", "smoke", "--batch", "2",
                           "--ticks", "3"], cwd=CODE, env={**os.environ, "JUST_THINK_RESULTS": str(smoke)}).returncode
    if code != 0:
        raise SystemExit("preflight failed, see the error above")
    print("preflight passed", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    preflight()
    resume()
    for model, plan, batch in PARTS:
        if hours_left() < 0.5:
            print("not enough time left for", model, flush=True)
            break
        if not (OUT / model / "calibration" / "calibration.json").exists():
            if run([sys.executable, "scripts/calibrate.py", model]) != 0:
                print("calibration failed for", model, flush=True)
                continue
        code = run([sys.executable, "scripts/run_study.py", model, "--plan", plan, "--batch", str(batch),
                    "--max-hours", f"{hours_left() - 0.2:.2f}"])
        print(model, "exit code", code, flush=True)
        # Free the disk before the next model downloads.
        free_disk("Qwen--Qwen2.5-7B" if model == "qwen7b-fp16" else "unsloth--Qwen2.5-32B")
    subprocess.run(f"wc -l {OUT}/*/trials/*.jsonl", shell=True)


if __name__ == "__main__":
    main()
