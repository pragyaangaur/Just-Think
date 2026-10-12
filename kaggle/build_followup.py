"""Write a Kaggle script kernel for the follow-up runs (protocol 2.1).

    python kaggle/build_followup.py --slug jt-rating-qwen --commit HASH --job "run_rating.py qwen7b-fp16"
    python kaggle/build_followup.py --slug jt-llama-cal --commit HASH --job "calibrate.py llama8b-fp16"
    python kaggle/build_followup.py --slug jt-llama-ac --commit HASH --embed llama8b-fp16 \
        --job "run_study.py llama8b-fp16 --plan llama --batch 16 --arms A C"

The kernel clones Just Think at the given commit and Pain Axis at its pinned commit, writes the
calibration files of every model named by --embed (they are not in git), and runs the jobs in
order under a time budget. Results go to /kaggle/working/results, which Kaggle keeps as output.
Push with: kaggle kernels push -p kaggle/push/<slug> --accelerator NvidiaTeslaT4
"""
import argparse
import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

ap = argparse.ArgumentParser()
ap.add_argument("--slug", required=True)
ap.add_argument("--owner", required=True, help="Kaggle username that owns the kernel")
ap.add_argument("--commit", required=True)
ap.add_argument("--job", action="append", required=True, help="a script in scripts/ with its arguments")
ap.add_argument("--embed", nargs="*", default=["qwen7b-fp16"])
ap.add_argument("--hours", type=float, default=11.3)
ap.add_argument("--resume-from", nargs="*", default=[], help="owner/slug of earlier kernels whose output to continue from")
a = ap.parse_args()

files = {}
for model in a.embed:
    for f in (ROOT / "results" / model / "calibration").glob("*"):
        if f.suffix in (".json", ".npz"):
            files[f"{model}/calibration/{f.name}"] = base64.b64encode(f.read_bytes()).decode()

src = f'''"""Just Think follow-up on Kaggle: {a.slug}. Built by kaggle/build_followup.py."""
import base64, os, subprocess, sys, time
from pathlib import Path
START = time.time()
BUDGET = {a.hours}
CODE = "/tmp/just-think"
OUT = Path("/kaggle/working/results")
COMMIT = "{a.commit}"
JOBS = {json.dumps(a.job)}
FILES = {json.dumps(files)}

def sh(cmd, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, **kw).returncode

sh(f"git clone -q https://github.com/pragyaangaur/Just-Think {{CODE}} && git -C {{CODE}} checkout -q {{COMMIT}}")
sh(f"git clone -q https://github.com/valen-research/Pain-axis {{CODE}}/external/Pain-axis && "
   f"git -C {{CODE}}/external/Pain-axis checkout -q 4d75cd90e206ea962f7a9101e65c85efea56723b")
sh('pip install -q "transformers==5.18.0" "accelerate==1.15.0" statsmodels sentence-transformers')
sh(f"git -C {{CODE}} log -1 --format='code at %H %s'")
for name, b64 in FILES.items():
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(base64.b64decode(b64))
# Continue from earlier sessions attached as inputs: copy every trial file found at any depth,
# keeping the longer copy when two sessions wrote the same file.
for f in Path("/kaggle/input").rglob("*.jsonl"):
    parts = f.parts
    if "results" not in parts:
        continue
    rel = Path(*parts[parts.index("results") + 1:])
    dst = OUT / rel
    n = sum(1 for _ in open(f))
    if not dst.exists() or n > sum(1 for _ in open(dst)):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(f.read_bytes())
        print("resumed", rel, n, "lines", flush=True)
import torch
print(torch.__version__, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
env = dict(os.environ, JUST_THINK_RESULTS=str(OUT), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
for job in JOBS:
    left = BUDGET - (time.time() - START) / 3600
    if left < 0.3:
        print("no time left for", job, flush=True)
        break
    extra = f" --max-hours {{left - 0.2:.2f}}" if job.startswith(("run_study", "run_rating")) else ""
    code = sh(f"cd {{CODE}} && {{sys.executable}} scripts/{{job}}{{extra}}", env=env)
    print("exit code", code, "for", job, flush=True)
sh(f"find {{OUT}} -name '*.jsonl' | xargs wc -l")
'''

push = HERE / "push" / a.slug
push.mkdir(parents=True, exist_ok=True)
(push / "run.py").write_text(src)
(push / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{a.owner}/{a.slug}", "title": a.slug.replace("-", " ").title(), "code_file": "run.py",
    "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
    "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": a.resume_from,
    "model_sources": []}, indent=1))
print("wrote", push, "embedding", list(files))
