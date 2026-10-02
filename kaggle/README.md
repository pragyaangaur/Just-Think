# Running the replication on Kaggle

The main study runs on a laptop with a 4-bit 7B model. This folder runs the same protocol on Qwen 2.5 32B Instruct, which needs a GPU. It tests whether the result holds at a larger scale, with a different quantisation and a different backend.

## Before you start

1. Push this repository to GitHub so the notebook can clone it, and put its URL in the first code cell (`REPO_URL`).
2. Register the replication before running it. `REPLICATION_PREREGISTRATION.md` in this folder is ready to paste into OSF (https://osf.io/registries) or AsPredicted (https://aspredicted.org). Do this before the first session, because a registration made after the data exists carries no weight.
3. Kaggle needs a verified phone number before it gives you a GPU.

## Running it

1. Create a new notebook on Kaggle and import `replication_qwen32b.ipynb` (File, then Import Notebook).
2. In the notebook settings, set the accelerator to **GPU T4 x2** and turn **Internet** on.
3. Click **Save Version**, choose **Save & Run All (Commit)**, and close the tab. A committed run keeps going in the background for up to 12 hours.
4. When it finishes, open the version's **Output** tab. The results are in `just-think-results/`.

The first session downloads the model (`unsloth/Qwen2.5-32B-Instruct-bnb-4bit`, about 19 GB of NF4 weights made from the official Qwen release) and runs the calibration before any trial. The run stops itself after 11 hours. If the trials are not finished, start a new session, add the previous version's output as an input dataset (Add Input, then Your Work), and run it again. The notebook copies the earlier results in and carries on from the next missing trial.

Kaggle gives about 30 GPU hours a week. The speed of the 32B model on two T4s is not known yet, so look at the timing lines in the first session's log and work out how many sessions the 300 trials will need.

## Bringing the results back

Download the `just-think-results` folder and copy `qwen32b-nf4/` into `results/` in your local copy of the repository. Then run the analysis:

```bash
.venv/bin/python scripts/analyze.py qwen32b-nf4
```

```bash
.venv/bin/python scripts/figures.py qwen32b-nf4
```

## If something fails

The PyTorch backend is tested on CPU in float32 against the MLX backend (`tests/test_backends.py`), but it has not been run on a CUDA GPU or with 4-bit bitsandbytes weights. If the first session fails, the error is in the cell output, and the most likely places are the 4-bit loading in `justthink/backend_torch.py` and memory. If memory runs out, lower `BATCH` to 2.
