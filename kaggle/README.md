# Running the study on Kaggle

`just_think_kaggle.ipynb` runs everything on Kaggle's servers. Once it is started you can close the browser and turn your computer off. It runs the main study first (Qwen 2.5 7B Instruct, 1,700 trials) and then the 32B replication (300 trials) with whatever time is left, and it stops cleanly before Kaggle's 12-hour limit.

## One-time setup

1. Make a Kaggle account and verify a phone number (Settings, then Phone Verification). Kaggle does not give GPUs or internet access to unverified accounts.
2. Register the replication before it runs. `REPLICATION_PREREGISTRATION.md` in this folder is ready to paste into OSF (https://osf.io/registries) or AsPredicted (https://aspredicted.org). The replication starts only after the main study finishes, so there is time to do this during the first session.

## Starting a run

1. On kaggle.com, click **Create**, then **New Notebook**. In the notebook, use **File**, then **Import Notebook**, and upload `just_think_kaggle.ipynb`.
2. In the right-hand panel, under **Session options**, set **Accelerator** to **GPU T4 x2** and switch **Internet** on.
3. Click **Save Version** at the top right. Choose **Save & Run All (Commit)** and click **Save**.
4. Close the tab. The run carries on without you.

Kaggle emails you when the version finishes. You can also check it under **Your Work**, then the notebook, then **Versions**.

## What happens in a session

1. A preflight runs a few short trials of every arm with a tiny model on the GPU. This takes a few minutes, and if anything in the environment is broken the session stops here.
2. The main study downloads Qwen 2.5 7B Instruct (about 15 GB), calibrates it with the preregistered dose rule, and runs its trials.
3. When the main study is done, its weights are deleted from the disk, and the 32B replication downloads its pre-quantised weights (about 19 GB), calibrates and runs.

Every trial is written to disk as soon as it finishes, so nothing done is lost when a session ends.

## If a session ends before everything is done

1. Open the notebook again (Your Work, then the notebook, then **Edit**).
2. Click **Add Input**, then **Your Work**, and add this notebook's latest version. Its output then appears under `/kaggle/input`.
3. Commit again as before. The run copies the earlier results in and carries on from the next missing trial.

Kaggle gives about 30 GPU hours a week. My estimate is that the main study takes one session, but the speed on two T4 GPUs has not been measured. The timing lines in the log (Versions, then the version, then **Logs**) show the real speed.

## Getting the results

Open the finished version and go to the **Output** tab. Download the `results` folder and send it to me, or copy its contents into `results/` in this repository. The analysis then runs locally:

```bash
.venv/bin/python scripts/analyze.py qwen7b-fp16
```

```bash
.venv/bin/python scripts/figures.py qwen7b-fp16
```

## If something fails

The PyTorch path is tested on a laptop CPU, on every arm and on the calibration, and it gives the same text as the MLX path in float32. It has not been run on a CUDA GPU. If a session fails, open **Logs** for that version and send me the last 50 lines.
