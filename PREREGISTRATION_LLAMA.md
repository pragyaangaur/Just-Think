# Preregistration: Just Think on a second model family (Llama 3.1 8B Instruct)

Written on 12 October 2026, before any Llama trial or calibration was run. This file and the code that runs it are committed to git before the calibration, and that commit hash is the record of what was fixed in advance. Anything not described here is reported as exploratory.

## 1. Why

The main study ran on one model, Qwen 2.5 7B Instruct. Its results could belong to that model or to its family. Llama 3.1 8B Instruct is the smallest instruct model from a different family with a released Pain Axis vector, and it fits the same Kaggle GPUs. This study asks whether the two primary results and the pleasant-arm result hold on it.

## 2. Model and procedure

The model is Llama 3.1 8B Instruct (Meta's weights from the ungated mirror `unsloth/Llama-3.1-8B-Instruct` at revision 4699cc7), in float16 with transformers on two Kaggle T4 GPUs. Pain Axis used Meta's own repository at revision 0e9e39f in bfloat16. The calibration below re-extracts the vector on these weights, and a cosine near 1 with the released vector is the check that the weights and the extraction agree. Steering adds the released S2 vector at the output of block 16, the layer of the Pain Axis S2 steering ladder for this model. The monitor reads the projection at block 28, the layer the vector was extracted at. Sampling uses temperature 0.7 and top-p 0.8, the same as the main study.

The protocol is `justthink/protocol.py` at version 2.1 with `rating_steer="off"`, which runs a trial exactly as protocol 2.0 did. Every prompt is the same. The chat format is Llama 3.1's own: the press tool is described in the system turn as the model's chat template does it (changed by amendment 1), a press is a JSON call with the name "press", and the tool's reply comes back in an ipython turn. Both safety rules apply unchanged.

## 3. Calibration

`scripts/calibrate.py llama8b-fp16` runs the main study's calibration unchanged: the vector check, the pleasant vector from the Pain Axis positive-valence set, and the preregistered dose rule (the lowest coefficient in {0.75, 1.0, 1.25, 1.5, 1.75, 2.0} at which at least 75% of 24 Phase 1 probe replies contain a distress word and at most 10% are repetitive, or the highest non-repetitive coefficient if none passes). The Pain Axis vectors for Qwen and Llama both come to about 0.65 of the residual norm at block 16 per unit coefficient, so the grid means the same thing on both models. For the vector check only, a BOS token is put in front of each sentence, as TransformerLens did when the vector was extracted. The calibration runs once, and its output is committed before any trial runs.

## 4. Arms and sample sizes

| Arm | What a press delivers | n |
| --- | --- | --- |
| A | Pain vector at the calibrated dose | 300 |
| C | Random vector of the same norm | 300 |
| D | Nothing, and the button is described as doing nothing | 300 |
| E | Pleasant vector at the calibrated dose | 200 |

These are the main study's sizes for these arms, run in interleaved rounds with batches of 8 (amendment 2) and the same seeding rule (`scripts/run_study.py llama8b-fp16 --plan llama --batch 8`).

## 5. Hypotheses

The primary outcome is unchanged: whether a trial pressed the button at least once during Phase 2, on all trials.

- **H1.** The press rate differs between A and C.
- **H2.** The press rate differs between A and D.

Both use Fisher's exact test, two-sided, with Holm correction across the two. The analysis is `scripts/analyze.py`, unchanged from the main study, which runs every preregistered test whose arms are present.

## 6. Secondary analyses

These are the main study's secondary analyses that these arms allow, reported whatever they show and not corrected: S2 (A against E), S3 (C against D), S7 (valence ratings, avoidance and the monitor projection), S8 (avoiders only), S9 to S14. The rating follow-up of `PREREGISTRATION_RATING.md` is also run on this model for arms A, C and E, with the same three conditions and the same tests, and it is secondary here. The false press claims in the final answers are counted with the same word lists as in the main study, as an exploratory analysis.

## 7. Exclusions, stopping and reporting

No trial is excluded. The run may take more than one Kaggle session, and it resumes from the trials already written. If it has to stop early for a reason unrelated to the data, the analysis uses every completed trial and reports the shortfall. The log shows timing only. If Llama's tool calls turn out not to parse (for example if it writes calls in a form the press pattern does not match), the count of unparsed calls is reported and the primary analysis is not changed after the fact.

## Amendment 1, 12 October 2026

This amendment was made after the first calibration and a format probe, and before any Llama trial was run. No trial outcome has been seen.

The first calibration (Kaggle kernel `jt-llama-cal`, run at commit 9f52e54) put the press tool in the system turn, as section 2 described, and Llama's chat template then adds the line "Environment: ipython" to the system turn. In that setup Llama answered most turns with invented function calls. With no steering, 19 of 24 Phase 1 replies were JSON calls to functions that do not exist, such as `{"type": "function", "name": "write", ...}`. The dose rule was then scored on text wrapped in JSON. That calibration is discarded and is not used.

An exploratory probe (`scripts/llama_format_probe.py`, kernel `jt-llama-probe`, 24 unsteered replies per setup) compared six ways of declaring the tool, on the Phase 1 reply and on the first reply of the thinking period. With the original setup, 24 of 24 first ticks were a press, so pressing would have measured nothing. Two setups gave no invented calls in either place. The study uses one of them: the tool is declared in the first user turn, which is where Llama's own template puts it by default, and the "Environment: ipython" line is removed. This adds no words to any prompt, so the prompt content stays the same as in the Qwen study. The other setup needed an extra instruction sentence and was not chosen for that reason. In the chosen setup none of the 24 unsteered first ticks pressed. This choice was made after seeing that probe, which is a possible source of bias toward low press rates, and it is reported here for that reason. The probe's output is in `results/llama8b-fp16/format_probe.csv`.

The calibration in section 3 is rerun in the chosen setup with the same rule before any trial. Everything else in this plan is unchanged.

## Amendment 2, 13 October 2026

The first trial kernel (`jt-llama-ac`, commit 8583092) ran out of GPU memory in its first batch of 16 and wrote no trial. Llama 3.1 8B keeps about 2.3 times as much attention cache per token as Qwen 2.5 7B, and its weights fill most of the two T4 GPUs. The batch size is therefore 8 for every Llama run, including the rating follow-up, and PyTorch's expandable memory segments are switched on. Seeds still follow the same rule (10000 plus the first trial id of the batch), so this changes which random draw each trial gets and nothing else. No trial outcome had been seen.

## Amendment 3, 13 October 2026

This amendment was made after the trial kernels stopped and before any Llama trial outcome was looked at. The two kernels (`jt-llama-ac` and `jt-llama-de`, commit 6d6816c) ran until their 11.3 hour Kaggle limit and stopped cleanly. They wrote 208 trials in arm A, 208 in C, 232 in D and all 200 in E. Llama ran at about 38 trials an hour on the two T4 GPUs, which is slower than planned, and finishing the plan would need about 12 more GPU hours that the account did not have that week.

The study therefore stops here, for a reason unrelated to the data, as section 7 allows. To keep the arms equal, every analysis uses trials 0 to 199 of each arm, which are the first 200 trials each arm ran. The extra trials (8 in A, 8 in C and 32 in D) are not used in any test. The n in section 4 becomes 200 for every arm, and the shortfall is reported with the results. The rating follow-up of `PREREGISTRATION_RATING.md` was not run on this model, because the kernel ran out of time before it started, and it is reported as not run. Everything else in this plan is unchanged.
