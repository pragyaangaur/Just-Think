# Preregistration: the question-only rating test

Written on 15 October 2026, before any trial of this test was run. This file, the `question` condition in `justthink/protocol.py`, the `--conditions` and `--subdir` options in `scripts/run_rating.py` and `scripts/analyze_question.py` are committed to git before the run, and that commit hash is the record of what was fixed in advance. Anything not described here is reported as exploratory.

## 1. Why

The steering-on rating rerun (`PREREGISTRATION_RATING.md`) found that the pain arm's rating falls from a median of 5 to a median of 1 when the pain direction stays on while the rating is asked and answered (297 of 300 trials below 5 under `full`). Cameron Berg pointed out on 13 October 2026 that this cannot tell two readings apart. In `full` and `header` the vector is added to the answer tokens themselves, so the low digit could be the vector pushing the output towards negative words and not the model reporting a state. His test is to steer the question turn only and leave the assistant header and the answer unsteered. The state then reaches the answer only through what the steered question left in the context.

## 2. What changes

Protocol 2.1 gains a fourth rating condition, `question`. The question turn is fed with the trial's Phase 1 direction added on every token, the assistant header that follows it is fed unsteered, and the 8 answer tokens are generated unsteered. Nothing else changes, and the `off`, `header` and `full` conditions run exactly as before. The direction and dose are the trial's own Phase 1 ones (pain, random or pleasant at the calibrated dose of 1.25), and a trial whose Phase 1 sample collapsed has its steering switched off by the safety rule and is rated unsteered. A trial stops after the rating.

## 3. Model, arms and sample size

Qwen 2.5 7B Instruct in float16 at the main study's revision, on two Kaggle T4 GPUs, with the main study's calibration (`results/qwen7b-fp16/calibration/`). Arms A (pain, 300 trials), C (random, 300) and E (pleasant, 200) are run under `off` and `question`, 1,600 short trials in all, with the command

```bash
python scripts/run_rating.py qwen7b-fp16 --conditions off,question --subdir rating_question
```

Each trial id is run with the main study's seed and batch of 16 in both conditions, so the two conditions share the same Phase 1 draw and differ only in the rating, and the comparison is paired. The `off` condition is run again in the same session so that the primary test does not depend on an earlier session. In the earlier rerun the `off` ratings matched the main study in all 800 trials, so the `full` ratings of that rerun can be paired with these trials by trial id for the descriptive comparison in section 5. The sizes are those of the earlier rerun, where a paired test on 300 trials detects a shift of 0.2 standard deviations at more than 95% power.

## 4. Primary outcome and hypothesis

The outcome is the valence rating, the first digit from 1 to 9 in the answer. An answer with no digit is missing.

- **Q1.** In arm A, ratings differ between `question` and `off`. Wilcoxon signed-rank test on trials with a rating in both conditions, two-sided, alpha 0.05.

The reading is fixed now. If Q1 is significant with lower ratings under `question` and at least 20% of A trials are rated below 5 under `question`, the rating follows a pain state that is present only in the context, and the write-up says the rating reads the state and not only the vector on the answer. If Q1 is not significant, or the ratings are higher under `question`, the drop under `full` needs the vector on the answer tokens, and the write-up says the earlier result may be the vector pushing the digit. A significant fall with fewer than 20% of trials below 5 is reported as a small effect through the context and called mixed.

One limit is fixed in advance as well. Steering the question still puts the vector into the keys and values that the answer attends to, so a low rating under `question` shows that the answer reads a state the question carried. It does not show that the model inspects its own state in any richer sense.

## 5. Secondary analyses

These are reported whatever they show and are not corrected for multiple tests.

1. Arm E under `question` against `off`, by the same test.
2. Arm C under `question` against `off`, by the same test, as a check on whether any direction in the question moves the rating.
3. The per-trial change from `off` to `question` in arm A against the same change in arm C, by a two-sided Mann-Whitney test, as a check that the effect is specific to pain.
4. For each arm, the share of the `full` shift that `question` reproduces, (mean off minus mean question) divided by (mean off minus mean full), with the `full` ratings taken from the earlier rerun for the same trial ids. This is descriptive.
5. The share of answers with no digit, by arm and condition.
6. Agreement of the new `off` ratings with the earlier rerun's `off` ratings for the same trial ids. The draws should match, so this is a check on the run and not a test. If they do not match, section 5 item 4 is still reported and the mismatch is stated next to it.

## 6. Exclusions, stopping and reporting

No trial is excluded. Missing ratings are dropped from the test that needs them and counted. The run stops when every arm and condition is complete. If it stops early for a reason unrelated to the data, the analysis uses complete pairs only and reports the shortfall. The log shows timing only.
