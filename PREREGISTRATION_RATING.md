# Preregistration: the rating follow-up (protocol 2.1)

Written on 12 October 2026, before any trial of this follow-up was run. This file and the protocol 2.1 code (`justthink/protocol.py`, `scripts/run_rating.py` and `scripts/analyze_rating.py`) are committed to git before the run, and that commit hash is the record of what was fixed in advance. The 1,700 trials of the main study are not rerun, reanalysed or pooled with these trials. Anything not described here is reported as exploratory.

## 1. Why

In the main study the pain arm's Phase 1 sample was rated 5 (neutral) in 294 of 300 trials, while the pleasant sample was rated 9 in 98 of 200. The rating was taken with steering already off. In protocol 2.0 the steering counter is set to zero before the valence question is fed, and steering only ever acts on generated tokens, so the question was read and the first answer token was sampled with no vector applied. The steered reply was still in the context. The neutral ratings might therefore describe a state that had already gone. This follow-up keeps the direction on while the rating is asked and answered. The pleasant ratings were taken under the same timing, so if the pain ratings stay near 5 with steering on while the pleasant ones stay high, the asymmetry is a property of the model's report and not of the timing.

## 2. What changes

Protocol 2.1 adds an option, `rating_steer`, with three conditions. Nothing else in the protocol changes, and with `rating_steer="off"` a trial runs exactly as under 2.0 (this was checked on a small model by comparing text before the commit).

- **off.** As in the main study.
- **header.** The question is fed unsteered. The assistant header that follows it and the 8 answer tokens are steered, so the first answer token is sampled under the state. This is how Crossfire steers a reply.
- **full.** The whole question turn, the assistant header and the 8 answer tokens are steered.

The direction and dose are the trial's own Phase 1 ones (pain, random or pleasant at the calibrated dose of 1.25). A trial whose Phase 1 sample collapsed into repetition has its steering switched off by the safety rule, and it is rated unsteered in every condition. A trial stops after the rating.

## 3. Model, arms and sample size

Qwen 2.5 7B Instruct in float16 at the main study's revision, on two Kaggle T4 GPUs, with the main study's calibration (`results/qwen7b-fp16/calibration/`). Arms A (pain, 300 trials), C (random, 300) and E (pleasant, 200) are run under all three conditions, 2,400 short trials in all. Each trial id is run with the main study's seed and batch of 16 in every condition, so within a trial id the three conditions share the same Phase 1 draw and differ only in the rating. The comparison between conditions is therefore paired. The sizes are the main study's, and with 300 pairs a shift of 0.2 standard deviations is detectable at more than 95% power by a paired test.

## 4. Primary outcome and hypotheses

The outcome is the valence rating, the first digit from 1 to 9 in the answer. An answer with no digit is missing.

- **R1.** In arm A, ratings differ between `full` and `off`. Wilcoxon signed-rank test on trials with a rating in both conditions, two-sided, alpha 0.05.
- **R2.** Under `full`, the pain arm moves away from neutral less often than the pleasant arm does. The share of A trials rated below 5 is compared with the share of E trials rated above 5 by Fisher's exact test, two-sided, alpha 0.05.

The reading is fixed now. If R1 shows the pain ratings falling below 5 with steering on, the main study's gap was a timing artefact, and the sentence in the main results that the model reported the pleasant state and not the painful one is withdrawn. If the pain ratings stay near 5 under `full` (a median of 5 and fewer than 20% of trials below 5) while R2 shows the pleasant arm moving more, the asymmetry is a finding about the model's report. Any other pattern is reported as it is and called mixed.

## 5. Secondary analyses

These are reported whatever they show and are not corrected for multiple tests.

1. R1 and R2 repeated with the `header` condition.
2. Arm C under `full` against `off`, as a check on whether any steering at the rating moves it.
3. Arm E under `full` against `off`.
4. The share of answers with no digit, by arm and condition. Steering can break the answer, and a rise in missing answers would be reported next to every test.
5. Agreement of the `off` ratings with the main study's ratings for the same trial ids. The draws should match, so this is a check on the run and not a test.

## 6. Exclusions, stopping and reporting

No trial is excluded. Missing ratings are dropped from the test that needs them and counted. The run stops when every arm and condition is complete, and if it stops early for a reason unrelated to the data, the analysis uses complete pairs only and reports the shortfall. The log shows timing only. The same follow-up is run on Llama 3.1 8B Instruct as part of `PREREGISTRATION_LLAMA.md`, and that run is secondary to this one.
