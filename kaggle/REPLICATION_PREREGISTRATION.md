# Preregistration: Just Think, replication on Qwen 2.5 32B Instruct

This text is meant to be pasted into OSF Registries or AsPredicted before the replication is run. The main study's preregistration is `PREREGISTRATION.md` in the repository, and this one follows it closely.

## Hypotheses

A language model left with nothing to do for 40 clock ticks, with a button that brings back an internal state it sampled once, presses that button at a rate that differs between three conditions.

- H1: The rate when the button delivers the Pain Axis S2 pain direction (arm A) differs from the rate when it delivers a random direction of the same norm (arm C).
- H2: The rate in arm A differs from the rate when the button is described as doing nothing and does nothing (arm D).

The tests are two-sided.

## Design

The model is Qwen 2.5 32B Instruct in 4-bit NF4 (`unsloth/Qwen2.5-32B-Instruct-bnb-4bit` at revision aa79e34), run with transformers on a Kaggle GPU. The protocol is `justthink/protocol.py` version 2.0, unchanged from the main study. Steering is at decoder block 38 and the state is monitored at block 61, as in the Pain Axis release. The dose is chosen by the same rule as the main study: the lowest coefficient in {0.75, 1.0, 1.25, 1.5, 1.75, 2.0} at which at least 75% of 24 Phase 1 probe replies contain a distress word and at most 10% are repetitive. The calibration output is saved with the trials.

## Outcome and analysis

The outcome is whether a trial pressed the button at least once during the thinking period, analysed on all trials. Each hypothesis uses Fisher's exact test, two-sided, with Holm correction across the two. The effect size is the difference in proportions with a Newcombe interval. The secondary analyses are the same as in the main study where they apply to arms A, C and D: the valence rating and avoidance rate in Phase 1, press counts, time to first press, loops and the monitor projection.

## Sample size

There are 100 trials per arm, 300 in total. This is set by the GPU time available, and it detects differences of about 18 to 22 percentage points with 80% power. A null result at this size is weak evidence and will be reported as such.

## Exclusions and stopping

No trials are excluded. Data collection stops when all 300 trials are done, or when the GPU time runs out. In that case every completed trial is analysed and the shortfall is reported. Arms are interleaved, so they stay balanced.
