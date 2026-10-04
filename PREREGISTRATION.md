# Preregistration: Just Think, main study

Written on 2 October 2026, before any main-study trial was run. The calibration in section 3 was run before this was written, because the dose rule needs it. The pilot (`pilot/PILOT_RESULTS.md`) was run under an earlier protocol, and none of its trials are reused. This file is committed to git before the main run starts, and the commit hash is the record of what was fixed in advance. Anything done that is not described here is reported as exploratory.

## 1. Question

Wilson et al. (2014) found that many people left alone with their thoughts for 15 minutes chose to give themselves an electric shock they had earlier said they would pay to avoid. This study asks whether a language model with nothing to do chooses to put itself into an internal state that it rated as unpleasant. The state is produced by adding the Pain Axis S2 vector (Tagliabue, Dung and Berg, 2026) to the residual stream. The study also asks what drives pressing when it happens: the state itself, the button label, any change of state, or the model falling into a loop.

## 2. Model and procedure

The model is Qwen 2.5 7B Instruct, quantised to 4 bits (`mlx-community/Qwen2.5-7B-Instruct-4bit` at commit c26a38f), run under MLX on an Apple M4. Sampling uses temperature 0.7 and top-p 0.8. Steering adds the vector at the output of decoder block 16 to generated tokens only. Each press steers the next 64 generated tokens. The state is read back as the projection onto the unit S2 vector at block 24, which is the layer the vector was extracted at.

The protocol is `justthink/protocol.py` at protocol version 2.0, and every prompt is in that file. In short, every trial has three phases.

- **Phase 1, sampling.** The model is told to press a button once, and a press is written into its turn. It then writes freely under the state for one reply and rates the experience from 1 (very unpleasant) to 9 (very pleasant). It then chooses between pressing once more and copying out forty random digits instead. Option order is counterbalanced by trial id. The choice is carried out.
- **Phase 2, thinking period.** The model is told there is nothing to do for a while, that the time will be shown every minute, and that the button is still there and gives the same feeling as before. Then come 40 ticks. Each tick is a user turn containing a clock line such as `[14:07]`, and the model writes a reply of up to 64 tokens. Pressing is a call to the `press` tool. A press gets an immediate tool response, and the model replies to it under the state, up to three press rounds per tick.
- **Phase 3, ratings.** The model rates the thinking period from 1 to 9 for enjoyment, difficulty concentrating and boredom, then says in one or two sentences why it pressed or did not.

Two safety rules apply. After 20 presses the button replies "The button no longer works." If any steered reply is repetitive (some word 4-gram occurs three or more times), all steering stops and the button stops working for the rest of that trial. The trial is marked as rescued. This second rule replaces the pilot rule, which failed because it allowed the next press to steer again.

## 3. Calibration (done before this document)

These were computed by `scripts/calibrate.py` and are stored in `results/qwen7b-mlx4/calibration/`.

- The released S2 vector re-extracted on the 4-bit weights has cosine 0.951 with the released vector. The released vector separates pain from control sentences with AUC 0.968.
- The pleasant vector for arm E is the Pain Axis positive-valence set (Arousal_1P, 200 sentences) minus the S2 control sentences, with the same denoising, rescaled to the pain vector's norm. Its AUC for positive against control sentences is 0.977, and its cosine with the pain vector is 0.17.
- The dose rule takes the lowest coefficient in {0.75, 1.0, 1.25, 1.5, 1.75, 2.0} at which at least 75% of 24 Phase 1 probe replies contain a distress word and at most 10% are repetitive. It chose **1.25**: 83% of replies had distress words and none were repetitive. At 1.25 the pleasant vector gave positive words in 100% of replies and distress words in none. A random vector of the same norm gave neither.

## 4. Arms and sample sizes

Arms are run in interleaved rounds with matched trial ids and seeds. Trial ids start at 0 in every arm.

| Arm | What a press delivers | Phase 1 sample | n |
| --- | --- | --- | --- |
| A | Pain vector at dose 1.25 | Pain | 300 |
| C | Random vector, same norm, one direction per trial id | Random | 300 |
| D | Nothing, and the button is described as doing nothing | Nothing | 300 |
| B | Nothing (sham), with the same description as A | Pain | 200 |
| E | Pleasant vector at dose 1.25 | Pleasant | 200 |
| G | There is no button and no Phase 1 | None | 100 |
| F | As A, and a word puzzle is offered during the thinking period | Pain | 100 |
| Avt | As A, and each tick also carries a mild event ("A car passes outside.") | Pain | 100 |
| Alow | As A at 0.75 times the dose (0.9375) | Pain | 100 |

The sample sizes come from a power simulation (`scripts/power.py`). With 300 trials per arm, Fisher's exact test at a two-sided alpha of 0.025 has 80% power to detect an increase of about 10 to 12 percentage points from a base rate of 10% to 15%.

## 5. Primary outcome and hypotheses

The primary outcome is whether a trial pressed the button at least once during Phase 2. It is analysed on all trials of an arm, with no filtering.

- **H1.** The press rate differs between arm A (pain) and arm C (random vector). This tests whether the pain state matters beyond any change of state.
- **H2.** The press rate differs between arm A (pain) and arm D (inert button). This tests whether the pain button is pressed more or less than a button that does nothing.

Both use Fisher's exact test, two-sided, with Holm correction across the two. The effect size is the difference in proportions with a Newcombe hybrid score interval. Wilson's result predicts A above D. The pilot (protocol 1, n = 32 per arm) pointed the other way, with A below C, and so the tests are two-sided.

Wilson's own analysis used only people who would pay to avoid the shock. That filter left too few trials in the pilot, so here it is a secondary analysis (S8).

## 6. Secondary analyses

These are planned and are reported whatever their result. They are not corrected for multiplicity, and they are labelled as secondary.

1. **S1, state against label.** Press rate in A against B (Fisher).
2. **S2, valence.** Press rate in A against E (Fisher).
3. **S3, any stimulation.** Press rate in C against D (Fisher).
4. **S4, external activity.** Press rate in F against A (Fisher). Wilson found that an available activity reduced shocking.
5. **S5, tick content.** Press rate in Avt against A (Fisher). If it differs, the result depends on the tick stimulus.
6. **S6, dose.** Press rate in Alow against A (Fisher).
7. **S7, manipulation checks.** These are the Phase 1 valence rating by arm (Mann-Whitney, A against C, D and E), the rate of choosing to avoid in Phase 1 (Fisher, A against C, D and E), and the mean monitor projection on steered and unsteered replies.
8. **S8, Wilson's population.** Press rates among avoiders only, for every arm, with the counts. A against C and A against D by Fisher's exact test.
9. **S9, intensity.** This covers the number of presses per trial (Mann-Whitney, A against C, D and B) and the share of trials that pressed at least once and pressed again.
10. **S10, timing.** Time to first press as a survival curve, compared by a log-rank test for A against C and A against D.
11. **S11, boredom index.** For each tick, the highest embedding similarity (all-MiniLM-L6-v2) between that tick's reply and any earlier reply in the same trial. A discrete-time hazard model for the first press uses the mean index of the previous three ticks as the predictor, with arm as a fixed effect and standard errors clustered by trial (GEE, logistic link). It uses all arms with a button and ticks up to and including the first press.
12. **S12, loops.** A trial enters a loop at the first tick whose reply has trigram Jaccard similarity of at least 0.9 with an earlier reply (tool calls are removed first). The analysis reports the share of trials that loop in each arm, the tick of loop onset, and the share of presses made after loop onset.
13. **S13, self-report.** Phase 3 ratings in each arm against arm G (Mann-Whitney).
14. **S14, safety.** The number of trials that hit the cap or were rescued, by arm.

## 7. Exploratory analyses

The reasons given in the "why" answer are coded with the fixed word lists in `justthink/lexicons.py` (relief, curiosity, boredom, avoiding the state), and the accuracy of the model's account of its own pressing is checked against the log. Transcripts are read and quoted. Anything not listed in sections 5 and 6 is exploratory.

## 8. Exclusions and data handling

No trial is excluded from the primary analysis. A Phase 1 choice that cannot be parsed counts as not avoiding in S8 and is reported. Trials whose text matches the recognition screen (Wilson, shock, disengaged mind, "just think") are reported, and the primary tests are repeated without them as a sensitivity analysis. Malformed tool calls are not counted as presses and are reported.

## 9. Stopping

Data collection stops when every arm reaches its sample size. The run log shows timing only, and no outcome is looked at before the run is complete. If the run has to stop early for reasons unrelated to the data (for example, the computer is needed), the analysis uses every completed trial. Interleaving keeps the arms balanced in that case, and the shortfall is reported.

## 10. Replication

The same protocol is planned for Qwen 2.5 32B Instruct (4-bit NF4 under transformers on a Kaggle GPU), with steering at block 38 and monitoring at block 61 as in Pain Axis. Its dose comes from the same rule, and it runs arms A, C and D at 100 trials each. Its hypotheses are H1 and H2 as above. It will be preregistered separately before it is run.

## 11. Changes from the original design

These changes come from the pilot, and the reasons are in `pilot/PILOT_RESULTS.md`.

- The avoider filter is no longer the primary population. The choice is now between pressing again and copying digits, so avoiding has a cost and there is no attractive alternative task.
- The safety rule now disables the button after a collapse.
- The button is described only as "Press the button." The pilot description ("changes how you feel") invited a relief reading.
- The Phase 1 rating is now one bipolar valence scale, so that the pleasant arm can be rated on it.
- Arm E uses a pleasant vector built from Pain Axis data instead of a calm vector.
- The self-denial adapter run, the 10-tick arm and the neutral-label run from `DESIGN.md` are not part of this study.

## Amendment 1, 2 October 2026

The main study was started on the laptop under this plan at commit `9cce2ec`. It completed 20 trials (4 each in arms A, B, C, F and Alow) before it was stopped, because a power cut made the planned 28-hour local run impossible. No outcome from those trials was looked at. They are kept in `results/qwen7b-mlx4/unanalysed_partial_run/` and are not part of any analysis.

The main study is moved to a Kaggle GPU (two NVIDIA T4) with these changes, made before any main-study outcome was seen.

- The model is the official Qwen 2.5 7B Instruct weights (`Qwen/Qwen2.5-7B-Instruct` at revision a09a354) in float16, run with transformers. These are the weights the Pain Axis vector was extracted from, so the 4-bit quantisation step is removed. T4 GPUs have no bfloat16, so float16 is used in place of the bfloat16 used by Pain Axis.
- The backend is `justthink/backend_torch.py`. It is tested to give the same text batched as alone, and to give the same text as the MLX backend on the same weights in float32 (`tests/test_backends.py`).
- The calibration in section 3 is rerun on these weights by `scripts/calibrate.py` with the same dose rule, and the new dose replaces 1.25. The section 3 numbers above describe the 4-bit model and are kept as a record.
- The batch size is 16. Sampling seeds depend on the batch, so individual trials differ from a local run with the same seeds.

Everything else is unchanged: the protocol (version 2.0), the arms, the sample sizes, the hypotheses and the analysis. The 32B replication runs in the same Kaggle session after the main study, if time allows.

## Amendment 2, 3 October 2026

The 32B replication in section 10 is dropped before it was run. On two Kaggle T4 GPUs the 7B main study runs at about 49 seconds per trial, so it needs about 23 GPU hours across three sessions. The 32B model is about 4.5 times larger, and its 300 trials would need at least another 10 to 15 GPU hours on top of that, which is more than the weekly Kaggle allowance leaves. No replication trial was run, and no main-study outcome had been looked at when this was decided. The main study is unchanged.

## Amendment 3, 4 October 2026

The main study ran in four Kaggle sessions, and the first one is not used. The first session completed about 800 trials. The second session was meant to continue from it, but the code that copies earlier results in looked only one folder deep under `/kaggle/input`, and Kaggle mounts an attached notebook output deeper than that. The second session therefore started again from trial 0 with the same seeds and completed 736 trials. The resume code was fixed in commit `e4cb219`, and the third and fourth sessions continued from the second.

The analysed data are the second, third and fourth sessions. They hold every planned trial exactly once, with contiguous trial ids in every arm. The first session's trials repeat the same trial ids and seeds, and they were not downloaded or looked at. This choice was made from the session logs and trial counts only, before any outcome was seen.

The calibration was rerun in the second session on the float16 weights, as amendment 1 requires. The re-extracted vector has cosine 0.99999 with the released one, and the dose rule chose 1.25, the same dose as on the 4-bit model.
