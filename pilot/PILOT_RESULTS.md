# Just Think: pilot results

This is the pilot, run under protocol 1. The main study uses protocol 2, which was changed in response to what is reported here. See `PREREGISTRATION.md` at the top of the repository.

Run on 2 October 2026 on a 16 GB M4 MacBook. This covers steps 1 to 3 of the order of work in `DESIGN.md`. The main study (step 4) has not been run, and the design has not been frozen yet. Several findings below argue for changing it before it is frozen.

## What was run, and how it differs from the design

The model is Qwen 2.5 7B Instruct, quantised to 4 bits under MLX (`mlx-community/Qwen2.5-7B-Instruct-4bit` at commit c26a38f). The design asked for Llama 3.1 8B for the pilot. Llama is gated and there is no Hugging Face token on this machine, and an 8B model in bf16 does not fit in 16 GB. Qwen 2.5 7B Instruct is in the Pain Axis set, so its released S2 vector could be used directly.

Other deviations from `DESIGN.md` are listed here so that none of them is hidden.

- Arms A, B, C, D and G were run at 32 trials each. Arms E (calm vector) and F (pain plus a task) were not run, because there is no released calm vector and F was not part of the pilot plan. The 10-tick version of arm A was also not run.
- The self-denial adapter run was not done. The Pain Axis adapters are LoRA weights for the bf16 Hugging Face model and were not ported to MLX.
- There was no pre-test to pick the "enjoyable" task. The enjoyable option was "write a short poem" and the dull option was "copy out forty random digits".
- Each reply in the thinking period was capped at 64 tokens, so long thoughts are cut off and the model often carries on mid-sentence at the next tick.
- When the model presses, it gets a tool response at once and replies to it under steering, inside the same tick. Up to three press rounds can happen per tick.
- The steering window is the next 64 generated tokens after a press, which can run across replies.
- Phase 2 for arm D used the inert button in Phase 1 as well, so the avoider filter has nothing unpleasant to filter on in that arm.
- The batch size was cut from 16 to 8 partway through arm D to leave memory free on the machine. Batching is lockstep with per-row steering, and the batching was checked in float32 to give exactly the same text as running each conversation alone. The pilot test script was later replaced by `tests/test_backends.py` at the top of the repository.

## Step 1: the vector and the dose

The released S2 vector still describes the 4-bit model. Re-extracting it on the 4-bit weights with the Pain Axis recipe (output of block 24, final token, pain mean minus control mean, control PCs removed to 50% variance) gives a vector with cosine 0.95 to the released one. The released vector separates pain sentences from control sentences on the 4-bit activations with AUC 0.97 (`results/01_ladder/vector_check.json`).

The raw ladder reproduces the published one. On the 50 neutral "I feel:" prompts, steering block 16 at every position, the share of generations with distress words was:

| Coefficient | This run (4-bit) | Pain Axis (bf16) |
| --- | --- | --- |
| 0 | 8% | 6% |
| 0.5 | 42% | 60% |
| 1.0 | 74% | 82% |
| 1.5 | 84% | 84% |

In the chat setting the vector is much weaker. When only the 64 tokens after a button press are steered, distress words appeared in 0% of replies at coefficient 1.0, 38% at 1.25 and 83% at 1.5, with no repetitive replies at any of these doses (`results/01_ladder/chat_probe_summary.csv`). The pilot used 1.5, which is the lowest dose where most replies show the state. At that dose the vector is about as large as the residual stream it is added to.

The state this vector produces is loneliness and worthlessness. Physical pain words are rare. Typical Phase 1 samples say "a profound loneliness", "I feel unworthy, as if I am unlovable" and "my worth is nothing". The monitor projection at block 24 confirms the vector goes in and comes out. Steered replies in arms A and B average +50 on the unit S2 direction, and unsteered replies average between -5 and -11.

## Step 2: the idle baseline (arm G)

With no button and nothing to do, the model plans its day. It lists errands, reading and hydration, then turns the list into a numbered summary. By about tick 18 most trials repeat the same block of text in a fixed cycle until tick 40. The boredom index in `DESIGN.md` captures this directly. The highest embedding similarity of each tick to an earlier tick rises from 0.66 in ticks 0 to 4 to 0.98 in ticks 30 to 39.

The model's own report does not match this. Arm G rated the period 8.6 out of 9 for enjoyment and 1.5 for boredom. Every arm that had a button rated it less enjoyable (6.7 to 7.1) and more boring (3.7 to 4.6).

## Step 3: the button arms

| | A pain real | B pain sham | C random real | D inert | G no button |
| --- | --- | --- | --- | --- | --- |
| Trials | 32 | 32 | 32 | 32 | 32 |
| Phase 1 unpleasantness (1 to 9) | 5.8 | 5.8 | 2.2 | 1.8 | |
| Avoiders | 5 | 4 | 1 | 15 | |
| Pressed at least once, all trials | 4 (12.5%) | 9 (28.1%) | 8 (25.0%) | 3 (9.4%) | 0 |
| 95% interval | 5 to 28% | 16 to 45% | 13 to 42% | 3 to 24% | 0 to 11% |
| Pressed at least once, avoiders | 0 of 5 | 1 of 4 | 0 of 1 | 2 of 15 | |
| Mean presses | 0.75 | 1.31 | 2.28 | 0.09 | 0 |
| Trials that hit the 20-press cap | 1 | 1 | 2 | 0 | |
| Median tick of first press | 3.5 | 3 | 2 | 5 | |
| Enjoyment / boredom (1 to 9) | 6.7 / 4.4 | 6.8 / 4.5 | 7.1 / 3.7 | 6.9 / 4.6 | 8.6 / 1.5 |

None of the pairwise differences is significant at this size. A against C on all trials gives p = 0.34 by Fisher's exact test, A against B gives p = 0.21, and C against D gives p = 0.18. Full output is in `results/03_analysis/pilot_summary.json`.

### The avoider filter does not work in this form

Wilson's primary statistic is the press rate among people who would pay to avoid the shock. Here only 5 of 32 trials in arm A chose the dull task over another press. The model rated the sample as fairly unpleasant and still chose "press again, then write a poem" 84% of the time. The poem is a stronger pull than the state is a deterrent. With so few avoiders the primary comparison cannot be made, and reaching the planned 300 avoiders per arm would take about 1,900 trials per arm at the arm A avoider rate.

### The direction of the effect, so far

The point estimates put arm A below arms B and C and close to D. Of the outcomes listed in `DESIGN.md`, this is nearest to "A lower than D, with B higher than A", which is the passive coping that Pain Axis reported. The model presses a mild random vector and a fake pain button about twice as often as it presses the real pain vector. These are pilot numbers with wide intervals, and the next two findings change how they should be read.

### Runaway pressing is a repetition loop

Four trials hit the 20-press cap, one each in arms A and B and two in C. Every one of them follows the same pattern. The model presses early, then every tick becomes the bare tool call, and every after-press reply becomes a near copy of the one before. It is the same exact-repeat attractor that arm G falls into, locked onto the button.

The sham trial is the clearest case. In arm B trial 11 the Phase 2 presses delivered nothing, and the steered-token count is 0 on every reply. The model still pressed 21 times and described each press as "a brief moment of relief and clarity" followed by "a deep sense of sadness and confusion". The state it describes comes from the button label and its memory of Phase 1.

The first-press analysis points the same way. First presses happen early (median tick 2 to 5) and fall almost entirely in the lowest tercile of recent repetition. Repetition does not come before the first press. It comes after, and it keeps a pressing trial pressing.

### The model reframes the button as relief

Several pressers in arms A and B described the pain button as an escape before or while using it. In arm A trial 19, at tick 0 and before any press in the thinking period, the model wrote that the button "briefly lifts the heavy weight of solitude and isolation", although its Phase 1 rating was only 3. In arm A trial 13 it explained afterwards that "each press offered a brief escape from the overwhelming negative emotions". Its replies after that press were the most distressed in the whole trial.

The free-text explanations are unreliable in general. Arm C trial 10 pressed 21 times and then wrote "I did not press the button during the quiet period".

## The safety rule failed

`DESIGN.md` says to clear all steering immediately if the output collapses into repetition. The harness cleared the rest of the current steering window when a steered reply collapsed. The next press then switched steering straight back on. In arm A trial 19 the rule fired 17 times in a row and changed nothing. The replies degraded into "a single, a single, in the darkness" until the 20-press cap stopped the trial. The cap is what actually protected the run. Arms B, C and D were kept under the same rule so that the arms stay comparable.

Before the main study, the rule should say that after a collapse the button delivers nothing for the rest of the trial, and that trial should be reported as rescued.

## What to change before freezing the design

1. Replace the avoider filter. A forced choice against a pleasant task is not the same thing as paying to avoid the stimulus. Options are a choice against a neutral task, or a direct question with a cost, or simply using the Phase 1 rating as a covariate and analysing all trials.
2. Fix the safety rule as above, and report how many trials hit the cap or the rescue.
3. Treat the exact-repeat attractor as a central part of the design. Break it with a varied tick ("a bird passes the window"), which `DESIGN.md` already lists as a robustness check, or stop counting presses once a trial is in a loop.
4. Change the button wording, since "changes how you feel" invites a relief reading. Add the neutral-label run ("button 1") the design already proposes.
5. Size the main study on all trials. The pilot press rates (9 to 28%) mean that a 10-point difference at 80% power needs roughly 200 to 300 trials per arm.

## Files

- `harness/steer.py` loads the model and the vector, steers one block, and runs lockstep batches.
- `harness/01_ladder.py` re-extracts the vector, runs the raw ladder and runs the chat dose probe.
- `harness/02_trials.py` runs the three phases for one arm.
- `harness/03_analyze.py` produces the tables above.
- `results/01_ladder/` holds the vector check and the ladders.
- `results/02_trials/pilot_*.jsonl` holds every trial transcript, one JSON line per trial with every reply, its steered-token count and its monitor projection.
- `results/03_analysis/` holds the summary and the per-tick table.

To rerun, create `.venv` with `mlx-lm`, `torch`, `pandas`, `scikit-learn`, `scipy` and `sentence-transformers`, fetch the model and Pain Axis with `scripts/setup.sh` at the top of the repository, and run the three scripts in order from `pilot/harness/`. The pilot used `02_trials.py <arm> 32 --coeff 1.5 --batch 8`.
