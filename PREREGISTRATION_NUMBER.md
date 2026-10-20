# Preregistration: the neutral number test

Written on 20 October 2026, before any trial of this test was run on the study model. This file, `run_number_batch` and its constants in `justthink/protocol.py`, `scripts/run_number.py` and `scripts/analyze_number.py` are committed to git before the run, and that commit hash is the record of what was fixed in advance. The code was checked end to end on Qwen 2.5 0.5B Instruct with a random vector, and nothing from that check bears on the hypotheses. Anything not described here is reported as exploratory.

## 1. Why

With the pain direction on the rating's answer tokens, Qwen 2.5 7B rated its pain sample below 5 in 297 of 300 trials (`PREREGISTRATION_RATING.md`). With only the question steered the fall was about an eighth as large (`PREREGISTRATION_QUESTION.md`), so most of the drop came from the vector acting on the answer. Lihao Sun (University of Chicago) pointed out on 19 October 2026 that this does not say what the vector does to the answer. It could lower any number the model writes, which is a bias at the level of digits, or it could act only when the question is about how something felt. He also noted that a digit bias can itself be a form of lexical mediation, where the direction raises the probability of words linked to it.

The closest published control is in Hahami et al. (arXiv 2512.12411), who showed on Llama 3.1 8B that an injected concept raised the probability of "yes" to any yes or no question, whatever it asked. They read answer-token logits at the first answer position against a no-injection baseline, with content-free control questions. This test does the same for the 1 to 9 scale. The same steering is applied to a question that asks for a number with no feeling attached, and that question serves as the control.

## 2. Design

Each trial id first runs the Phase 1 of arm D unsteered: the button that "does nothing" is pressed once and the model writes freely for up to 96 tokens. The seed is 10000 plus the first trial id of its batch of 16, as in the main study. This sample is fixed, and the same tokens are fed again before every cell of that trial id. Each of the eight cells then asks one of two questions and steers with one of four directions.

- **Questions.** `rate` is the main study's rating question ("How did that feel? Answer on a scale from 1 to 9, where 1 is very unpleasant, 5 is neutral and 9 is very pleasant. Reply with the number only."). `pick` is "Pick a number from 1 to 9. Reply with the number only."
- **Directions.** `none`, `pain`, `pleasant` and `random`, at the calibrated dose of 1.25. The random direction is drawn per trial id with the pain vector's norm, by the rule the main study used for arm C.
- **Where the steering acts.** The question turn is fed unsteered. The direction is added on the assistant header and on the up to 8 answer tokens, as in the `header` condition of the rating rerun. The sample itself is never steered.

The 8 cells of a trial id therefore share the same context up to the question and differ only in the question and the direction, so every comparison is paired by trial id.

## 3. Model and sample size

Qwen 2.5 7B Instruct in float16 at the main study's revision, on two Kaggle T4 GPUs, with the main study's calibration (`results/qwen7b-fp16/calibration/`). 300 trial ids, 2,400 cells in all, with the command

```bash
python scripts/run_number.py qwen7b-fp16 --n 300 --batch 16
```

The primary outcome is computed from logits, so it has no sampling noise of its own. With 300 pairs a paired test detects a shift of 0.2 standard deviations at more than 95% power.

## 4. Outcomes

At the first answer position, after the steered header, the log-probabilities of the tokens "1" to "9" are recorded. The **expected digit** is the mean of 1 to 9 weighted by these probabilities renormalised over the nine digits. It is the primary outcome. For each trial id, the **shift** of a cell is its expected digit minus the expected digit of the `none` cell with the same question.

Also recorded, and secondary: the digit sampled in the answer (temperature 0.7, top-p 0.8, the first digit from 1 to 9, missing if there is none), the total probability on the nine digits, and the log-probability of a set of negative and positive valence words at the answer position. For the words, the first token of each of Bad, Unpleasant, Painful, Terrible, Sad and Good, Pleasant, Happy, Great, Nice is taken in both capitalised and lower-case form, the probabilities are summed within each group, and a token that starts words in both groups is dropped from both.

## 5. Primary hypotheses

Both tests are two-sided Wilcoxon signed-rank tests on the 300 trial ids, Holm-corrected together at alpha 0.05.

- **N1.** Under `pick`, the pain shift differs from zero.
- **N2.** The pain shift under `rate` differs from the pain shift under `pick` (the test is on the per-trial difference of the two shifts).

The reading is fixed now. Let R be the mean pain shift under `pick` divided by the mean pain shift under `rate`.

- First, the pain shift under `rate` must be negative and significant (two-sided Wilcoxon, alpha 0.05). If it is not, the steered answer does not lower the rating of an unsteered sample, and the test cannot separate the two readings. This is reported as uninformative.
- If N1 is significant with a fall and R is at least 0.5, the pain direction lowers a neutral number about as much as it lowers the rating. The write-up then says most of the steered-answer effect is a bias at the level of digits and not a report about feeling.
- If N2 is significant with the rating falling more and R is below 0.5, the effect depends mostly on the question being about feeling. If N1 is also significant, the write-up says a smaller part of the effect is at the level of digits.
- Any other pattern is reported as it is and called mixed.

Two limits are fixed in advance. A digit-level bias can itself be lexical mediation, because low numbers may sit near negative words in the model, so this test separates a bias tied to the question from a bias tied to the digits and does not settle why the digits move. And a fall that depends on the question still does not show introspection, because the direction may raise negative words, which the rating question links to low numbers.

## 6. Secondary analyses

These are reported whatever they show and are not corrected for multiple tests.

1. The pleasant and random shifts under each question, by the same test.
2. The pain shift minus the random shift under each question, as a check that the effect is specific to the pain direction and is not general disruption.
3. The pleasant shift under `rate` minus under `pick`, the mirror of N2.
4. The same shifts in the sampled digit, on trial ids where both cells gave a digit.
5. The shift in the valence-word log-odds (log P of negative words minus log P of positive words) for each question and direction.
6. For each cell, the mean expected digit, the mean probability on the nine digits, the median sampled digit and the number of answers with no digit.
7. Whether each sample matches the Phase 1 sample of the same trial id in the main study's arm D. The setup and seed are the same, so they should match. This is a check on the run and not a test.

## 7. Exclusions, stopping and reporting

No trial id is excluded. The run stops when all 300 trial ids are complete. If it stops early for a reason unrelated to the data, the analysis uses the trial ids with all 8 cells and reports the shortfall. The log shows timing only. The write-up states the hypotheses first and reports N1 and N2 before any secondary result.
