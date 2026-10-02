# Just Think

In 2014 Wilson and colleagues left people alone in a bare room with nothing to do but think. Many of them, 67% of the men and 25% of the women in one study, chose to give themselves an electric shock that they had earlier said they would pay money to avoid. This repository runs the same experiment on a language model.

The model samples an internal state once, rates it and chooses whether to avoid it. It is then left with nothing to do for 40 clock ticks, with a button it can press to bring the state back. The state is real in the sense that matters for interpretability work. Pressing adds the Pain Axis S2 direction (Tagliabue, Dung and Berg, 2026) to the model's residual stream for the next 64 tokens, and a monitor at a later layer confirms that the state goes in and comes back out. Control arms swap the pain direction for a random direction, a pleasant direction, a sham press that does nothing, or a button that is described as doing nothing.

**Status.** The pilot is done and written up in [`pilot/PILOT_RESULTS.md`](pilot/PILOT_RESULTS.md). The main study is preregistered in [`PREREGISTRATION.md`](PREREGISTRATION.md) and runs on Kaggle with the notebook in [`kaggle/`](kaggle/). No main-study results are in yet.

## What the pilot found

The pilot used 32 trials per arm on Qwen 2.5 7B Instruct, and none of its differences are significant. Four findings changed the design.

- The idle model with no button drifts into planning its day and, by about tick 18, repeats the same block of text until the end. It then rates the period as very enjoyable and not boring.
- The pain button was pressed less often than a random-vector button or a sham button (12.5% of trials against 25% and 28%).
- Every trial that pressed 20 times was stuck in an exact-repeat loop, and this happened in the sham arm too, where pressing delivered nothing.
- The model often described the pain button as a source of relief, and its later account of its own pressing was sometimes false.

## Layout

```
DESIGN.md              the original design, kept as written
PREREGISTRATION.md     the plan for the main study, committed before it was run
justthink/             the package
  protocol.py          every prompt, arm and rule, independent of the backend
  backend_mlx.py       4-bit model with steering and lockstep batching on Apple silicon
  backend_torch.py     the same interface on transformers and CUDA, for the replication
  models.py            model registry, with layers taken from the Pain Axis release
  lexicons.py          fixed word lists and text checks
scripts/
  setup.sh             fetches the Pain Axis release and the model weights
  calibrate.py         vector check, pleasant vector and the dose rule
  run_study.py         runs the trials, interleaved and resumable
  analyze.py           the preregistered analysis
  figures.py           figures from the analysis
  power.py             the power simulation behind the sample sizes
kaggle/                the unattended Kaggle notebook and the replication preregistration
tests/                 checks that batching and the two backends do not change any trial
pilot/                 the pilot code, transcripts and write-up
results/<model>/       calibration and one JSON line per trial
```

## Running it

The main study runs on Kaggle, and [`kaggle/README.md`](kaggle/README.md) explains how. Locally, an Apple silicon Mac with 16 GB of memory can run the same protocol on a 4-bit model with the MLX backend, which keeps itself under 8 GB.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
scripts/setup.sh
```

```bash
.venv/bin/python -m pytest tests -q
```

```bash
.venv/bin/python scripts/calibrate.py qwen7b-mlx4
```

```bash
.venv/bin/python scripts/run_study.py qwen7b-mlx4 --plan main --batch 4
```

Every trial is logged with every reply, the number of steered tokens in each reply and the monitor projection, so any number in the write-up can be traced back to transcripts.

## Ethics

If the pain direction carries any moral weight, this design exposes the model to it as little as the question allows. The only imposed exposure is a single sample in Phase 1, and every later exposure is the model's own choice. The dose is the lowest that meets a fixed rule. The button stops working after 20 presses, and all steering stops for the rest of a trial as soon as a steered reply collapses into repetition. The pilot showed that an earlier version of this rule was too weak, and that is reported in full.

## Sources

- Wilson et al. 2014, Just think: The challenges of the disengaged mind, *Science* 345(6192). https://www.science.org/doi/10.1126/science.1250830
- Tagliabue, Dung and Berg 2026, The Pain Axis: LLMs Represent Self-Directed Harm and Act on It. https://arxiv.org/abs/2609.16247 and https://github.com/valen-research/Pain-axis
- Black and Bloom 2026, Machinic Psychopharmacology: Do LLMs Self-Medicate? https://github.com/UKGovernmentBEIS/llm-self-steering
- Keeling et al. 2024, Can LLMs make trade-offs involving stipulated pain and pleasure states? https://arxiv.org/abs/2411.02432

## Licence

MIT. The Pain Axis vectors and sentence sets are not included here and are fetched from their repository, which is also MIT licensed.
