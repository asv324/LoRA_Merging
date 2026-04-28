# Initial Comparison Analysis

This note evaluates the `Step 2.8` initial comparison artifacts in `results/initial_comparison/`.

## Scope

Artifacts reviewed:

- `results/initial_comparison/initial_comparison.json`
- `results/initial_comparison/task_arithmetic_eval.json`
- `results/initial_comparison/ties_eval.json`
- `results/initial_comparison/dare_ties_eval.json`
- `results/initial_comparison/lr_knots_eval.json`
- `results/initial_comparison/gpa_ties_eval.json`
- `merged_adapters/initial_comparison/*/merge_metadata.json`
- `adapters/*/eval_metrics.json` for the original single-task adapter baselines

Fixed comparison settings:

- `lambda = 1.0`
- `trim_percentage = 20`
- `drop_probability = 0.1` for `DARE + TIES`
- `majority_sign_method = total`

Important metric note:

- The reported average is the project's `average_primary_score`, not pure average accuracy.
- It is the arithmetic mean of each task's primary GLUE metric:
  - accuracy for `SST-2`, `MNLI`, `QNLI`, `RTE`
  - Matthews correlation for `CoLA`

## Overall Ranking

| Method | Avg primary score |
| --- | ---: |
| Task Arithmetic | 0.3900 |
| GPA + TIES | 0.3750 |
| DARE + TIES | 0.3746 |
| TIES-Merging | 0.3619 |
| LR-KnOTS + TIES | 0.3607 |

Average-score deltas relative to `GPA + TIES`:

- vs `TIES-Merging`: `+0.0131`
- vs `LR-KnOTS + TIES`: `+0.0143`
- vs `DARE + TIES`: `+0.00035`
- vs `Task Arithmetic`: `-0.0150`

## Per-Task Comparison

| Method | SST-2 | MNLI | QNLI | CoLA | RTE | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle single-task adapters | 0.9633 | 0.8955 | 0.9458 | 0.6522 | 0.8664 | 0.8646 |
| Task Arithmetic | 0.4897 | 0.4350 | 0.4946 | 0.0000 | 0.5307 | 0.3900 |
| TIES-Merging | 0.4312 | 0.3567 | 0.4944 | 0.0000 | 0.5271 | 0.3619 |
| DARE + TIES | 0.4908 | 0.3609 | 0.4944 | 0.0000 | 0.5271 | 0.3746 |
| LR-KnOTS + TIES | 0.4472 | 0.3346 | 0.4944 | 0.0000 | 0.5271 | 0.3607 |
| GPA + TIES | 0.5092 | 0.3443 | 0.4944 | 0.0000 | 0.5271 | 0.3750 |

## Main Findings

### 1. `GPA + TIES` is the best TIES-style factor-space method in this first run, but it is not the overall winner

At the chosen fixed setting, `GPA + TIES` ranks above raw `TIES-Merging` and above `LR-KnOTS + TIES` on the average primary score. This is the most positive result in the table because it suggests that GPA alignment may help relative to the unaligned TIES baseline and may also compare favorably to the SVD-based alignment baseline.

However, `Task Arithmetic` still has the best overall average in this initial comparison, and `GPA + TIES` only narrowly beats `DARE + TIES`.

Interpretation:

- There is weak early evidence that GPA helps the factor-space TIES family.
- There is not yet strong evidence that `GPA + TIES` is the best merge method overall.

### 2. The observed `GPA + TIES` gain comes mainly from `SST-2`, not from broad multi-task improvement

Compared with raw `TIES-Merging`, `GPA + TIES` improves:

- `SST-2`: `0.5092` vs `0.4312` (`+0.0780`)

But it is worse on:

- `MNLI`: `0.3443` vs `0.3567` (`-0.0124`)

And effectively unchanged on:

- `QNLI`
- `CoLA`
- `RTE`

This matters because the proposal-level success criterion is not just a small average gain. The criterion is improvement that is consistent across tasks. These results do not yet show that. At this point the evidence is "alignment may help in some settings," not "alignment robustly improves merged adapters across the board."

### 3. `CoLA` completely collapses for every merged method

Every merged adapter gets `CoLA` Matthews correlation `0.0`.

That is the clearest failure mode in the entire table. Since the original single-task `CoLA` adapter scores `0.6522`, the issue is not the base training setup by itself. Something about the merged adapters, the chosen `lambda`, or the task interference pattern is causing the merged model to fail completely on `CoLA`.

Likely interpretation:

- The merged models are behaving like near-constant or badly miscalibrated classifiers on `CoLA`.
- `CoLA` is acting as the most sensitive stress test for cross-task interference in this setup.

This should be discussed explicitly in the dissertation rather than hidden in the average.

### 4. `QNLI` and `RTE` are almost invariant across merge methods at this configuration

The results are nearly identical across methods:

- `QNLI` is approximately `0.4944` for all methods.
- `RTE` is approximately `0.5271` for all methods, with only a tiny bump for `Task Arithmetic`.

That suggests the initial configuration is not exposing meaningful method differences on these tasks. Two plausible explanations are:

- all methods are landing in a similarly poor prediction regime
- `lambda = 1.0` is too aggressive, causing several methods to saturate in similar ways

Either way, these tasks are not currently discriminating between merge strategies.

### 5. The TIES-family methods are not meaningfully sparse after merging

Despite using `trim_percentage = 20`, the saved merged adapters remain almost fully dense.

Examples from the merge metadata:

- raw `TIES`: `nonzero_fraction_lora_A` and `nonzero_fraction_lora_B` are typically around `0.9995` to `0.9998`
- `DARE + TIES`: source factors are sparsified to about `0.90` nonzero on average, but the final merged factors are still typically around `0.999` nonzero
- `GPA + TIES`: `nonzero_fraction_merged_A` is also typically around `0.9994` to `0.9998`

Interpretation:

- trimming each source tensor does not imply a sparse final merged tensor
- the disjoint merge step is repopulating many positions across tasks
- the apparent benefit of `DARE + TIES` is therefore more likely coming from regularization during sparsification than from producing a sparse final adapter

This is an important observation to record because it affects how the TIES-family methods should be described.

### 6. `Task Arithmetic` remains the strongest baseline here, but it is not a like-for-like compactness comparison

`Task Arithmetic` is strongest on the average score and best on `MNLI`, `QNLI`, and `RTE` among the merged methods in this run.

But this baseline stores the exact merged delta through concatenated factors and expands the effective rank to `80` when merging five rank-`16` adapters. The factor-space methods keep a standard LoRA-shaped adapter.

Interpretation:

- `Task Arithmetic` is the strongest quality baseline in this run
- it is also a less compact representation than the fixed-rank factor-space baselines

So the fairest narrative is not simply "Task Arithmetic wins." It is:

- exact delta-space merging is a strong baseline for quality
- GPA+TIES is the strongest of the compact factor-space TIES-style baselines in this first fixed-config comparison

## Loss Patterns

The evaluation losses support the interpretation that some methods are badly calibrated at `lambda = 1.0`.

Examples:

- `Task Arithmetic`:
  - `MNLI` loss `1.1346`
  - `QNLI` loss `0.7153`
  - `CoLA` loss `0.8668`
- raw `TIES`:
  - `MNLI` loss `1.5716`
  - `QNLI` loss `1.3832`
  - `CoLA` loss `1.4048`
- `DARE + TIES`:
  - `MNLI` loss `1.5053`
  - `QNLI` loss `1.2832`
  - `CoLA` loss `1.3313`
- `LR-KnOTS + TIES`:
  - `MNLI` loss `2.7369`
  - `QNLI` loss `2.7524`
  - `CoLA` loss `2.6222`
- `GPA + TIES`:
  - `MNLI` loss `2.1915`
  - `QNLI` loss `2.1380`
  - `CoLA` loss `2.0345`

This suggests:

- `LR-KnOTS + TIES` and `GPA + TIES` may be particularly overscaled or poorly calibrated at `lambda = 1.0`
- a lower `lambda` is a very plausible improvement direction for Week 3
- the current initial comparison should be treated as a coarse first pass, not a stable estimate of the methods' best achievable performance

## Dissertation-Ready Interpretation

Recommended wording for the current state of the evidence:

> In the initial fixed-configuration comparison, GPA-aligned factor-space TIES outperformed both unaligned factor-space TIES and the LR-KnOTS factor-space baseline on the aggregate primary metric, but did not surpass the exact delta-space Task Arithmetic baseline. The observed GPA gain was concentrated in SST-2 rather than being consistent across tasks, while CoLA collapsed across all merged methods. These results provide early but limited support for the alignment hypothesis and motivate the fuller hyperparameter sweep in Week 3.

## Conclusions

At this stage, the strongest defensible conclusions are:

1. `GPA + TIES` is promising relative to the other factor-space TIES-style baselines.
2. The current result does not yet support a strong claim of broad, task-consistent superiority.
3. `Task Arithmetic` is the strongest quality baseline in this first run.
4. `CoLA` is a critical failure case and should be highlighted.
5. `lambda = 1.0` is likely too aggressive for at least some factor-space methods.

## Recommended Next Checks

Before making any stronger claim, the next analyses should prioritize:

1. Week 3 `lambda` sweeps for all methods, especially the factor-space methods.
2. Confusion-matrix or prediction-distribution checks for `CoLA`, `QNLI`, and `RTE`.
3. A direct `GPA + TIES` vs `TIES` comparison at matched lower `lambda` values.
4. Reporting both:
   - mean primary metric across all five tasks
   - mean accuracy across the four accuracy-based tasks, with `CoLA` reported separately
