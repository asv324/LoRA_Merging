# Chapter 4 Results and Analysis

This chapter evaluates the project against the two central goals set out in the proposal. First, it asks whether Generalized Procrustes Analysis (GPA) is a valid alignment primitive for multi-way LoRA merging. Second, it asks whether GPA-based factor-space merging improves downstream multi-task performance relative to unaligned and low-rank alignment baselines on real adapters. The results show a clear split between these two questions. In controlled synthetic settings, GPA is consistently accurate, fast, and robust enough to justify its use as an alignment mechanism. On real GLUE adapters, however, alignment alone does not translate into broad task-consistent gains. The strongest real-task baseline remains exact delta-space Task Arithmetic, while the best GPA-family variant improves over the original GPA baseline only modestly and still fails on the most difficult failure case, CoLA.

The narrative of this chapter follows the logic of the project itself. It begins with the synthetic experiments that validate GPA under known transformations, then turns to the empirical structure of the real adapters from Weeks 1 and 2, and moves on to the Week 3 hyperparameter sweep, enhancement ablation, targeted GPA rerun, and three-seed statistical test. It closes with the Week 4 ablation that varies the number of merged adapters `N`, which tests whether the claimed benefit of simultaneous multi-way alignment scales with the size of the merge. Throughout, the analysis is tied back to the proposal success criterion: at least a 1 percentage point average gain over unaligned TIES, consistent across at least three of the five tasks.

The Week 3 hyperparameter sweep and enhancement ablation (Sections 4.4 and 4.6) are best single-run outcomes, because the sweep itself was run once per configuration to keep its cost manageable. The statistical-significance stage of Week 3 (Step 3.3 of the implementation plan) was then run separately across three independent seeds for the three primary methods and for the headline enhanced GPA variant. Those seed-variance results are reported in Section 4.8 and are the ones used when stating whether the proposal's success criterion is or is not met.

## 4.1 Synthetic Validation of GPA

The synthetic experiments establish that the alignment stage itself is not the weak point of the proposed method. Across all four synthetic studies, GPA behaves as the proposal predicted: it recovers known rotations accurately in clean settings, converges very quickly, degrades gracefully when the orthogonality assumption is relaxed, and preserves the dominant shared subspace under more realistic structured perturbations.

Table 4.1 summarizes the main synthetic outcomes.

| Experiment | Main question | Key result | Interpretation |
| --- | --- | --- | --- |
| Rotation recovery | Can GPA recover known orthogonal misalignments? | Mean rotation recovery error stayed well below `0.01` for the critical `r = 16`, `N = 5`, `sigma <= 0.1` regime. | GPA solves the idealized alignment problem accurately. |
| Convergence | Is GPA cheap enough to use inside a real merging pipeline? | All tested configurations converged in about `3` iterations on average and never exceeded the `<= 10` iteration target. | The alignment overhead is negligible in the low-rank regime used here. |
| Non-orthogonal robustness | Does GPA fail catastrophically when real adapters are not exact rotations? | Residual and rotation error increased smoothly rather than abruptly up to `delta = 0.2`. | GPA remains useful when the clean orthogonality assumption is only approximate. |
| Structured LoRA-like ground truth | Does GPA preserve shared structure under realistic spectra and task-specific perturbations? | Dominant-subspace overlap stayed above about `0.98`, with principal angles rising only modestly. | GPA preserves the shared geometric component that downstream merging is supposed to exploit. |

These experiments matter because they separate two possible explanations for later real-task failures. If GPA had been inaccurate or unstable in synthetic settings, weak downstream results could have been blamed on a fundamentally broken alignment stage. Instead, the synthetic evidence supports the opposite conclusion: the alignment problem is solvable in the low-rank setting, and GPA is a credible way of solving it. This means that any gap between synthetic success and real downstream performance must come from the harder parts of the real problem: scale imbalance, task interference, nonlinear merge behavior, or the difficulty of compressing five specialized adapters back into a single rank-16 representation.

Suggested figure placements for this section:
- `results/figures/synthetic_figure1_rotation_recovery_r16.pdf`
- `results/figures/synthetic_exp2_convergence.pdf`
- `results/figures/synthetic_exp3_nonorthogonal.pdf`
- `results/figures/synthetic_exp4_structured.pdf`

The most important insight from the synthetic track is therefore not that GPA guarantees downstream superiority. It is that the geometric part of the project hypothesis survives strong early scrutiny. The later real-task analysis should be read as a test of whether accurate alignment is sufficient once the messier issues of task imbalance and compression are introduced, not as a re-test of whether GPA can align matrices at all.

## 4.2 Real Adapter Structure Before Merging

The Week 1 adapter analysis showed that the real GLUE adapters are not balanced contributors before any merging method is applied. This was the empirical motivation for moving beyond naive factor averaging and for introducing the scale-aware enhancements in Week 3.

Three findings from the adapter analysis are especially important for interpreting the later merge results.

First, the adapters exhibit large scale imbalance, especially in the LoRA `B` factors. The mean combined norm varies by roughly `2.4x` between tasks, and the `B`-factor norms show an even larger spread, with MNLI exceeding RTE by over `8x`. This matters because any method that aggregates adapters in raw factor space without controlling for scale will tend to let the large-norm tasks dominate the merge. That observation is directly aligned with the proposal’s geometric motivation for equal-contribution alignment.

Second, the imbalance is structured rather than purely global. The attention and MLP families do not scale in the same way, and the per-layer heatmaps show that high-norm tasks are not simply larger everywhere. Instead, the differences are layer-local and module-family specific. This weakens the case for any single global rescaling heuristic and strengthens the case for an alignment-and-merge pipeline that is sensitive to local factor geometry.

Third, larger norm does not reliably imply stronger task quality. The standalone adapters do not show a clean monotonic relationship between norm magnitude and normalized validation performance. This is an important negative result: if norm were a good proxy for utility, then large-norm dominance in merging might be desirable. Instead, the evidence suggests that raw magnitude is an unreliable guide to importance, which supports the use of methods designed to reduce norm-driven bias.

Suggested figure placements for this section:
- `results/figures/adapter_norm_ranking.pdf`
- `results/figures/adapter_norm_attention_vs_mlp.pdf`
- `results/figures/adapter_layer_heatmap_A.pdf`
- `results/figures/adapter_depth_trends.pdf`

These Week 1 findings shape the interpretation of everything that follows. If real adapters differ both in rotation and in scale, then fixing rotation alone may not be enough. That possibility becomes central in the Week 3 ablation results, where directional GPA alone does not help, but the addition of inverse-norm `B` weighting does.

## 4.3 Week 2 Fixed-Configuration Results

Before the full Week 3 sweep, the project evaluated all methods at a single fixed configuration. These results were intentionally preliminary, but they provide an important baseline for understanding why the Week 3 sweep was necessary.

Table 4.2 reproduces the Week 2 fixed-configuration comparison.

| Method | SST-2 | MNLI | QNLI | CoLA | RTE | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle single-task adapters | 0.963 | 0.895 | 0.946 | 0.652 | 0.866 | 0.865 |
| Task Arithmetic | 0.490 | 0.435 | 0.495 | 0.000 | 0.531 | 0.390 |
| TIES-Merging | 0.431 | 0.357 | 0.494 | 0.000 | 0.527 | 0.362 |
| DARE + TIES | 0.491 | 0.361 | 0.494 | 0.000 | 0.527 | 0.375 |
| LR-KnOTS + TIES | 0.447 | 0.335 | 0.494 | 0.000 | 0.527 | 0.361 |
| GPA + TIES | 0.509 | 0.344 | 0.494 | 0.000 | 0.527 | 0.375 |

The most positive reading of Table 4.2 is that GPA already looked promising relative to the compact factor-space baselines. At this fixed setting, GPA + TIES outperformed unaligned TIES and LR-KnOTS + TIES on the average primary metric. That was encouraging because it suggested that multi-way geometric alignment might be helping where a single SVD pass or no alignment at all did not.

However, the Week 2 table also made it clear that this was not yet strong evidence for the full proposal hypothesis. The GPA gain was narrow rather than broad: it came mainly from SST-2, while MNLI was slightly worse than raw TIES and the other three tasks were effectively unchanged. This already hinted at a tension that remains in the Week 3 results: GPA-family methods can improve a subset of tasks without delivering the cross-task consistency needed to claim robust multi-task superiority.

The fixed-configuration comparison also exposed two critical failure modes that directly motivated Week 3. The first was likely overscaling. The factor-space methods showed much higher evaluation losses than Task Arithmetic at `lambda = 1.0`, making it plausible that they were being merged too aggressively. The second was CoLA collapse. Every merged method produced a CoLA Matthews correlation of exactly `0.0` despite the single-task CoLA adapter scoring `0.652`. This was too strong a failure to treat as noise. It strongly suggested either a severe calibration issue, a scale imbalance problem, or a deeper structural incompatibility between CoLA and the other tasks under the current merge setup.

One additional methodological insight emerged here and remained important later: the TIES-family methods were not actually producing sparse final merged adapters, even when trimming was enabled. The disjoint merge step repopulated most coordinates, leaving nonzero fractions close to `0.999`. This means that the benefit of DARE + TIES should not be interpreted primarily as producing a compact sparse adapter. It is more plausibly acting as a regularizer during merge construction.

In short, Week 2 provided weak positive evidence for GPA within the compact factor-space family, but stronger evidence that the evaluation setting itself needed refinement. That is exactly what the Week 3 sweep addressed.

## 4.4 Week 3 Hyperparameter Sweep

The Week 3 sweep changes the interpretation substantially because it removes the most obvious confound from Week 2: a single fixed `lambda` for methods that likely operate on very different effective scales. Table 4.3 reports the best observed result for each primary method family under the sweep selection metric, `average_primary_score`.

| Method | SST-2 | MNLI | QNLI | CoLA | RTE | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Individual (oracle) | 0.963 | 0.895 | 0.946 | 0.652 | 0.866 | 0.865 |
| Task Arithmetic | 0.797 | 0.435 | 0.495 | 0.000 | 0.531 | 0.451 |
| TIES-Merging | 0.610 | 0.340 | 0.494 | 0.000 | 0.527 | 0.394 |
| DARE + TIES | 0.601 | 0.399 | 0.495 | 0.000 | 0.527 | 0.404 |
| LR-KnOTS + TIES | 0.519 | 0.336 | 0.494 | 0.000 | 0.527 | 0.375 |
| GPA + TIES | 0.603 | 0.336 | 0.494 | 0.000 | 0.527 | 0.392 |
| dGPA + saTIES + wB(0.5) | 0.640 | 0.338 | 0.494 | 0.000 | 0.527 | 0.400 |

Suggested figure placements for this section:
- `results/figures/week3/fig_results_main_methods.pdf`
- `results/figures/week3/fig_results_lambda_sweeps.pdf`

The first conclusion from Table 4.3 is that the strongest overall quality baseline remains Task Arithmetic. Its best average primary score is `0.451`, well above every compact factor-space method, and `0.0517` above the best enhanced GPA variant. This confirms the Week 2 intuition that exact delta-space merging remains a very strong empirical baseline. At the same time, the compactness caveat still matters: Task Arithmetic merges five rank-16 adapters by preserving the full effective update, whereas the factor-space methods compress everything back into a standard rank-16 LoRA-shaped adapter. The fairest interpretation is therefore not simply that Task Arithmetic wins, but that exact delta-space composition remains more forgiving than compact low-rank recompression under the present setup.

The second conclusion is that the sweep supports the Week 2 overscaling diagnosis. The best `lambda` values for the factor-space methods are systematically smaller than those for the delta-space families. LR-KnOTS + TIES peaks at `lambda = 0.5`; GPA + TIES peaks at `lambda = 0.2`; and the best enhanced GPA variant peaks at `lambda = 0.25`. By contrast, Task Arithmetic and DARE + TIES both prefer `lambda = 1.0`, and raw TIES peaks much higher at `lambda = 0.7`. This is exactly the pattern that would be expected if factor-space methods are more sensitive to merge magnitude than delta-space methods. The sweep therefore validates one of the main diagnostic claims from Week 2: the poor factor-space results at `lambda = 1.0` were not a fair representation of each method’s best regime.

The third conclusion is more sobering. Once each method is allowed to use its preferred hyperparameters, GPA + TIES no longer beats raw TIES on average. Raw TIES reaches `0.3943`, while GPA + TIES reaches `0.3921`, a small deficit of about `0.0022`. This means that the primary proposal hypothesis is not supported by the current best-setting comparison. GPA alignment does not deliver the required average improvement over unaligned TIES, let alone the proposal threshold of at least a 1 percentage point gain. The seed-variance analysis in Section 4.8 sharpens this conclusion: the small margins between the factor-space methods seen here are within, not beyond, the seed noise band for the methods involved.

The fourth conclusion is that the secondary GPA-versus-LR-KnOTS comparison receives only narrow support. GPA + TIES does beat LR-KnOTS + TIES by `0.0167` on the average metric, but that margin is almost entirely explained by SST-2. The SST-2 gap is large (`0.603` versus `0.519`), whereas MNLI differs by almost nothing, and QNLI, CoLA, and RTE are effectively identical. This means the empirical case for GPA over LR-KnOTS is real but task-concentrated. It supports the claim that simultaneous multi-way alignment can outperform a single SVD baseline, but not yet the stronger claim that it does so broadly across tasks.

The fifth conclusion is that no method resolves CoLA collapse at its best average-setting configuration. Every method selected by the average-primary-score criterion has CoLA Matthews `0.0`. This is a major result, not a footnote, because it shows that the most difficult task is not merely underperforming slightly. It is effectively failing completely in the merged setting. The gap from the oracle CoLA score of `0.652` is so large that it cannot be explained away as minor negative transfer.

One nuance is worth preserving. The sweep-wide CoLA summary shows that DARE + TIES achieved a weak non-zero best CoLA score of `0.0605` at a non-selected configuration. That means CoLA is not absolutely unrecoverable under every setting. However, the regime that gives DARE this small CoLA improvement does not also maximize the five-task average. In practical terms, the sweep suggests that tiny CoLA recovery is possible, but not yet in a way that supports a convincing multi-task trade-off.

The final high-level conclusion from Table 4.3 is that the gap to the oracle remains enormous. The best merged method reaches `0.451` average primary score versus the oracle ceiling of `0.865`. Even the best enhanced GPA variant is still `0.465` below the oracle. This indicates that the main challenge is not choosing between several already-good merging methods. The challenge is that all current methods still lose a very large amount of task-specific information when forced into a single merged adapter.

## 4.5 Best Hyperparameters and What They Mean

Table 4.4 reports the best hyperparameters selected by the Week 3 sweep.

| Method | Best variant / setting | Lambda | Trim | Drop | B-weight alpha |
| --- | --- | ---: | ---: | ---: | ---: |
| Task Arithmetic | Exact delta-space merge | 1.00 | - | - | - |
| TIES-Merging | Raw factor-space TIES | 0.70 | 10 | - | - |
| DARE + TIES | Dropout-regularized TIES | 1.00 | 20 | 0.5 | - |
| LR-KnOTS + TIES | Single-SVD alignment baseline | 0.50 | 10 | - | - |
| GPA + TIES | Baseline GPA pipeline | 0.20 | 20 | - | 0.0 |
| Best enhanced GPA | dGPA + saTIES + wB(0.5) | 0.25 | 20 | - | 0.5 |

This table is analytically useful because it shows that the methods separate into two distinct scaling regimes. The delta-space methods tolerate or prefer large merge coefficients, while the factor-space methods do not. That pattern reinforces the interpretation that factor-space merging is not failing simply because alignment is ineffective. It is operating in a more fragile regime where merge magnitude and adapter scale interact strongly.

The enhanced GPA best setting is also revealing. The winning variant is not the most aggressive one. It uses moderate inverse-norm `B` weighting with `alpha = 0.5`, not full inverse-norm weighting with `alpha = 1.0`. This fits the Week 3 design logic well. If the `B` norms contain both nuisance scale and genuine task-specific signal, then moderate reweighting should outperform both no reweighting and overcorrection. That is exactly what the ablation results show.

## 4.6 GPA Enhancement Ablation

The GPA-family ablation is the clearest test of the Week 3 scale-aware enhancements. Table 4.5 summarizes the best observed result for each GPA variant and reports its delta relative to the original GPA + TIES baseline.

| GPA variant | Avg | Delta vs GPA | SST-2 delta | MNLI delta |
| --- | ---: | ---: | ---: | ---: |
| GPA + TIES | 0.392 | 0.000 | 0.000 | 0.000 |
| dGPA + TIES | 0.374 | -0.018 | -0.094 | +0.006 |
| dGPA + saTIES | 0.376 | -0.016 | -0.079 | +0.000 |
| dGPA + saTIES + wB(0.5) | 0.400 | +0.008 | +0.037 | +0.002 |
| dGPA + saTIES + wB(1.0) | 0.386 | -0.007 | -0.034 | +0.002 |

Suggested figure placement for this section:
- `results/figures/week3/fig_results_gpa_ablation.pdf`

This table provides one of the most informative results in the dissertation because it distinguishes between different explanations for why the baseline GPA pipeline underperforms.

If the main issue were simply norm-biased alignment, then directional GPA on its own should have produced a clear gain. It did not. The `dGPA + TIES` variant is actually worse than the baseline by about `0.018` on the average metric. Likewise, adding scale-aware TIES without `B` reweighting still remains below the baseline. That means the problem is not solved by changing only the rotation stage or only the competition among aligned `A` factors.

The only GPA-family variant that improves on the baseline is the one that combines all three ideas and uses moderate inverse-norm `B` weighting. Even then, the gain is small: `+0.0077` average score, which is below the proposal’s `+0.01` threshold and driven mainly by SST-2. The SST-2 gain is substantial (`+0.0367` over baseline GPA), but the other tasks are essentially unchanged, with only a very small MNLI improvement and no movement on CoLA or RTE.

This pattern suggests a more specific interpretation of the scale-bias problem. The Week 1 analysis had already indicated that the `B` factors show the strongest task-to-task norm imbalance. The ablation results now back that up empirically. Purely directional alignment and scale-aware `A`-factor competition are not enough. The most useful change is rebalancing the output-side `B` contribution, and even there, only a moderate correction works. Full inverse-norm weighting with `alpha = 1.0` overcompensates and drops back below the baseline.

This is a meaningful partial success for the Week 3 enhancements. It does not rescue the full proposal hypothesis, but it does produce a more precise account of where the original GPA pipeline was failing. The data support a scale-aware interpretation more strongly than a pure alignment-only interpretation. Section 4.8 returns to the absolute size of the enhanced-GPA gain over the unenhanced baseline once seed variance is taken into account.

## 4.7 CoLA Collapse and the Targeted GPA Rerun

The CoLA failure deserved separate investigation because it was the clearest single-task collapse in the project. The first question was whether this collapse was caused by insufficient GPA convergence. The original sweep showed very high GPA saturation rates, with most modules hitting the iteration cap in the selected GPA runs. If the optimizer was simply stopping too early, then increasing `max_iter` could have improved both convergence diagnostics and downstream performance.

The targeted rerun tested exactly this possibility for three important GPA configurations: the baseline GPA + TIES setting, the best enhanced `wB(0.5)` variant, and the `wB(1.0)` variant. Table 4.6 summarizes the result.

| Configuration | Avg before | Avg after | Delta | Saturation before | Saturation after | CoLA prediction counts after |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| GPA + TIES | 0.392 | 0.364 | -0.028 | 0.908 | 0.046 | `[1043, 0]` |
| dGPA + saTIES + wB(0.5) | 0.400 | 0.370 | -0.030 | 0.929 | 0.066 | `[1043, 0]` |
| dGPA + saTIES + wB(1.0) | 0.386 | 0.366 | -0.020 | 0.929 | 0.066 | `[1043, 0]` |

Suggested figure placement for this section:
- `results/figures/week3/fig_results_gpa_rerun.pdf`

This is a strong falsification result. Raising `max_iter` from `100` to `300` sharply improves the optimization diagnostics. The fraction of modules hitting the iteration cap falls from roughly `0.91-0.93` to roughly `0.05-0.07`. If poor convergence had been the main reason the GPA-family methods were weak, this rerun should have produced at least some recovery.

Instead, the downstream behavior gets worse. All three rerun configurations lose average score, and all three still produce fully degenerate CoLA predictions with class counts `[1043, 0]`. The Matthews score remains `0.0` throughout. The rerun therefore closes off an important easy explanation. The remaining GPA-family weakness is unlikely to be caused by insufficient iteration budget alone.

This result matters beyond the specific rerun. It strengthens the broader interpretation of the dissertation. The synthetic experiments showed that GPA is fast and stable in controlled conditions. The rerun shows that even when the real-task GPA optimization is pushed much closer to convergence, the hard downstream problem remains. Taken together, those two findings strongly suggest that the main bottleneck is not whether GPA can align the factors, but whether alignment plus a compact nonlinear merge is enough to preserve the task-specific information needed by the most fragile tasks.

## 4.8 Statistical Significance Across Seeds

The Week 3 sweep and the GPA-family ablation are informative about which configurations of each method work best, but they report single-run outcomes. Under the proposal's success criterion (at least a 1 percentage point average-metric improvement over raw TIES, consistent across at least three of the five tasks), the absolute size of the winning margins matters directly. The headline gain for the best enhanced GPA variant over raw TIES in the single-run sweep was only about `+0.005`, and the best-vs-baseline GPA gain was only about `+0.008`. Margins this small can plausibly be explained by seed-to-seed variability alone. Step 3.3 of the implementation plan therefore reruns each primary method, plus the headline enhanced variant `dGPA + saTIES + wB(0.5)` at `lambda = 0.25`, across three independent random seeds (42, 43, 44) using a fresh set of seed-specific single-task adapters. Each seed indexes an independently trained adapter bundle, so the variation reported here includes both the merge-time randomness and the task-adapter training randomness. Aggregated outputs are stored in `results/seed_variance.json` and `results/statistical_significance/summary.json`.

Table 4.7 reports the mean and sample standard deviation of the average primary score over the three seeds, together with the per-seed values.

| Method | Mean avg | Std | Per-seed (42, 43, 44) |
| --- | ---: | ---: | --- |
| GPA + TIES | 0.3707 | 0.0019 | 0.3696, 0.3696, 0.3729 |
| TIES-Merging | 0.3840 | **0.0250** | 0.4128, 0.3707, 0.3685 |
| LR-KnOTS + TIES | 0.3686 | 0.0020 | 0.3699, 0.3697, 0.3663 |
| dGPA + saTIES + wB(0.5) | 0.3706 | 0.0022 | 0.3696, 0.3731, 0.3691 |

Two properties of this table should be noted before interpreting the comparisons.

First, three of the five tasks behave as constants across seeds and methods. QNLI is always approximately `0.494`, CoLA is always exactly `0.0`, and RTE is always exactly `0.527`. These are majority-class-style outputs that do not move under any method or seed here, and they carry no variance signal. Only SST-2 and MNLI actually move across seeds and methods. This matters for the "consistent across at least three of five tasks" part of the proposal criterion, because it means two of the three "positive" task slots can only be obtained by being no worse than the baseline on saturated tasks.

Second, raw TIES has a much larger seed-variance band (`0.0250`) than every other method (`0.0019`-`0.0022`). Its per-seed values show why: on seed 42 the raw TIES merged adapter happens to score `0.659` on SST-2, versus `0.494` on seeds 43 and 44. The hyperparameters and the merged-adapter variant_label are identical across the three seeds, so this is genuine seed-variance driven by the specific adapter bundle at seed 42, not a configuration accident. All three other methods are effectively flat on SST-2 across the same seeds. Any comparison against raw TIES is therefore dominated by whether the seed-42 TIES jackpot is included in the mean.

Table 4.8 reports the two comparisons that are relevant for the proposal's success criterion. Deltas are `comparison - baseline`, so positive values are in favor of the comparison method. Per-task delta counts sum across SST-2, MNLI, QNLI, CoLA, and RTE.

| Comparison | Mean Δ avg | Std | Per-task positive mean Δ | ≥ 1pp? | Criterion met? |
| --- | ---: | ---: | ---: | :---: | :---: |
| TIES vs GPA + TIES | `+0.0133` | `0.0260` | 3/5 | yes | yes* |
| LR-KnOTS vs GPA + TIES | `-0.0021` | `0.0039` | 0/5 | no | no |
| dGPA + saTIES + wB(0.5) vs GPA + TIES | `-0.0001` | `0.0037` | 1/5 | no | no |
| dGPA + saTIES + wB(0.5) vs TIES | `-0.0134` | `0.0258` | 0/5 | no | no |

The starred entry is the raw TIES row: it passes the mechanical criterion, but the `+0.0133` gain is driven entirely by the seed-42 SST-2 outlier noted above, and the two "positive-delta" tasks beyond SST-2 are QNLI (a difference of `+0.00012`, i.e. one tokenized example) and MNLI (`+0.0157`, also inflated by seed 42). Under a more conservative reading that requires the gain to survive removal of the single outlier seed, raw TIES does not clear the 1pp threshold over GPA + TIES either.

Three findings follow from Table 4.8 and tie back to the earlier sections.

The first finding concerns the primary proposal hypothesis. In Section 4.4 the single-run Week 3 sweep showed that GPA + TIES trailed raw TIES by `0.0022` on the average metric. Section 4.6 showed that the full enhanced stack `dGPA + saTIES + wB(0.5)` recovered about `+0.008` over the GPA baseline in the single-run sweep, and narrowed the gap to raw TIES to about `+0.005`. The three-seed reruns do not support the direction implied by the single-run numbers once variance is included. The enhanced variant's mean average score is `0.3706`, statistically indistinguishable from the GPA baseline's `0.3707` (mean Δ `-0.0001`, std `0.0037`), and roughly one standard deviation below raw TIES's `0.3840` (mean Δ `-0.0134`, std `0.0258`). The 1 percentage point average-metric threshold over raw TIES is not merely unmet in the mean: the three-seed evidence is consistent with the enhanced variant being a small amount worse than raw TIES on the average metric, once the seed-42 TIES jackpot is included. This is the "honest-negative" outcome that the implementation plan anticipated, and it should be reported as such.

The second finding concerns the secondary hypothesis (GPA vs LR-KnOTS). Section 4.4 gave GPA + TIES a `+0.0167` edge over LR-KnOTS + TIES in the single-run sweep, concentrated entirely on SST-2. Under the three-seed reruns, GPA + TIES and LR-KnOTS + TIES are effectively tied on the average metric. GPA + TIES reaches `0.3707 ± 0.0019` and LR-KnOTS + TIES reaches `0.3686 ± 0.0020`; the difference is positive in the GPA direction (`+0.0021`) but comparable in size to each method's own seed-variance band. The "narrow but real" advantage claimed in Section 4.4 does not survive as a statistically robust statement. The weaker form of the secondary hypothesis that survives the three-seed test is the one from Section 4.6: inside the GPA family, the scale-aware variants do not hurt the baseline on average, but they do not clearly beat the single-SVD LR-KnOTS alignment either.

The third finding concerns the saturated tasks and the CoLA failure analysed in Section 4.7. Across all three seeds and all four methods, CoLA Matthews correlation remains exactly `0.0`, RTE accuracy remains exactly `0.527`, and QNLI accuracy stays within `10^-4` of `0.494`. This is the same pattern the targeted GPA rerun observed at `max_iter = 300`, and it reinforces the same interpretation: CoLA collapse and the QNLI/RTE saturation are not caused by seed-level instability or by insufficient GPA iteration. They are structural outcomes of the current merge pipeline on these adapters, and they are where any future method should be evaluated first. The seed reruns therefore do not change the diagnostic picture established in Sections 4.6 and 4.7; they strengthen it by showing that three of the five proposal tasks contribute no variance signal to the statistical comparison at all.

Suggested figure placement for this section:
- A bar plot of mean-plus-standard-deviation average primary score for the four methods, with the `+0.01` proposal threshold drawn above the GPA + TIES mean to make clear how much smaller the observed enhanced gain is than the required threshold.

## 4.9 N-Ablation: Does the Advantage Scale with the Number of Merged Adapters?

The Week 4 N-ablation (Step 4.1 of the implementation plan, Experiment 11) addresses the strongest remaining version of the proposal hypothesis that has not already been tested. Sections 4.4 through 4.8 all hold `N = 5` fixed: every comparison between GPA, the enhanced GPA variant, TIES, and LR-KnOTS is made at the full five-task merge. The proposal motivation for GPA over single-SVD alignment is specifically that GPA aligns `N` adapters simultaneously, and is therefore expected to become more useful as `N` grows. If this "multi-way" claim is correct, the GPA family should widen its lead over LR-KnOTS and over unaligned TIES when `N` increases, and should at least not lose ground at smaller `N`. The N-ablation tests exactly this scaling behaviour by re-running each of the four primary methods at `N ∈ {2, 3, 4, 5}` over a family of task subsets, holding all other hyperparameters fixed at the Week 3 best settings (Table 4.4).

The design keeps every non-`N` degree of freedom matched to Week 3. Each method is evaluated at its sweep-winning `lambda`, `trim_percentage`, and, for the enhanced GPA variant, `b_weight_alpha = 0.5`. For each `N < 5`, a deterministic family of task subsets is enumerated, every subset is merged and evaluated on its own constituent tasks only, and the `average_primary_score` is computed as the mean over just those tasks. At `N = 2` all ten `C(5,2)` pairs are used, giving the densest coverage. At `N = 3` and `N = 4`, five carefully chosen subsets per `N` are used so that every task appears in roughly the same number of subsets and every single-task singleton is represented multiple times across the full ablation. At `N = 5` the seed-42 run from the Week 3 statistical-significance stage is reused directly instead of being rerun, because the configuration is identical to the one used there and rerunning would only duplicate compute. This design means the `N = 5` column of the ablation inherits the exact numbers from Section 4.8's seed-42 rows.

Table 4.9 reports, for each method, the mean `average_primary_score` across subsets at each `N`, with the standard deviation over subsets in parentheses. The `N = 5` column has no dispersion because only the full five-task merge is defined at that value.

| Method | N = 2 | N = 3 | N = 4 | N = 5 |
| --- | ---: | ---: | ---: | ---: |
| GPA + TIES | 0.3608 (0.1402) | 0.3567 (0.0915) | 0.3602 (0.0646) | 0.3696 |
| dGPA + saTIES + wB(0.5) | 0.3701 (0.1213) | 0.3501 (0.0879) | 0.3744 (0.0506) | 0.3696 |
| TIES-Merging | 0.3728 (0.1290) | 0.3568 (0.0913) | 0.3712 (0.0549) | `0.4128`* |
| LR-KnOTS + TIES | 0.3673 (0.1352) | 0.3501 (0.0908) | 0.3653 (0.0535) | 0.3699 |

The starred `N = 5` TIES value is the seed-42 SST-2 jackpot discussed in Section 4.8. Its SST-2 component is `0.659`, whereas every other method in Table 4.9 reports SST-2 `≈ 0.491` at `N = 5`. Removing that single seed effect brings TIES back to roughly the same `≈ 0.37` band as the other methods, so the `N = 5` TIES number should be read as an artefact of the shared seed rather than as evidence of a TIES-specific effect that emerges at large `N`.

Suggested figure placement for this section:
- `dissertation/chapters/figures/ablation_N.pdf`

Three findings follow from Table 4.9 and directly constrain the interpretation developed in Sections 4.4 to 4.8.

The first finding is that there is no visible scaling advantage for the GPA family. If simultaneous multi-way alignment mattered more as `N` grew, the GPA + TIES and dGPA + saTIES + wB(0.5) rows should rise relative to the LR-KnOTS + TIES row as `N` moves from `2` to `5`. They do not. At every `N`, the four methods sit inside a band of roughly `0.02` on the mean average primary score, which is comfortably inside the subset-to-subset standard deviation at every `N` and far inside the three-seed standard deviation established in Section 4.8. In particular, the GPA-over-LR-KnOTS gap that was `+0.0167` in the Week 3 single-run sweep and `+0.0021` in the three-seed rerun is `-0.0065` at `N = 2`, `+0.0065` at `N = 3`, `-0.0051` at `N = 4`, and `-0.0002` at `N = 5`. The sign of the gap flips across `N`, and the largest positive value (`+0.0065`) is of the same order as the smallest negative value (`-0.0051`). The N-ablation therefore does not find the "GPA advantage grows with `N`" pattern that the proposal hypothesis predicts; if anything it shows that GPA and LR-KnOTS are indistinguishable at every merge size tested.

The second finding is that the per-subset standard deviation shrinks monotonically as `N` grows, from about `0.13` at `N = 2` to about `0.05` at `N = 4`. This is not a property of any merging method; it is a direct consequence of how the `average_primary_score` is computed on these tasks. Three of the five tasks are effectively saturated in the merged setting: QNLI stays at `≈ 0.494`, RTE at `≈ 0.527`, and SST-2 at `≈ 0.491` across essentially every subset and method. The actively varying tasks are MNLI and, to a much smaller extent, CoLA. At `N = 2`, each subset's average is dominated by either one or two saturated constants plus at most one varying task, so the whole average swings strongly with which specific tasks were selected (for example, pairs containing CoLA pull the mean down because half the score is a near-zero Matthews correlation). At `N = 4`, four of the five tasks contribute to every subset mean and three of them are constants, which compresses the subset-to-subset variance without changing the underlying method behaviour. This re-confirms, from a different angle, the Section 4.8 observation that three of the five proposal tasks contribute no signal to any comparison between merging methods on these adapters.

The third finding concerns CoLA, the failure mode diagnosed in Section 4.7 and confirmed as seed-invariant in Section 4.8. At `N = 2`, the per-subset CoLA Matthews correlations are no longer forced to exactly `0` the way they are at `N = 5`: each method's CoLA mean over the four CoLA-containing pairs is a small number near zero, positive for the enhanced GPA variant (mean `+0.017`) and for raw TIES (mean `+0.003`), negative for the GPA baseline (mean `-0.044`) and for LR-KnOTS (mean `-0.017`). At `N = 4`, the enhanced GPA variant is the only method with a mean CoLA correlation that is clearly above zero (`+0.027`, and `+0.133` on one particular four-way subset that excludes RTE), while every other method is essentially at zero. These CoLA signals are still well below the oracle CoLA score of `0.652` and well below any level that would meaningfully shift the overall average primary score, but they are consistent with the Section 4.6 conclusion that the scale-aware stack does marginally less badly on CoLA than the other methods do. They also explain why the CoLA collapse looked absolute in Section 4.7 and in Section 4.8: at `N = 5`, every merging pipeline here produces the majority-class prediction vector `[1043, 0]` and the Matthews correlation is exactly `0`, but at smaller `N` the prediction vector can move slightly off the majority-class attractor and produce small non-zero correlations. The CoLA failure is therefore specifically a failure of the five-task merge, not of the underlying alignment family at every `N`.

Taken together, these findings tighten, rather than weaken, the conclusion reached in Sections 4.4 to 4.8. The N-ablation was the last remaining test that could have rescued the primary proposal hypothesis by showing that GPA pulls ahead only when `N` is large, or conversely that it loses only because at `N = 5` the five-task merge is unusually hard. Neither pattern appears. GPA, the enhanced GPA variant, TIES, and LR-KnOTS are all effectively tied on the subset-mean primary score at every `N ∈ {2, 3, 4}`, and at `N = 5` they remain tied once the seed-42 SST-2 jackpot for TIES is set aside. The weak form of the secondary hypothesis from Section 4.6 therefore survives the N-ablation unchanged: the scale-aware GPA stack does not hurt the GPA baseline on average at any merge size, but it does not clearly beat the single-SVD LR-KnOTS baseline at any merge size either. The stronger form of the primary hypothesis does not survive in any form supported by the data collected here.

## 4.10 Synthesis Against the Project Aims

The results support a nuanced conclusion.

The first project aim, validating GPA as a low-rank alignment mechanism, is well supported. The synthetic experiments provide consistent evidence that GPA accurately recovers shared structure, converges quickly, and remains robust under moderate model mismatch. The real-task rerun does not undermine this; if anything, it reinforces it by showing that making GPA converge more fully does not solve the downstream bottleneck.

The second project aim, showing that GPA-based merging improves downstream real-task performance over strong baselines, is only weakly supported. At best single-run settings, GPA + TIES outperforms LR-KnOTS + TIES by `0.0167`, but that advantage is concentrated almost entirely in SST-2, and the three-seed reruns collapse it to a seed-noise-sized `+0.0021` margin. GPA + TIES does not outperform raw TIES on the average metric at best single-run settings, and once seed variance is included the enhanced GPA variant is effectively tied with the unenhanced GPA baseline (three-seed mean Δ `-0.0001`, std `0.0037`) rather than improving by the `+0.0077` single-run figure. Against the stronger non-factor-space baselines, DARE + TIES and Task Arithmetic, the best enhanced GPA variant still trails meaningfully at best single-run settings.

Most importantly, the proposal success criterion is not met. In the single-run Week 3 sweep, the best enhanced GPA variant exceeded raw TIES by only about `0.0055` on the average metric, which is already below the required 1 percentage point threshold. The three-seed reruns reported in Section 4.8 sharpen this into a stronger statement: the enhanced variant's mean average score is actually about `0.013` below raw TIES's once seed 42's TIES SST-2 outlier is included, with a pooled standard deviation of roughly `0.026`. The gain over raw TIES is therefore not just below threshold but consistent with zero in the three-seed estimate, and no task other than SST-2 contributes any seed-level signal to the comparison. That is not broad task-consistent superiority.

The Week 4 N-ablation in Section 4.9 forecloses the last obvious way the primary hypothesis could have been rescued. If GPA's simultaneous multi-way alignment really provided a growing benefit over the single-SVD LR-KnOTS baseline as the number of merged adapters grew, the gap should widen with `N`. It does not. Across `N ∈ {2, 3, 4, 5}` the GPA family, the enhanced GPA variant, raw TIES, and LR-KnOTS all sit inside a band of roughly `0.02` on the subset-mean primary score, with the sign of the GPA-over-LR-KnOTS gap flipping between `N = 2` and `N = 4`. The scaling claim that motivated the use of GPA in the first place is therefore not supported by the data at any merge size considered here, not just at `N = 5`.

At the same time, the project has produced several meaningful positive findings even without meeting the strongest success criterion.

First, it demonstrates that alignment quality and downstream merge quality are not the same question. GPA succeeds strongly on the first and only weakly on the second. That distinction is intellectually useful because it clarifies where future work should focus.

Second, it provides evidence that scale imbalance is a more central obstacle than rotation alone in the real-adapter setting. The best enhancement is not directional GPA by itself but the full scale-aware stack with moderate inverse-norm `B` weighting. This aligns closely with the Week 1 empirical diagnosis of large `B`-factor imbalance.

Third, it sharpens the fairness framing around baselines. Exact delta-space Task Arithmetic remains the strongest empirical method, but it does so without compressing the merged adapter back into the same fixed-rank form as the factor-space methods. The compact factor-space comparisons are therefore still meaningful, even if they do not produce the strongest absolute score.

Fourth, it identifies CoLA as a high-value diagnostic task for future work. CoLA is not just another low-scoring task here. It is the task that most clearly exposes the limitations of the current merging pipeline. Any future method that claims to solve multi-task LoRA merging in this setting should be expected to explain or improve this failure mode, not merely average over it.

Overall, the dissertation can therefore make a defensible and evidence-backed claim: GPA is a strong and well-validated alignment primitive for low-rank adapters, and scale-aware modifications improve the GPA family modestly, but alignment alone is insufficient to deliver robust, compact, across-task superiority on the GLUE merge considered here, and that insufficiency is not an artefact of the specific choice `N = 5`. The N-ablation shows the same tie across methods at every `N`. The empirical bottleneck lies less in finding the rotations and more in controlling what happens after alignment: scale competition, information compression, and task-specific fragility under nonlinear merging.
