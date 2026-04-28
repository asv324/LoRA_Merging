# Results and Analysis Chapter Plan

## Purpose

This document outlines a proposed structure for Chapter 4, `Results and Analysis`, of the dissertation. It is a planning document rather than polished chapter prose. Each subsection below gives the intended argumentative role, the main talking points, and the figures or tables to keep in the main chapter versus move to the appendix.

The chapter should flow directly from Chapter 3's methodological claims:

- `C1`: GPA is a validated alignment primitive for low-rank factor spaces.
- `C2`: Alignment quality and merge quality are separable problems.
- `C3`: Scale imbalance, not rotational misalignment alone, is the binding constraint on factor-space LoRA merging.
- `C4`: Within compact rank-preserving factor-space merging, enhanced GPA+TIES is competitive with or modestly superior to direct baselines.
- `C5`: CoLA exposes a structural limit or stress case for compact factor-space merging.

The chapter should not present artifacts in filename order. It should follow the experimental logic from Chapter 3: synthetic validation, real-adapter structure, main empirical comparison, diagnostics and ablations, then synthesis against the claims.

## Important Source Note

The existing draft at `dissertation/chapters/results.md` appears to contain stale artifact paths and older result values in several places. For this plan, treat `dissertation/results_artifacts_manifest.json` and the generated tables in `dissertation/tables/results/*.csv` as the authoritative artifact inventory unless the chapter draft is deliberately revised later.

## Proposed Chapter Structure

### 4.1 Chapter Orientation and Evaluation Logic

Purpose:

- Reconnect the reader to the end of Chapter 3 before presenting any results.
- State that Chapter 4 evaluates the five claims `C1`-`C5`, not just a single average score.
- Re-state the formal success criterion from Section 3.4: the enhanced GPA+TIES pipeline must improve the average primary score over the strongest unaligned baseline by at least one percentage point, consistently across at least three of the five tasks.

Talking points:

- Explain the chapter flow: synthetic validation -> real adapter structure -> five-task empirical comparison -> CoLA and alignment diagnostics -> targeted ablations -> synthesis.
- Remind the reader that SST-2, MNLI, QNLI, and RTE use accuracy, while CoLA uses Matthews correlation.
- Make clear that Task Arithmetic is a strong empirical comparator but not a like-for-like rank-preserving baseline, because it preserves a higher effective rank than the compact factor-space methods.

Main-text artifacts:

- None. This should be a short orienting subsection.

Appendix artifacts:

- None.

Claim links:

- Frames all of `C1`-`C5`.

### 4.2 Synthetic Validation of the GPA Primitive

Purpose:

- Establish `C1` before interpreting real-adapter results.
- Show that the GPA primitive is mathematically and computationally credible in controlled settings.
- Separate "does GPA align matrices?" from "does alignment produce a good merged model?" so later downstream failures can be interpreted through `C2`.

Talking points:

- Begin with the clean rotation-recovery experiment: GPA recovers known orthogonal rotations in the low-noise regime relevant to rank-16 LoRA adapters.
- Move to convergence: GPA converges in very few iterations across the tested `N` and rank grid, supporting the computational feasibility claim from Chapter 3.
- Discuss non-orthogonal perturbations as the first robustness check: the real-adapter setting violates exact rotation assumptions, so graceful degradation matters more than perfect recovery.
- End with the structured LoRA-like experiment: dominant shared subspace preservation is the closest synthetic analogue to the downstream merge setting.
- Use these experiments to argue that later real-task weakness should not be attributed to GPA being an invalid alignment primitive.

Keep in main text:

- `dissertation/figures/results/synthetic/fig_04_01_synthetic_rotation_recovery.pdf`
  - Discuss as the direct test of `C1`.
  - Link to Chapter 3 Experiment 1 and Equation 3.2's GPA objective.
  - Use it to show that the alignment objective works when the ground-truth rotations are known.
- `dissertation/figures/results/synthetic/fig_04_04_synthetic_structured_overlap.pdf`
  - Discuss as the bridge from ideal rotations to LoRA-like shared structure plus task-specific perturbations.
  - Link to Chapter 3 Experiment 4.
  - Use it to support the claim that GPA can preserve the dominant shared subspace that downstream merging is supposed to exploit.

Move to appendix and reference in main text:

- `dissertation/figures/results/synthetic/fig_04_02_synthetic_convergence.pdf`
  - Discuss briefly in the main text as evidence that GPA is cheap enough to use inside the merging pipeline.
  - Link to `C1`'s computational feasibility component and Chapter 3 Experiment 2.
  - Place in the appendix because it supports the synthetic conclusion but is less central to the dissertation's main empirical argument.
- `dissertation/figures/results/synthetic/fig_04_03_synthetic_nonorthogonal_robustness.pdf`
  - Discuss briefly in the main text as evidence that GPA degrades smoothly when exact orthogonality is violated.
  - Link to `C1` and Chapter 3 Experiment 3.
  - Place in the appendix because the structured LoRA-like figure is the stronger bridge to real adapters.

### 4.3 Pre-Merge Adapter Structure and Scale Imbalance

Purpose:

- Bridge from synthetic assumptions to the real trained QLoRA adapters.
- Motivate `C3` before discussing downstream merging results.
- Show why a purely rotational story is incomplete: the adapters differ substantially in scale, module family, layer depth, and task-specific structure.

Talking points:

- Use the Week 1 adapter analysis as the empirical motivation for the scale-aware variants introduced in Section 3.2.7.
- Explain that MNLI and QNLI have larger overall norms, with especially large differences in the LoRA `B` factors.
- Emphasise that the imbalance is structured, not just global: attention and MLP modules differ, and early MLP layers show particularly strong task-dependent variation.
- Note that adapter magnitude is not a reliable proxy for standalone task quality, which weakens any argument for raw norm dominance during merging.
- Use this subsection to prepare the reader for why dGPA, saTIES, and `wB(alpha)` are tested later.

Keep in main text:

- `dissertation/figures/results/adapter_analysis/fig_04_05_adapter_norm_structure.pdf`
  - Discuss as the compact main-text summary of Experiment 5.
  - Link to `C3`.
  - Use it to show that scale imbalance is real, structured, and local enough that a single global rescaling would be inadequate.

Move to appendix and reference in main text:

- `dissertation/figures/results/adapter_analysis/adapter_norm_ranking.pdf`
  - Discuss as detailed evidence for task-level norm imbalance.
  - Link to `C3` and the motivation for dGPA and B-factor weighting.
- `dissertation/figures/results/adapter_analysis/adapter_norm_attention_vs_mlp.pdf`
  - Discuss as detailed evidence that scale imbalance differs by module family.
  - Link to `C3` and the per-module design of the GPA pipeline.
- `dissertation/figures/results/adapter_analysis/adapter_layer_heatmap_A.pdf`
  - Discuss as detailed evidence for layer-local heterogeneity in LoRA `A` factors.
  - Link to `C3` and Experiment 13's later residual heatmap.
- `dissertation/figures/results/adapter_analysis/adapter_depth_trends.pdf`
  - Discuss as detailed evidence for depth-dependent adaptation, especially early MLP structure.
  - Link to `C3`.
- `dissertation/figures/results/adapter_analysis/adapter_perf_vs_norm.pdf`
  - Discuss as evidence that larger adapter norm is not automatically higher adapter quality.
  - Link to `C3` by showing why raw magnitude should not be treated as a reliable importance weight.

### 4.4 Main Empirical Comparison on Five-Task GLUE Merge

Purpose:

- Present the main downstream test of `C4` and the formal success criterion.
- Establish the central empirical outcome before moving into diagnostics.

Talking points:

- Lead with the restored-head main results rather than the older fixed-configuration Week 2 comparison.
- `Task Arithmetic` is the strongest merged method overall, with an average primary score of `0.768`.
- `DARE + TIES` reaches `0.619`, making it a strong unaligned TIES-family baseline.
- `TIES-Merging` reaches `0.591`.
- `GPA + TIES` reaches `0.533`, `dGPA + saTIES + wB(0.5)` reaches `0.543`, and `LR-KnOTS + TIES` reaches `0.503`.
- Enhanced GPA improves over baseline GPA and LR-KnOTS, but not over raw TIES, DARE+TIES, or Task Arithmetic.
- This means `C4` is only partially supported in the narrow compact alignment-baseline sense.
- The formal success criterion is not met against the strongest unaligned baselines.

Keep in main text:

- `dissertation/tables/results/table_04_01_main_results.csv`
  - Discuss as the authoritative main result table.
  - Link to `C4` and the Chapter 3 success criterion.
  - Use the table to make the rank/fairness distinction explicit: Task Arithmetic is strong but not rank-preserving.
- `dissertation/figures/results/main_results/fig_04_06_main_method_comparison.pdf`
  - Discuss as the visual version of the main method comparison.
  - Link to `C4`.
  - Use it to make the relative ordering readable without forcing the reader to inspect all task columns.

Move to appendix and reference in main text:

- None for this subsection. The main comparison should stay central.

### 4.5 CoLA as a Structural Failure Mode

Purpose:

- Give `C5` dedicated attention rather than burying CoLA inside the average score.
- Explain why CoLA is the clearest stress test for compact multi-task adapter merging.

Talking points:

- Compare every merged CoLA score against the oracle CoLA score in `table_04_01_main_results.csv`.
- CoLA is not universally exactly zero in the latest restored-head table, so avoid overstating "complete collapse" as the final result.
- The right claim is more careful: CoLA remains far below the single-task oracle and is the most visibly fragile task under merging.
- The CoLA prediction distribution diagnostic should be used to assess whether low CoLA performance is merely a lambda/calibration issue or a broader structural limitation.
- Link this back to the Week 2 observation that earlier fixed-configuration results showed complete CoLA collapse, which motivated Experiment 8.

Keep in main text:

- `dissertation/figures/results/main_results/fig_04_11_cola_prediction_distribution.pdf`
  - Discuss as the main diagnostic for `C5`.
  - Link to Chapter 3 Experiment 8.
  - Use it to show whether any method or lambda regime recovers meaningful CoLA predictions.

Move to appendix and reference in main text:

- None. CoLA is important enough to keep the diagnostic figure in the main chapter.

### 4.6 Scale-Aware GPA Enhancement Ablation

Purpose:

- Test `C3` directly by asking which part of the enhanced GPA stack matters.
- Explain whether the bottleneck is pure rotation, scale-aware `A`-factor conflict resolution, or output-side `B` weighting.

Talking points:

- Directional GPA alone slightly hurts average score relative to baseline GPA (`-0.002`).
- Adding scale-aware TIES improves average score by about `+0.006` relative to dGPA+TIES.
- Moderate inverse-norm B weighting, `wB(0.5)`, gives the best GPA-family result, reaching `0.543` average and `+0.010` versus baseline GPA.
- Full inverse-norm weighting, `wB(1.0)`, helps CoLA and RTE more strongly but depresses other tasks enough that it is not the best average variant.
- This pattern supports `C3`: the useful signal comes from scale-aware merge/output handling more than from pure directional alignment alone.

Keep in main text:

- `dissertation/figures/results/ablations/fig_04_07_enhancement_ablation.pdf`
  - Discuss as the visual summary of Experiment 9.
  - Link to `C3` and `C4`.
- `dissertation/tables/results/table_04_03_enhancement_ablation.csv`
  - Discuss as the per-task GPA-family ablation table.
  - Link to `C3` by identifying which tasks benefit from scale-aware variants.

Move to appendix and reference in main text:

- `dissertation/tables/results/table_04_04_enhancement_contributions.csv`
  - Discuss as the detailed contribution decomposition behind Section 4.6.
  - Link to `C3`.
  - Place in the appendix because the main text can summarise the key increments without showing another table.

### 4.7 Alignment Diagnostics: Does GPA Improve Real-Adapter Geometry?

Purpose:

- Connect real-adapter geometry diagnostics to `C1`, `C2`, and `C3`.
- Test whether GPA alignment visibly increases similarity between real adapter factors.
- Explain why good synthetic alignment does not automatically imply strong downstream merging.

Talking points:

- Mean pairwise CKA changes only slightly after GPA, with `table_04_07_alignment_summary.csv` reporting a mean delta of about `+0.000172`.
- This makes the real-adapter alignment story subtler than the synthetic story: real adapters are not simply rotated copies of the same low-rank object.
- The residual-vs-B-norm-spread relationship is not globally monotonic: overall `rho=-0.021`, attention `rho=+0.337`, and MLP `rho=-0.427`.
- These mixed diagnostics support `C2` and qualify `C3`: scale is important, but the residual/norm relation depends strongly on module family.

Keep in main text:

- `dissertation/figures/results/alignment_analysis/fig_04_09_cka_before_after.pdf`
  - Discuss as the real-adapter counterpart to the synthetic validation experiments.
  - Link to `C1` and `C2`.
  - Use it to show whether GPA alignment materially increases pairwise factor similarity.
- `dissertation/figures/results/alignment_analysis/fig_04_10_layer_residual_heatmap.pdf`
  - Discuss as the module-level residual diagnostic.
  - Link to `C3` and Chapter 3 Experiment 13.
  - Use it to connect the pre-merge adapter structure from Section 4.3 to post-alignment residual patterns.
- `dissertation/tables/results/table_04_07_alignment_summary.csv`
  - Discuss as the numerical summary for CKA and residual correlations.
  - Link to `C1`, `C2`, and `C3`.

Move to appendix and reference in main text:

- None. The alignment diagnostics are central to interpreting why the downstream results are nuanced.

### 4.8 Task Arithmetic in Aligned Factor Space

Purpose:

- Use Experiment 12 as the sharpest diagnostic for `C2`.
- Ask whether alignment alone helps when the merge rule is simple Task Arithmetic, and whether enhanced alignment/weighting changes the picture.

Talking points:

- Plain GPA-aligned Task Arithmetic is essentially unchanged from unaligned Task Arithmetic, with average delta about `-0.001`.
- Enhanced-GPA-aligned Task Arithmetic rises to `0.802` average, a `+0.108` gain over the unaligned Task Arithmetic reference in `table_04_06_ta_aligned.csv`.
- The strongest gains are on CoLA and RTE, which makes this table important for interpreting `C5` and `C3`.
- This result supports `C2`: alignment alone is not sufficient, but scale-aware alignment/output handling can matter substantially when the method is not bottlenecked by compact rank-16 TIES recompression.
- This subsection should come after the alignment diagnostics because it explains why "alignment quality" and "merge quality" must be separated.

Keep in main text:

- `dissertation/tables/results/table_04_06_ta_aligned.csv`
  - Discuss as a main diagnostic table, not a minor appendix item.
  - Link to `C2`, `C3`, and `C5`.

Move to appendix and reference in main text:

- None. This table is too interpretively important to hide in the appendix.

### 4.9 N-Ablation: Does Multi-Way Alignment Scale?

Purpose:

- Test the multi-way component of `C4`.
- Ask whether GPA's simultaneous alignment becomes more useful as the number of merged adapters grows.

Talking points:

- `table_04_05_n_ablation.csv` shows that TIES remains strongest by average across the tested `N` values.
- Enhanced GPA is consistently slightly above baseline GPA, supporting a narrow GPA-family improvement.
- Enhanced GPA does not overtake TIES, and no clear pattern shows GPA's advantage growing with `N`.
- LR-KnOTS is generally lower than GPA variants in the generated summary, so the GPA-vs-LR-KnOTS comparison has some support, but the stronger multi-way scaling claim is not established.
- This weakens the strongest version of `C4` while preserving a narrower claim: scale-aware GPA is better than unenhanced GPA, but not enough to dominate unaligned baselines.

Keep in main text:

- `dissertation/figures/results/ablations/fig_04_08_n_ablation.pdf`
  - Discuss as the visual summary of Experiment 11.
  - Link to `C4`.

Move to appendix and reference in main text:

- `dissertation/tables/results/table_04_05_n_ablation.csv`
  - Discuss as the numerical backing for the N-ablation plot.
  - Link to `C4`.
  - Place in the appendix because the figure can carry the main trend and the table is mainly supporting detail.

### 4.10 Seed Variance and Robustness of the Headline Claims

Purpose:

- Avoid overclaiming small margins in the GPA-family results.
- Use the seed-variance table as a cautionary robustness diagnostic rather than the main empirical result.

Talking points:

- `table_04_02_seed_variance.csv` should be cited cautiously because its scale appears inconsistent with the restored-head main results in `table_04_01_main_results.csv`.
- Unless regenerated or reconciled, keep it in the appendix rather than making it a headline table.
- Use it only for the qualitative point that small GPA-family margins should not be overinterpreted without variance estimates.
- If this table is later regenerated to match the restored-head operating points, it can be promoted into the main text.

Keep in main text:

- None by default.

Move to appendix and reference in main text:

- `dissertation/tables/results/table_04_02_seed_variance.csv`
  - Discuss as an uncertainty and robustness caveat.
  - Link to `C4` because it calibrates whether the observed GPA-family margins are robust.

### 4.11 Synthesis Against C1-C5

Purpose:

- Close the chapter by answering the dissertation claims directly.
- Distinguish positive methodological findings from the negative or partial downstream result.

Talking points:

- `C1`: Supported by the synthetic experiments, especially `fig_04_01_synthetic_rotation_recovery.pdf` and `fig_04_04_synthetic_structured_overlap.pdf`; qualified by small real-adapter CKA changes in `fig_04_09_cka_before_after.pdf`.
- `C2`: Supported by the gap between strong synthetic alignment evidence and weaker compact downstream merging; strengthened by `table_04_06_ta_aligned.csv`, where plain aligned TA barely changes but enhanced aligned TA improves strongly.
- `C3`: Supported by adapter norm analysis, enhancement ablation, and B-weighting effects; qualified by mixed residual correlations in `table_04_07_alignment_summary.csv`.
- `C4`: Partially supported only in the narrow GPA-family and GPA-vs-LR-KnOTS sense; not supported against stronger unaligned baselines or the formal success criterion.
- `C5`: CoLA remains the strongest stress task and should be discussed as the clearest failure mode, but final wording should reflect the latest nonzero restored-head CoLA scores.
- End with the balanced thesis: GPA is a valid and informative low-rank alignment primitive, and scale-aware variants reveal useful structure, but compact factor-space merging remains the practical bottleneck.

Main-text artifacts:

- No new artifacts. This subsection should synthesise the earlier evidence rather than introduce new results.

Appendix artifacts:

- None.

### Appendix Implementation Details: Restored-Head Evaluation

Restored-head evaluation means that, when a merged adapter is evaluated on a GLUE task, the evaluation script restores that task's trained sequence-classification head from the original source adapter instead of leaving the base model with a freshly initialised random `score.weight`. This matters because the LoRA factors are not the whole task-specific model for sequence classification: the Week 1 adapters saved trained classifier heads via `modules_to_save`, and GLUE tasks can have different label spaces and head shapes, especially MNLI with three labels versus the binary tasks. Earlier merged adapters discarded these heads, which made evaluation depend partly on a random classifier hyperplane rather than only on the merged LoRA update. The restored-head implementation stores each source task head separately under the merged adapter's `classifier_heads/` directory and restores the matching head at evaluation time. This was chosen because there is no semantically meaningful single merged classifier head across heterogeneous GLUE tasks, while per-task restoration gives a fairer measurement of whether the merged LoRA body preserves task information.

## Artifact Placement Summary

### Main Chapter Figures

- `dissertation/figures/results/synthetic/fig_04_01_synthetic_rotation_recovery.pdf`
  - Placement: Section 4.2.
  - Claim link: `C1`.
  - Role: Direct ground-truth rotation recovery evidence.
- `dissertation/figures/results/synthetic/fig_04_04_synthetic_structured_overlap.pdf`
  - Placement: Section 4.2.
  - Claim link: `C1`.
  - Role: Synthetic bridge to LoRA-like shared structure plus task-specific perturbations.
- `dissertation/figures/results/adapter_analysis/fig_04_05_adapter_norm_structure.pdf`
  - Placement: Section 4.3.
  - Claim link: `C3`.
  - Role: Main summary of real-adapter scale imbalance and structured heterogeneity.
- `dissertation/figures/results/main_results/fig_04_06_main_method_comparison.pdf`
  - Placement: Section 4.4.
  - Claim link: `C4`.
  - Role: Visual summary of the main restored-head method comparison.
- `dissertation/figures/results/main_results/fig_04_11_cola_prediction_distribution.pdf`
  - Placement: Section 4.5.
  - Claim link: `C5`.
  - Role: CoLA diagnostic across lambda and prediction distributions.
- `dissertation/figures/results/ablations/fig_04_07_enhancement_ablation.pdf`
  - Placement: Section 4.6.
  - Claim link: `C3`, `C4`.
  - Role: Visual summary of the GPA enhancement ablation.
- `dissertation/figures/results/alignment_analysis/fig_04_09_cka_before_after.pdf`
  - Placement: Section 4.7.
  - Claim link: `C1`, `C2`.
  - Role: Real-adapter CKA before and after GPA.
- `dissertation/figures/results/alignment_analysis/fig_04_10_layer_residual_heatmap.pdf`
  - Placement: Section 4.7.
  - Claim link: `C3`.
  - Role: Layer/module residual diagnostic and residual/norm-spread interpretation.
- `dissertation/figures/results/ablations/fig_04_08_n_ablation.pdf`
  - Placement: Section 4.9.
  - Claim link: `C4`.
  - Role: Multi-way scaling test as `N` varies.

### Appendix Figures

- `dissertation/figures/results/synthetic/fig_04_02_synthetic_convergence.pdf`
  - Referenced from: Section 4.2.
  - Claim link: `C1`.
  - Role: Computational feasibility and iteration-count detail.
- `dissertation/figures/results/synthetic/fig_04_03_synthetic_nonorthogonal_robustness.pdf`
  - Referenced from: Section 4.2.
  - Claim link: `C1`.
  - Role: Robustness to non-orthogonal model mismatch.
- `dissertation/figures/results/adapter_analysis/adapter_norm_ranking.pdf`
  - Referenced from: Section 4.3.
  - Claim link: `C3`.
  - Role: Detailed task-level adapter norm imbalance.
- `dissertation/figures/results/adapter_analysis/adapter_norm_attention_vs_mlp.pdf`
  - Referenced from: Section 4.3.
  - Claim link: `C3`.
  - Role: Detailed attention-vs-MLP norm imbalance.
- `dissertation/figures/results/adapter_analysis/adapter_layer_heatmap_A.pdf`
  - Referenced from: Section 4.3.
  - Claim link: `C3`.
  - Role: Detailed per-layer LoRA `A` structure.
- `dissertation/figures/results/adapter_analysis/adapter_depth_trends.pdf`
  - Referenced from: Section 4.3.
  - Claim link: `C3`.
  - Role: Detailed adapter norm depth profiles.
- `dissertation/figures/results/adapter_analysis/adapter_perf_vs_norm.pdf`
  - Referenced from: Section 4.3.
  - Claim link: `C3`.
  - Role: Evidence that adapter norm is not a reliable quality proxy.

### Main Chapter Tables

- `dissertation/tables/results/table_04_01_main_results.csv`
  - Placement: Section 4.4.
  - Claim link: `C4`.
  - Role: Authoritative main restored-head method comparison and success-criterion test.
- `dissertation/tables/results/table_04_03_enhancement_ablation.csv`
  - Placement: Section 4.6.
  - Claim link: `C3`, `C4`.
  - Role: Per-task GPA enhancement ablation.
- `dissertation/tables/results/table_04_06_ta_aligned.csv`
  - Placement: Section 4.8.
  - Claim link: `C2`, `C3`, `C5`.
  - Role: Direct alignment-vs-scale diagnostic using Task Arithmetic in aligned factor space.
- `dissertation/tables/results/table_04_07_alignment_summary.csv`
  - Placement: Section 4.7.
  - Claim link: `C1`, `C2`, `C3`.
  - Role: Numerical summary of real-adapter CKA and residual correlation diagnostics.

### Appendix Tables

- `dissertation/tables/results/table_04_02_seed_variance.csv`
  - Referenced from: Section 4.10.
  - Claim link: `C4`.
  - Role: Variance and robustness caveat for small method margins; reconcile before using as headline evidence.
- `dissertation/tables/results/table_04_04_enhancement_contributions.csv`
  - Referenced from: Section 4.6.
  - Claim link: `C3`.
  - Role: Detailed contribution decomposition for dGPA, saTIES, and B weighting.
- `dissertation/tables/results/table_04_05_n_ablation.csv`
  - Referenced from: Section 4.9.
  - Claim link: `C4`.
  - Role: Numerical backing for the N-ablation figure.

## Draft Captions

### Figure Captions

- `dissertation/figures/results/synthetic/fig_04_01_synthetic_rotation_recovery.pdf`: Rotation-recovery performance of GPA on synthetic low-rank matrices generated by applying known random orthogonal rotations and Gaussian noise to a shared ground-truth factor. The heatmap is produced from the Experiment 1 sweep over noise level, rank, and number of observations, using the known rotations to compute recovery error.
- `dissertation/figures/results/synthetic/fig_04_02_synthetic_convergence.pdf`: GPA convergence behaviour on synthetic rotated low-rank matrices. The curves and summary statistics are generated by running the Experiment 2 convergence sweep across adapter counts and ranks, recording the GPA objective residual and iteration count for each configuration.
- `dissertation/figures/results/synthetic/fig_04_03_synthetic_nonorthogonal_robustness.pdf`: Robustness of GPA when the synthetic observations are no longer exact orthogonal rotations of a shared matrix. The data comes from Experiment 3, where controlled non-orthogonal perturbations are added before running GPA and measuring residual and rotation-recovery degradation.
- `dissertation/figures/results/synthetic/fig_04_04_synthetic_structured_overlap.pdf`: Dominant shared-subspace preservation in a LoRA-like synthetic setting with decaying singular values and task-specific perturbations. The figure is generated from Experiment 4 by comparing the GPA consensus subspace against the known shared ground-truth subspace using overlap and principal-angle metrics.
- `dissertation/figures/results/adapter_analysis/fig_04_05_adapter_norm_structure.pdf`: Summary of scale and structural imbalance across the five trained GLUE QLoRA adapters. The plotted values are computed from the Frobenius norms of extracted LoRA `A` and `B` factors across the 196 LoRA modules of each adapter.
- `dissertation/figures/results/adapter_analysis/adapter_norm_ranking.pdf`: Per-task ranking of average LoRA `A` and `B` factor norms for the five trained adapters. The data is obtained by extracting all LoRA modules from each task adapter and averaging Frobenius norms across modules.
- `dissertation/figures/results/adapter_analysis/adapter_norm_attention_vs_mlp.pdf`: Adapter norm decomposition by attention and MLP module families. The figure is generated by grouping extracted LoRA modules according to the project’s parameter mapping and averaging `A` and `B` Frobenius norms within each family.
- `dissertation/figures/results/adapter_analysis/adapter_layer_heatmap_A.pdf`: Layerwise heatmap of LoRA `A` factor norms across tasks and module groups. The data is produced by extracting per-module Frobenius norms from each trained adapter and arranging them by transformer layer and module family.
- `dissertation/figures/results/adapter_analysis/adapter_depth_trends.pdf`: Depth profiles of mean LoRA `A` norms across transformer layers. The plotted curves are generated by averaging extracted per-module norms by task, layer, and module family.
- `dissertation/figures/results/adapter_analysis/adapter_perf_vs_norm.pdf`: Relationship between combined adapter norm and standalone validation performance for each task adapter. The figure combines Frobenius-norm summaries from adapter extraction with each adapter’s standalone GLUE validation metric.
- `dissertation/figures/results/main_results/fig_04_06_main_method_comparison.pdf`: Main restored-head comparison of merging methods on the five GLUE tasks. The data is generated by evaluating each merged adapter at its selected best hyperparameters and plotting per-task and average primary scores from `results/main_results_restored_heads.json`.
- `dissertation/figures/results/ablations/fig_04_07_enhancement_ablation.pdf`: Ablation of the enhanced GPA+TIES components, including directional GPA, scale-aware TIES, and inverse-norm `B` weighting. The data is generated by evaluating each GPA-family variant at its best restored-head hyperparameter setting and comparing scores against baseline GPA+TIES.
- `dissertation/figures/results/ablations/fig_04_08_n_ablation.pdf`: Effect of varying the number of merged adapters from `N=2` to `N=5`. The figure is generated by merging selected task subsets with TIES, LR-KnOTS+TIES, GPA+TIES, and enhanced GPA+TIES, then averaging the primary scores across tasks in each subset.
- `dissertation/figures/results/alignment_analysis/fig_04_09_cka_before_after.pdf`: Pairwise CKA similarity between adapter `A` factors before and after GPA alignment. The data is produced by running GPA on the real trained adapters and computing CKA matrices over all task pairs for raw and aligned factor representations.
- `dissertation/figures/results/alignment_analysis/fig_04_10_layer_residual_heatmap.pdf`: Layerwise GPA alignment residuals and their relationship to module-level norm imbalance. The data is generated from the real-adapter GPA run by recording final per-module residuals and comparing them with `B`-factor norm spread across tasks.
- `dissertation/figures/results/main_results/fig_04_11_cola_prediction_distribution.pdf`: CoLA prediction behaviour across merge settings and scaling coefficients. The figure is generated during the restored-head hyperparameter sweep by logging CoLA scores and predicted-class distributions for each method and lambda configuration.

### Table Captions

- `dissertation/tables/results/table_04_01_main_results.csv`: Main restored-head GLUE results for the oracle adapters and all merging methods. The table is generated by selecting each method’s best restored-head hyperparameter configuration and reporting per-task primary metrics, average score, and selected merge hyperparameters.
- `dissertation/tables/results/table_04_02_seed_variance.csv`: Three-seed variance summary for the primary compact merging comparisons. The table is generated by rerunning selected methods across independent adapter/evaluation seeds and reporting the mean, standard deviation, and per-seed average primary scores.
- `dissertation/tables/results/table_04_03_enhancement_ablation.csv`: Per-task ablation results for the GPA-family variants. The table is generated by evaluating baseline GPA+TIES and each scale-aware enhancement combination, then reporting task scores, average score, and the delta from baseline GPA.
- `dissertation/tables/results/table_04_04_enhancement_contributions.csv`: Pairwise contribution decomposition for each GPA enhancement step. The table is computed from the enhancement-ablation results by subtracting adjacent variants to isolate the effect of directional alignment, scale-aware TIES, and inverse-norm `B` weighting.
- `dissertation/tables/results/table_04_05_n_ablation.csv`: Mean average primary score as the number of merged adapters varies from `N=2` to `N=5`. The table is generated by evaluating each method on fixed task subsets and reporting the mean and subset standard deviation at each merge size.
- `dissertation/tables/results/table_04_06_ta_aligned.csv`: Task Arithmetic results before and after applying GPA-based factor alignment. The table is generated by comparing unaligned Task Arithmetic with GPA-aligned and enhanced-GPA-aligned Task Arithmetic using the same evaluation protocol and reporting deltas against the unaligned reference.
- `dissertation/tables/results/table_04_07_alignment_summary.csv`: Summary statistics for real-adapter alignment diagnostics. The table is generated from the CKA and layer-residual analyses, reporting mean pairwise CKA before and after GPA plus correlations between alignment residuals and `B`-factor norm spread.

## Coverage Checklist

All generated figures from `dissertation/results_artifacts_manifest.json` are explicitly discussed above:

- `fig_04_01_synthetic_rotation_recovery`
- `fig_04_02_synthetic_convergence`
- `fig_04_03_synthetic_nonorthogonal_robustness`
- `fig_04_04_synthetic_structured_overlap`
- `fig_04_05_adapter_norm_structure`
- `adapter_norm_ranking`
- `adapter_norm_attention_vs_mlp`
- `adapter_layer_heatmap_A`
- `adapter_depth_trends`
- `adapter_perf_vs_norm`
- `fig_04_06_main_method_comparison`
- `fig_04_07_enhancement_ablation`
- `fig_04_08_n_ablation`
- `fig_04_09_cka_before_after`
- `fig_04_10_layer_residual_heatmap`
- `fig_04_11_cola_prediction_distribution`

All generated tables from `dissertation/results_artifacts_manifest.json` are explicitly discussed above:

- `table_04_01_main_results`
- `table_04_02_seed_variance`
- `table_04_03_enhancement_ablation`
- `table_04_04_enhancement_contributions`
- `table_04_05_n_ablation`
- `table_04_06_ta_aligned`
- `table_04_07_alignment_summary`

## Suggested Final Chapter Thesis

The chapter should report a nuanced outcome. GPA is well supported as a low-rank alignment primitive by the synthetic experiments, and the real-adapter analysis shows that scale imbalance is a genuine structural issue. However, compact factor-space GPA+TIES does not beat the strongest unaligned baselines under the restored-head main comparison, so the formal success criterion is not met. The most defensible conclusion is that alignment is useful and diagnostically revealing, especially when combined with scale-aware handling, but the practical bottleneck is the compact merge stage: scale competition, rank compression, and task-specific fragility remain unresolved.
