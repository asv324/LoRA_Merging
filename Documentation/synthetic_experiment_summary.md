# Synthetic Experiment Summary

## Purpose

This document consolidates the evidence from Track A of the revised implementation plan and explains how the synthetic experiments support the central research question in `Documentation/Research_Project_Proposal (1).pdf`: whether Generalized Procrustes Analysis (GPA) can align multiple LoRA adapters in low-rank space accurately, efficiently, and robustly enough to make downstream factor-space merging meaningful.

The proposal frames the core problem as one of misaligned low-rank coordinate systems. Independent LoRA adapters learn task-specific features in different rotated subspaces, so naive averaging causes destructive interference and representation collapse. The project hypothesis is that GPA can rotate the LoRA factors into a shared coordinate system while preserving the memory-efficiency benefits of operating directly on low-rank matrices. The synthetic experiments test that hypothesis before committing to the full training and evaluation pipeline.

## Experiment 1: Ground-Truth Rotation Recovery

Purpose:
Test the cleanest possible case first. If GPA cannot recover known rotations and a clean consensus when the data are generated exactly from rotated copies of a shared matrix, then the whole research direction is in trouble.

Link to the research:
This experiment validates the core mathematical objective from the proposal, namely minimising `sum_i ||Q_i A_i - C||_F^2` over orthogonal `Q_i`. It checks whether the estimated alignments are meaningful before any downstream merging is attempted.

Results:
The critical success criterion from the plan was satisfied: for `r = 16`, `N = 5`, and `sigma <= 0.1`, mean rotation recovery error stayed far below `0.01`. In the critical slice, the mean rotation recovery error was approximately `6.33e-06` at `sigma = 0.01`, `1.56e-04` at `sigma = 0.05`, and `6.30e-04` at `sigma = 0.1`. Consensus error and alignment residual also increased smoothly rather than abruptly.

Figure reference:
`results/figures/synthetic_figure1_rotation_recovery_r16.png`

What it shows:
The heatmap shows that GPA recovers the correct alignments almost perfectly at low noise and remains accurate well into the moderate-noise regime. Error only becomes clearly larger at `sigma = 0.5`, which is outside the proposal’s main target regime. This supports the claim that the GPA alignment step is mathematically sound in the idealized setting.

## Experiment 2: Convergence Curves

Purpose:
Establish that GPA is not only correct in principle, but also practical to run repeatedly as part of a LoRA merging workflow.

Link to the research:
The proposal argues that GPA is computationally feasible because it works on small `r x r` SVDs and should converge quickly. This experiment directly tests the convergence-rate claim from Section 2.5 of the proposal.

Results:
The baseline case at `sigma = 0.1`, `N = 10`, `r = 16`, `d = 1536` converged in `3` iterations with monotone residual decrease and wall-clock time of about `0.0135s`. Across the entire additional sweep over `N in {3, 5, 10, 20}` and `r in {8, 16, 32}`, every tested configuration converged in `3` iterations on average and never exceeded the plan’s `<= 10` iteration target.

Figure reference:
`results/figures/synthetic_exp2_convergence.png`

What it shows:
The baseline curve drops immediately and flattens by iteration `2-3`, while the sweep panel shows the same fast convergence behavior across all tested `(N, r)` combinations. This strongly supports the proposal’s claim that GPA is cheap enough to use as a practical alignment primitive in low-rank space.

## Experiment 3: Robustness to Non-Orthogonal Perturbations

Purpose:
Move beyond the idealized “pure rotation + noise” setting and test what happens when the orthogonality assumption is violated. This is the most important synthetic robustness check because real LoRA adapters will not be exact rotations of one shared matrix.

Link to the research:
The proposal’s geometric argument is built around alignment of rotated low-rank subspaces, but real adapters will also contain task-specific distortions. This experiment tests whether GPA degrades gracefully when that assumption is only approximately true.

Results:
The saved robustness check passed. Mean alignment residual increased gradually from about `0.1273` at `delta = 0.0` to `0.1443` at `delta = 0.2`. Mean rotation recovery error also rose smoothly, from about `6.36e-04` at `delta = 0.0` to `1.56e-02` at `delta = 0.2`. Importantly, there was no explosion in residual around `delta = 0.1`, which the plan identified as the likely real-adapter regime.

Figure reference:
`results/figures/synthetic_exp3_nonorthogonal.png`

What it shows:
The residual-vs-delta curve rises slowly through `delta = 0.1` and only steepens modestly at `delta = 0.2`. The rotation error curve is more sensitive, but still smooth. Together, these results suggest that GPA remains viable when the exact orthogonality assumption is relaxed, which is essential for the real LoRA setting.

## Experiment 4: Structured LoRA-Like Ground Truth

Purpose:
Test a more realistic synthetic model of LoRA adapters by combining a structured shared matrix with decaying singular values and task-specific rank-1 perturbations.

Link to the research:
This experiment is the strongest synthetic argument for the project because it asks whether GPA can preserve the dominant shared subspace even when adapters contain both shared and task-specific structure. That question is directly tied to the project’s intended downstream use: merge aligned factors in a way that keeps shared knowledge while limiting destructive interference.

Results:
After correcting the metric to evaluate the dominant shared subspace rather than the full row space, the shared-subspace check passed. The dominant subspace rank selected by the `0.9` energy threshold was consistently `4`. Mean dominant-subspace overlap remained very high across the whole sweep: about `0.9850` at perturbation strength `0.0`, `0.9846` at `0.05`, `0.9841` at `0.1`, and `0.9812` at `0.2`. Mean principal angle increased only modestly, from about `6.58` degrees to about `7.33` degrees.

Figure reference:
`results/figures/synthetic_exp4_structured.png`

What it shows:
The dominant shared subspace remains highly preserved even as task-specific perturbations increase. This is the strongest synthetic support for the dissertation’s core argument: GPA can isolate the shared geometric structure that should be preserved before factor-space merging.

## Overall Takeaways

1. GPA solves the idealized alignment problem accurately.
Experiment 1 shows that when adapters differ by clean orthogonal rotations plus moderate noise, GPA recovers the alignments and consensus with very small error.

2. GPA is computationally practical.
Experiment 2 shows that convergence is extremely fast in the tested low-rank regime, matching the proposal’s claim that alignment overhead should remain small.

3. GPA is robust to moderate model mismatch.
Experiment 3 shows that when the orthogonality assumption is violated, performance degrades gradually rather than catastrophically.

4. GPA preserves the dominant shared structure in a realistic synthetic setting.
Experiment 4 shows that even when adapters contain structured spectra and task-specific directions, the GPA consensus still captures the dominant shared subspace strongly.

## Relevance to the Dissertation

Taken together, these synthetic experiments provide early validation for the dissertation hypothesis that GPA is a viable low-rank, memory-efficient alignment method for multi-way LoRA merging. They do not yet prove that the full merging pipeline will outperform existing baselines on real downstream tasks, but they do show that the alignment phase itself is mathematically sound, fast to compute, robust to moderate deviations from the ideal assumptions, and capable of preserving the dominant shared structure that the later merging stages need.
