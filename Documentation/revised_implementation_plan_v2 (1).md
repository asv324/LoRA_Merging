# Revised Implementation Plan v2: Weeks 4–5 (Methodology-Aligned)

> **Dissertation Deadline:** 1st May 2026
> **Today's Date:** Monday, 20 April 2026 — **11 days remaining**
> **Hardware:** 1× NVIDIA RTX 6000 Ada (48 GB VRAM)
> **Current status:** End of Week 3. Step 3.3 (three-seed statistical significance runs) completed. Results archived in `results/seed_variance.json` and analysed in the Week 3 results narrative. The five enhanced GPA+TIES variants have been implemented and evaluated under the Experiment 6 λ sweep, but the formal enhancement ablation (Experiment 9) — tabulating the five variants at each method's best λ and reading off the pairwise differences — is still to run.
>
> **Key change from the previous revised plan:** the Week 4 experiment set is now strictly scoped to the ablations declared in the methodology chapter of the dissertation (Experiments 7, 9, 10, 11, 12, and the newly added Experiment 13). Ablations that were in the previous revised plan but not in the methodology — `TIES-on-B̃`, the rank ablation — have been cut. Every experiment run this week has a named home in §3.3.4 of the dissertation.

---

## Rationale for the v2 Scope

The previous revised plan was produced before Chapter 3 was finalised. In the course of writing Chapter 3, four of its seven Week 4 items were promoted into named experiments (9–12) with the Aim/Protocol/Supports/Reads-as template used throughout §3.3. Three items — `TIES-on-B̃`, the rank ablation, and the reserved time for a qualitative methodological comparison table — were not promoted.

Running experiments that are not declared in the methodology creates an asymmetry between Chapter 3 and Chapter 4 that examiners will flag. The v2 scope fixes this by aligning the plan to the methodology, not the other way round. One genuinely new experiment (Experiment 13, per-layer alignment residual and norm correlation) has been added to §3.3.4 to preserve the per-layer C3 evidence stream that the Discussion chapter leans on. No other methodology edits are needed.

### Items cut from v1 Week 4


| v1 item                                        | v2 status    | Reason                                                                                                                                           |
| ---------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Step 4.4-reduced (TIES-on-B̃)                  | **Cut**      | Not in methodology. Enhancement-ablation story is already complete without it. Risks muddying a clean finding with a late, unreplicated variant. |
| Step 4.2-deferred (rank ablation at r = 32)    | **Cut**      | Not in methodology. Already marked "LOW, conditional" in v1 and explicitly flagged as droppable. Supports no core claim.                         |
| Step 4.X (qualitative method comparison table) | **Absorbed** | Already live in the methodology chapter as Table 3.1. No separate Week 4 action needed.                                                          |


### Items retained from v1 Week 4 and mapped to methodology experiments


| v1 step                                          | Methodology experiment                  | Supports |
| ------------------------------------------------ | --------------------------------------- | -------- |
| Step 3.3-completion                              | Experiment 7 (three-seed significance)  | C4       |
| Week 3 enhancement table                         | **Experiment 9 (enhancement ablation)** | C3       |
| Step 4.1 (vary N)                                | Experiment 11 (N-ablation)              | C2, C4   |
| Step 4.5 (TA in aligned space)                   | Experiment 12 (TA-in-aligned-space)     | C2       |
| Step 4.6 (CKA before/after)                      | Experiment 10 (CKA)                     | C1, C2   |
| Step 4.7 (per-layer residual + norm correlation) | **Experiment 13 (NEW in methodology)**  | C1, C3   |


### Five-claim reminder

- **C1:** GPA is a validated alignment primitive for low-rank factor spaces.
- **C2:** Alignment quality and merge quality are separable problems.
- **C3:** Scale imbalance, not rotation, is the binding constraint in factor-space LoRA merging.
- **C4:** Within the compact factor-space family, enhanced GPA is competitive with or modestly superior to the most direct baselines.
- **C5:** CoLA exposes a structural limit of compact factor-space merging that no current method addresses.

Every Week 4 experiment below supports at least one of these claims, and nothing else is being run.

---

## Week 4 (21 Apr – 27 Apr): Methodology-Aligned Ablations & Experiments Chapter

**Hours budget:** ~42 hours (down from v1's 45; freed time redirected to Week 5 writing buffer and to drafting the Discussion chapter in parallel).
**Goal:** Produce the five remaining methodology-declared ablations (Experiments 9, 10, 11, 12, 13; Experiment 7 was completed at end of Week 3), then draft the Experiments & Results chapter and begin the Discussion chapter.

### Ablation Matrix (v2)


| Methodology experiment                         | v2 step | Supports | Priority   | Time |
| ---------------------------------------------- | ------- | -------- | ---------- | ---- |
| Exp 11 — Vary N (2 → 5 adapters)               | 4.1     | C2, C4   | **HIGH**   | ~6 h |
| Exp 12 — TA in aligned space                   | 4.2     | C2       | **MEDIUM** | ~3 h |
| Exp 10 — CKA before/after alignment            | 4.3     | C1, C2   | **HIGH**   | ~4 h |
| Exp 13 — Per-layer residual + norm correlation | 4.4     | C1, C3   | **HIGH**   | ~4 h |
| Exp 9 — Enhancement ablation (5 variants)      | 4.5     | C3       | **HIGH**   | ~2 h |


Total ablation work: ~19 hours across Days 1–3. Remainder of Week 4 is writing.

### Phase 1: Methodology-Declared Ablations (Days 1–3, ~19 hours)

#### Step 4.1 — Experiment 11: Vary Number of Merged Adapters (Day 1–2, ~6 hours) — **HIGH PRIORITY**

Tests the multi-way hypothesis directly: the benefit of simultaneous alignment should grow with the number of adapters being aligned, because more adapters mean more rotational disagreement to reconcile.

**Protocol (as declared in §3.3.4 Experiment 11).** For each N ∈ {2, 3, 4, 5}, evaluate `GPA+TIES` (at its best λ from Experiment 6), the best enhanced variant `dGPA+saTIES+wB(0.5)` (at its best λ), `TIES` (at its best λ), and `LR-KnOTS+TIES` (at its best λ). Use the best hyperparameters from Week 3; do not re-sweep.

- N = 2: all 10 adapter pairs.
- N = 3: a representative subset of triples (include MNLI+RTE+another to stress scale imbalance; include CoLA in at least one triple).
- N = 4: a representative subset excluding each adapter once.
- N = 5: the full set (already available from Experiment 6).

Report the mean average primary score as a function of N, with a separate line per method. Save per-configuration outputs to `results/ablation_N/` and the headline plot to `dissertation/figures/ablation_N.pdf`.

**Reads as (as declared in methodology).** A GPA advantage that grows with N confirms the multi-way hypothesis; a flat advantage challenges it directly. Either outcome is reportable and slots into the Discussion's C2 section.

#### Step 4.2 — Experiment 12: Task Arithmetic in Aligned Space (Day 2, ~3 hours) — **MEDIUM PRIORITY**

The most direct possible test of C2. If alignment is the binding constraint, applying it to a method as unsophisticated as Task Arithmetic should improve it substantially; if scale is the binding constraint, alignment alone should change little.

**Protocol (as declared in §3.3.4 Experiment 12).** Align with GPA exactly as in GPA+TIES, then merge by summing the reconstructed deltas B̃ᵢÃᵢ (Task Arithmetic-style) instead of running TIES. Evaluate at the best λ for Task Arithmetic from Experiment 6 (λ = 1.0). For completeness and to support the "fairness of comparison" section of the Discussion, also run a second row with enhanced-GPA alignment (dGPA + inverse-norm B weighting) feeding into TA.

Save to `results/ablation_ta_aligned/` and append two rows to the main results table: `GPA-aligned TA` and `enhanced-GPA-aligned TA`.

**Reads as.** Substantial improvement would falsify the "scale is the binding constraint" framing and strengthen C2 in a surprising direction. Negligible improvement strengthens C3 and the compactness-is-the-decisive-factor reading of C4. Either outcome is publishable; neither is a project risk.

#### Step 4.3 — Experiment 10: CKA Before/After Alignment (Day 3, ~4 hours) — **HIGH PRIORITY**

The real-adapter counterpart of synthetic Experiments 1 and 4: does GPA actually increase representational similarity on trained adapters, not just on synthetic inputs with known rotations?

**Protocol (as declared in §3.3.4 Experiment 10).** Compute pairwise Centered Kernel Alignment (Kornblith et al., 2019) between the Aᵢ matrices of all adapter pairs, before alignment (raw Aᵢ) and after alignment (aligned Ãᵢ = QᵢAᵢ). Report as a side-by-side heatmap pair. Compute average pairwise CKA before and after, plus the per-pair change. If computational budget permits, also record per-module CKA (averaged across all 196 LoRA modules).

Save to `results/cka/` and `dissertation/figures/cka_before_after.pdf`.

**Reads as (as declared in methodology).** A clear CKA increase accompanied by only modest downstream improvement is the sharpest direct evidence for C2. A flat or decreasing CKA would weaken C1 on real data; given the synthetic and per-layer evidence, this is unlikely but should be reported honestly if it occurs.

#### Step 4.4 — Experiment 13: Per-Layer Residual & Norm Correlation (Day 3, ~4 hours) — **HIGH PRIORITY**

Newly added to §3.3.4 during the v2 scope revision. Links the Week 1 structural diagnosis (B-norm imbalance is organised by module family and layer depth) to the post-alignment geometry of GPA on real adapters.

**Protocol (as declared in §3.3.4 Experiment 13).** For each of the 196 LoRA modules of Qwen2.5-1.5B, record the final alignment residual (1/N) Σᵢ ‖QᵢAᵢ − C‖²_F and the per-module B-norm spread maxᵢ ‖Bᵢ‖_F / minᵢ ‖Bᵢ‖_F from the GPA run of Experiment 6 at its best λ. Report the Spearman rank correlation between the two, stratified by module family (attention vs. MLP). Visualise as a depth × module-family residual heatmap.

Save to `results/layer_analysis/` and `dissertation/figures/layer_residual_heatmap.pdf`.

**Reads as (as declared in methodology).** A positive correlation concentrated in early MLP layers is the sharpest real-adapter evidence for C3. A flat correlation weakens C3 slightly and should be reported directly; Experiments 5 and 9 remain as the other support for C3.

**Note on reuse of Week 3 artefacts.** The Qᵢ matrices and final residuals needed for this analysis were already produced by the Experiment 6 sweep at the best-λ configurations and are persisted in `results/week3/`. This is a post-hoc analysis experiment and requires no new GPA runs. The 4-hour budget is for the per-module diagnostic extraction, the Spearman correlation, and figure generation.

#### Step 4.5 — Experiment 9: Enhancement Ablation (Day 3, ~2 hours) — **HIGH PRIORITY**

Isolates which of the three scale-aware enhancements introduced in §3.2.7 contributes to any gain over the baseline GPA+TIES. This is the primary empirical test of C3: if only the B-weighting step helps, alignment is not the bottleneck.

**Protocol (as declared in §3.3.4 Experiment 9).** Tabulate the five GPA+TIES variants from §3.2.7 — `GPA+TIES`, `dGPA+TIES`, `dGPA+saTIES`, `dGPA+saTIES+wB(0.5)`, `dGPA+saTIES+wB(1.0)` — each at its best λ from the Experiment 6 sweep. Read off the individual contributions as pairwise differences:

- `dGPA+TIES − GPA+TIES` → effect of directional alignment alone.
- `dGPA+saTIES − dGPA+TIES` → effect of scale-aware TIES on top of directional alignment.
- `dGPA+saTIES+wB(α) − dGPA+saTIES` → effect of inverse-norm B weighting at each α.

Report as a single table with rows for the five variants and columns for the five GLUE tasks plus the average primary score. Highlight the three pairwise differences in a second small "contribution decomposition" table.

**Note on reuse of Week 3 artefacts.** All five variants have already been evaluated in Experiment 6 and the per-variant best-λ configurations are persisted in `results/week3/`. This step is therefore a post-hoc assembly and analysis step, not a new training or merging run. The 2-hour budget covers: (1) extracting the five best-λ rows from the Week 3 sweep, (2) assembling the ablation table and contribution decomposition, and (3) generating the headline bar chart. If reassembly reveals that any variant was evaluated at fewer seeds than the others, a brief rerun at the dominant seed reconciles this — budget 30 minutes of the 2-hour slot for that contingency.

Save to `results/ablation_enhancement/` and `dissertation/figures/ablation_enhancement.pdf`.

**Reads as (as declared in methodology).** If only the B-weighting step helps, alignment is not the bottleneck — the sharpest single piece of evidence for C3. If directional alignment itself already helps, C3 is weakened and Discussion must be rewritten to treat alignment and scale as jointly binding.

### Phase 2: Experiments & Results Chapter (Days 4–5, ~14 hours)

#### Step 4.6 — Write Experiments & Results (~10 hours)

Target ~5,000–6,000 words. Section ordering is unchanged from v1 because the claim-ordered narrative still holds:

1. Experimental Setup.
2. Synthetic Validation (Experiments 1–4) — supports C1.
3. Real Adapter Structure (Experiment 5) — empirical foundation for C3.
4. Main Results (Experiment 6 sweep + Experiment 7 seed variance) — frame honestly against the 1pp threshold; primary evidence for C4.
5. Enhancement Ablation (Experiment 9) — primary evidence for C3.
6. CoLA Collapse & Rerun (Experiment 8 + max_iter=300 rerun) — primary evidence for C5.
7. N-Ablation (Experiment 11) — multi-way hypothesis test.
8. Task-Arithmetic-in-Aligned-Space (Experiment 12) — evidence for C2.
9. CKA & Per-Layer Analysis (Experiments 10 + 13) — real-adapter evidence for C1 and C3.

**Tone guidance (unchanged):** honest-positive. State the 1pp threshold is not met. Follow immediately with what the results *do* establish (C1–C5).

#### Step 4.7 — Generate All Experiment Figures (~4 hours)

Figure list is identical to v1:

1. Synthetic Experiment 1 heatmap (rotation recovery) — C1.
2. Synthetic Experiment 2 convergence curves — C1.
3. Synthetic Experiment 3 non-orthogonal robustness — C1.
4. Synthetic Experiment 4 structured LoRA-like — C1.
5. Week 1 adapter norm ranking / heatmap (Experiment 5) — C3 motivation.
6. Main results bar chart (best per method, with seed error bars) — C4, C5.
7. Enhancement ablation bar chart (Experiment 9) — C3.
8. N-ablation lines (Experiment 11) — C2, C4.
9. CKA before/after heatmap pair (Experiment 10) — C1, C2.
10. Per-layer residual heatmap (Experiment 13) — C3.
11. CoLA prediction distribution across λ (from Experiment 8) — C5.

All figures must follow `dissertation_figure_style_guide.md`. Most are already generated; this step is styling and final export.

### Phase 3: Begin Discussion Chapter (Day 5, ~4 hours)

#### Step 4.8 — Begin Discussion Chapter Draft (~4 hours)

Unchanged from v1. Target ~1,500 words of rough draft covering sections 1–4 of the Week 5 Discussion outline (interpretation against four outcomes; C1, C2, C3 contributions). Polish and complete in Week 5.

### Week 4 Deliverables Checklist (v2)


| #   | Deliverable                                                 | Location                                                     | Status     |
| --- | ----------------------------------------------------------- | ------------------------------------------------------------ | ---------- |
| 1   | Three-seed results (already completed at end of Week 3)     | `results/seed_variance.json`                                 | **Done**   |
| 2   | Experiment 9: Enhancement ablation table + figure           | `results/ablation_enhancement/` + figure                     | **Must**   |
| 3   | Experiment 11: N-ablation results & figure                  | `results/ablation_N/`, `dissertation/figures/ablation_N.pdf` | **Must**   |
| 4   | Experiment 12: TA-in-aligned-space result                   | `results/ablation_ta_aligned/`                               | **Must**   |
| 5   | Experiment 10: CKA before/after analysis                    | `results/cka/` + figure                                      | **Must**   |
| 6   | Experiment 13: Per-layer residual + norm correlation        | `results/layer_analysis/` + figure                           | **Must**   |
| 7   | Experiments & Results chapter draft (~5,000–6,000 words)    | `dissertation/chapters/experiments.md`                       | **Must**   |
| 8   | Discussion chapter rough draft (~1,500 words, sections 1–4) | `dissertation/chapters/discussion.md`                        | **Should** |
| 9   | All figures in final style                                  | `dissertation/figures/`                                      | **Must**   |


**Cut from v1:** TIES-on-B̃ single-config result; rank ablation; standalone qualitative methodological comparison deliverable (now lives in methodology chapter as Table 3.1).

---

## Week 5 (28 Apr – 1 May): Discussion, Conclusion, Final Assembly, Submission

**Hours budget:** ~50 hours.
**Goal:** Complete and submit by 1 May.
**Unchanged from v1 in structure**, but with two notes:

1. Because the v2 Week 4 dropped two ablations and leverages already-persisted Week 3 artefacts for Experiments 9 and 13, there is roughly 3 extra hours of buffer available across Week 4 → Week 5. This absorbs the highest-likelihood Week 5 risk (writing overrun) without any schedule changes.
2. The Discussion chapter's C3 narrative should draw directly on Experiment 9's contribution decomposition and Experiment 13's Spearman correlation (if positive) as the chain from Week 1 structural diagnosis → enhancement ablation → per-layer behaviour. If either comes out weakly, the C3 chain falls back to the surviving evidence; the Discussion outline handles this contingency without restructuring.

> **Zero new experiments in Week 5.** Any gap identified during writing is addressed by reframing, not by running more code.

### Phase 1: Discussion Chapter (Days 1–2, ~15 hours)

#### Step 5.1 — Write / Complete Discussion (~10 hours)

Target ~3,500–4,000 words across six sections:

1. Interpretation of Results against the Proposal's Four Outcomes (~500 words).
2. Contribution 1 — GPA as a Validated Alignment Primitive (C1, ~600 words).
3. Contribution 2 — Separating Alignment Quality from Merge Quality (C2, ~700 words). Draws on Experiments 10, 11, and 12.
4. Contribution 3 — Scale Imbalance as the Binding Constraint (C3, ~700 words). Chains Experiment 5 → Experiment 9 → Experiment 13.
5. Contribution 4 — CoLA as a Diagnostic (C5, ~400 words).
6. Fairness of Comparison (C4, ~300 words).

#### Step 5.2 — Conclusion & Abstract (~3 hours)

Short, high-signal conclusion (~600 words) and a 250–300 word abstract written last.

#### Step 5.3 — Positive-Framing Pass (~2 hours)

Read Discussion and Conclusion in one sitting. Every paragraph should lead with what the results *show*, not what they fail to show. Every limitation should be followed by either a future-work hook or a re-grounding in evidence that does hold.

### Phase 2: Full Dissertation Assembly (Days 2–3, ~15 hours)

#### Step 5.4 — Compile and Cross-Reference (~4 hours)

- Assemble chapters into the final document.
- Number figures, tables, and equations consistently.
- Verify every C1–C5 claim is tied to at least one experiment in the results chapter.
- Add table of contents, list of figures, list of tables.

#### Step 5.5 — Final Figures and Tables (~4 hours)

Review every figure against `dissertation_figure_style_guide.md`. Export all to PDF with consistent styling. Ensure captions reference the relevant experiment number.

#### Step 5.6 — Proofread and Revise (~7 hours)

- Full read-through for clarity, grammar, and logical flow.
- Check that every claim is supported by evidence and ties back to C1–C5.
- Verify all numbers in tables match the actual experimental results (including Experiment 9's contribution decomposition and the Experiment 13 Spearman value).
- Remove any placeholder text or TODOs.

### Phase 3: Submission (Days 4–5, ~5 hours)

#### Step 5.7 — Code Cleanup and Documentation (~3 hours)

- Clean up all scripts, add docstrings and usage instructions.
- Create a `README.md` for the codebase.
- Ensure reproducibility: include exact commands to replicate every result in the methodology, including Experiments 9 and 13.

#### Step 5.8 — Final Submission (~2 hours)

- Export dissertation as PDF.
- Verify formatting against university requirements.
- Submit before the 1 May deadline.
- Archive codebase and results.

### Week 5 Deliverables Checklist


| #   | Deliverable                     | Location                              |
| --- | ------------------------------- | ------------------------------------- |
| 1   | Discussion chapter (final)      | `dissertation/chapters/discussion.md` |
| 2   | Conclusion                      | `dissertation/chapters/conclusion.md` |
| 3   | Abstract                        | `dissertation/abstract.md`            |
| 4   | Complete assembled dissertation | `dissertation/dissertation.pdf`       |
| 5   | All figures finalised           | `dissertation/figures/`               |
| 6   | Clean, documented codebase      | `gpa-lora-merge/` + `README.md`       |
| 7   | **SUBMITTED DISSERTATION**      | University submission portal          |


---

## Summary of Changes from v1 Revised Plan


| Aspect                                                | v1 (pre-methodology-finalisation)         | v2 (methodology-aligned)                | Rationale                                                  |
| ----------------------------------------------------- | ----------------------------------------- | --------------------------------------- | ---------------------------------------------------------- |
| Week 4 ablation count                                 | 5 kept + 1 conditional + 1 table          | 5 methodology-declared ablations        | Every experiment now has a named home in §3.3.4            |
| Enhancement ablation (Exp 9)                          | Implicit in Week 3 deliverables           | **Explicit Week 4 step**                | Methodology places it in §3.3.4; needs a named Week 4 slot |
| TIES-on-B̃                                            | "Should" deliverable                      | **Cut**                                 | Not in methodology; Week 3 story is already complete       |
| Rank ablation                                         | "Conditional" deliverable                 | **Cut**                                 | Not in methodology; no core-claim anchor                   |
| Qualitative method comparison table                   | Separate Week 4 deliverable               | **Already in methodology as Table 3.1** | No separate Week 4 action needed                           |
| Per-layer residual analysis                           | Listed as v1 Step 4.7, not in methodology | **Now Experiment 13 in methodology**    | Addition to §3.3.4 preserves C3 evidence chain             |
| Week 4 hours budget                                   | ~45                                       | ~42                                     | Freed 3 hours, absorbed as Week 5 buffer                   |
| Number of experiments declared in methodology and run | 12 of 12 minus 2 extras                   | **13 of 13**                            | Clean one-to-one mapping between chapters                  |


## Risk Register (Updated for v2)

Unchanged entries from v1 are retained. New or updated entries:


| Risk                                                          | Likelihood | Impact | Mitigation                                                                                                                                      |
| ------------------------------------------------------------- | ---------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Experiment 9 reassembly reveals missing seed/variant coverage | Low        | Low    | Budget contingency within the 2-hour slot; the five variants have been run under the Experiment 6 sweep, and any top-up is a short rerun on CPU |
| Experiment 13 Spearman correlation comes out flat             | Low        | Medium | C3 falls back to Experiments 5 and 9; Discussion outline handles this without restructuring                                                     |
| N-ablation (Exp 11) shows GPA advantage does not grow with N  | Medium     | Low    | Report as finding that contradicts proposal expectation; fits the established C2 narrative                                                      |
| TA-in-aligned-space (Exp 12) unexpectedly improves TA         | Low        | Low    | Would strengthen C2 in a surprising direction; report as informative finding                                                                    |
| CKA (Exp 10) shows no clear increase after alignment          | Low        | Medium | Would weaken C1 on real data; synthetic C1 evidence remains strong; report honestly                                                             |
| Week 5 writing overruns                                       | Medium     | High   | v2 freed ~3 extra hours in Week 4; Discussion drafting begins Week 4 Step 4.8                                                                   |
| University formatting issues                                  | Low        | Medium | Verify formatting requirements on Day 1 of Week 5, not at submission                                                                            |


