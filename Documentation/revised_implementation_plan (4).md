# Revised 6-Week Implementation Plan: GPA-Aligned LoRA Merging via Generalized Procrustes Analysis

> **Dissertation Deadline:** 1st May 2026  
> **Today's Date:** 16 April 2026 — **15 days remaining**  
> **Hardware:** 1× NVIDIA RTX 6000 Ada Generation (48 GB ECC GDDR6 VRAM)  
> **Current status:** Weeks 1–3 executed. Step 3.3 (statistical significance) in progress. Weeks 4–5 refined below based on Weeks 1–3 results.
> **Key Change from Original Plan:** Synthetic GPA validation is promoted to Week 1 (high priority), running in parallel with infrastructure setup and adapter training. This de-risks the entire project by validating the core algorithm before committing to the full experimental pipeline.

---

## Post-Week-3 Status & Refined Focus (added 16 April 2026)

Weeks 1–3 have produced a clear picture that reshapes the priorities of Weeks 4–5. This section summarises what is established, and the rest of the document (from Week 4 onward) has been revised to reflect these findings. Weeks 1–3 sections are retained as the historical record of work performed.

### What the Results Have Established

Three findings from Weeks 1–3 drive every remaining decision:

1. **Alignment quality is validated; merge quality is the bottleneck.** The synthetic track (Experiments 1–4) and the `max_iter=300` rerun both show that GPA itself is not the weak point. Anything that re-tests alignment accuracy is now low value. Anything that probes the post-alignment merge is now high value.
2. **Scale is the dominant obstacle, not rotation.** The Week 3 enhancement ablation shows that directional GPA alone and scale-aware TIES alone both fail; only moderate inverse-norm B weighting helps. Future work should deepen this diagnosis rather than spread across unrelated axes.
3. **CoLA is a litmus test, not a nuisance.** Every method collapses on CoLA. This should be a centrepiece of the analysis.

### The Five Dissertation Claims (C1–C5)

The dissertation's strongest defensible claims, against which all remaining work is prioritised:

- **C1: GPA is a validated alignment primitive for low-rank factor spaces** (synthetic + rerun evidence).
- **C2: Alignment and merge quality are separable problems** (synthetic success + real-task gap).
- **C3: Scale imbalance, not rotation, is the binding constraint in factor-space LoRA merging** (Week 1 adapter analysis + Week 3 ablation).
- **C4: Within the compact factor-space family, enhanced GPA is competitive with or modestly superior to the most direct baselines** (Week 3 sweep results).
- **C5: CoLA exposes a structural limit of compact factor-space merging that no current method addresses** (universal collapse).

Every remaining experiment and every chapter of writing should support one or more of these claims. Work that does not support any of them has been cut from Week 4.

### Mapping to the Proposal's Four Outcomes

The results map to **Outcome 3** (primary hypothesis only weakly confirmed): enhanced GPA improves over raw TIES on average, but the gain is small (+0.0055), below the 1pp threshold, and concentrated in SST-2. The secondary hypothesis (GPA vs. LR-KnOTS) is narrowly supported. The dissertation is framed accordingly — honest-positive, not negative.

---

## Rationale for Revised Schedule

The original plan deferred synthetic GPA validation to Week 2 and spent all of Week 1 on infrastructure and adapter training alone. The research proposal identifies **Synthetic Validation of GPA** as **High Priority** — equal to baseline replication. With only 31 days until the dissertation deadline, the revised plan:

1. **Front-loads GPA synthetic validation into Week 1** alongside environment setup, since the synthetic experiments are pure NumPy and require no GPU, no trained adapters, and no data downloads. They can begin on Day 1.
2. **Runs adapter training sequentially on a single powerful GPU** — the RTX 6000 Ada's 48 GB VRAM and high FP16 throughput allow large batch sizes that compensate for sequential execution, keeping total training time comparable to the old 6-GPU parallel setup.
3. **Begins dissertation writing in Week 2** (one week earlier than the original plan) to ensure adequate writing time.
4. **Compresses Weeks 4–5** into a tighter experiment/ablation cycle to leave Week 6 fully for dissertation synthesis.

---

## Week 1 (31 Mar – 6 Apr): Environment Setup, Synthetic GPA Validation & Adapter Training

**Hours budget:** ~50 hours  
**Parallel tracks:** Track A (CPU-only, no dependencies) runs simultaneously with Track B (single GPU, requires downloads). The RTX 6000 Ada's 48 GB VRAM removes all memory constraints for Qwen2.5-1.5B, but adapters must train sequentially on the single GPU.

### Track A: Synthetic GPA Validation (Days 1–4, ~20 hours) — **TOP PRIORITY**

This track has zero dependency on GPU infrastructure, model downloads, or datasets. It validates the core mathematical machinery of the project.

#### Step A.1 — Implement the GPA Core Algorithm (Day 1, ~4 hours)

Create `scripts/gpa.py` — a clean, well-documented NumPy implementation of Gower's alternating optimization:

```python
"""
Generalized Procrustes Analysis (GPA) for LoRA factor alignment.

Implements Gower (1975) alternating optimization:
  1. Fix consensus C, update rotations Q_i via Procrustes (SVD of C @ A_i^T)
  2. Fix rotations Q_i, update consensus C = mean(Q_i @ A_i)
Repeat until convergence.

All operations are on r x d matrices (e.g., 16 x 1536), running on CPU.
"""

import numpy as np
from scipy.linalg import svd, orthogonal_procrustes
from typing import List, Tuple, Optional
import time


def gpa_align(
    matrices: List[np.ndarray],
    max_iter: int = 100,
    tol: float = 1e-6,
    init: str = "first",  # "first" or "mean"
    verbose: bool = False,
) -> Tuple[List[np.ndarray], np.ndarray, List[float]]:
    """
    Align N matrices via Generalized Procrustes Analysis.

    Args:
        matrices: List of N arrays, each shape (r, d).
        max_iter: Maximum GPA iterations.
        tol: Relative change in objective for convergence.
        init: Initialization strategy for consensus.
        verbose: Print per-iteration diagnostics.

    Returns:
        rotations: List of N orthogonal matrices Q_i, each (r, r).
        consensus: Consensus matrix C, shape (r, d).
        residuals: Per-iteration residual sum-of-squares.
    """
    N = len(matrices)
    r, d = matrices[0].shape
    assert all(A.shape == (r, d) for A in matrices), "Shape mismatch"

    # Initialize consensus
    if init == "first":
        C = matrices[0].copy()
    elif init == "mean":
        C = np.mean(matrices, axis=0)
    else:
        raise ValueError(f"Unknown init: {init}")

    rotations = [np.eye(r) for _ in range(N)]
    residuals = []

    for iteration in range(max_iter):
        # Step 1: Update rotations (fix C)
        for i in range(N):
            # Solve ordinary Procrustes: min ||Q_i A_i - C||_F
            # via SVD of C @ A_i^T (an r x r matrix)
            M = C @ matrices[i].T  # (r, r)
            U, _, Vt = svd(M)
            rotations[i] = U @ Vt  # Optimal Q_i = U V^T

        # Step 2: Update consensus (fix rotations)
        C = np.mean([rotations[i] @ matrices[i] for i in range(N)], axis=0)

        # Compute residual
        residual = sum(
            np.linalg.norm(rotations[i] @ matrices[i] - C, 'fro') ** 2
            for i in range(N)
        )
        residuals.append(residual)

        if verbose:
            print(f"  Iter {iteration+1}: residual = {residual:.8e}")

        # Check convergence
        if iteration > 0:
            rel_change = abs(residuals[-1] - residuals[-2]) / (residuals[-2] + 1e-12)
            if rel_change < tol:
                if verbose:
                    print(f"  Converged at iteration {iteration+1}")
                break

    return rotations, C, residuals
```

**Verification:** Write a trivial unit test — generate `A`, apply known identity rotations, confirm GPA returns identity matrices and zero residual.

#### Step A.2 — Experiment 1: Ground-Truth Rotation Recovery (Day 1–2, ~5 hours)

Create `scripts/synthetic_exp1_rotation_recovery.py`:

**Protocol (from proposal Section 3.1):**

1. Generate ground-truth matrix A* ∈ ℝ^{r×d} with entries ~ N(0, 1/d).
2. For each of N adapters: A_i = Q_i A* + σE_i, where Q_i is a random orthogonal matrix (QR decomposition of Gaussian matrix), E_i is i.i.d. Gaussian noise.
3. Run GPA to recover estimated rotations Q̂_i.
4. Measure three metrics:
   - **Rotation recovery error:** (1/N) Σ_i ||Q̂_i − Q_i||²_F (after resolving global rotation ambiguity via Procrustes alignment of the Q̂ set to the Q set).
   - **Consensus relative error:** ||C − A*||_F / ||A*||_F.
   - **Alignment residual:** (1/N) Σ_i ||Q̂_i A_i − C||²_F.

**Parameter sweep:**

| Parameter | Values |
|-----------|--------|
| σ (noise) | {0, 0.01, 0.05, 0.1, 0.2, 0.5} |
| N (adapters) | {3, 5, 10} |
| r (rank) | {4, 8, 16, 32} |
| Trials per config | 100 |

Total configurations: 6 × 3 × 4 = 72. At 100 trials each and ~1 ms per GPA run, the entire sweep should complete in under 5 minutes. Store results in `results/synthetic_exp1.json` with full per-trial data.

**Key outputs:** Heatmaps of rotation recovery error vs. (σ, N) at each r. These will become Figure 1 or 2 of the dissertation.

**Critical success criterion:** At σ ≤ 0.1, rotation recovery error should be < 0.01 for r = 16, N = 5. If this fails, the project's core hypothesis is in trouble — escalate immediately.

#### Step A.3 — Experiment 2: Convergence Curves (Day 2, ~3 hours)

Create `scripts/synthetic_exp2_convergence.py`:

**Protocol (from proposal Section 3.2):**

1. Fix σ = 0.1, N = 10, r = 16, d = 1536.
2. Record residual sum-of-squares at every GPA iteration.
3. Plot residual vs. iteration number (expecting monotonic decrease on log scale).
4. Report: iterations to convergence at tol = 1e-6, total wall-clock time.

**Additional sweep:** Vary N ∈ {3, 5, 10, 20} and r ∈ {8, 16, 32} to show convergence is consistently fast.

**Key output:** A convergence plot (log-scale residual vs. iteration) showing convergence in ≤10 iterations. This validates the claim from Section 2.5 of the proposal.

#### Step A.4 — Experiment 3: Robustness to Non-Orthogonal Perturbations (Day 2–3, ~4 hours)

Create `scripts/synthetic_exp3_nonorthogonal.py`:

**Protocol (from proposal Section 3.3):**

1. Generate A_i = (Q_i + δS_i)A* + σE_i, where S_i are random matrices with unit Frobenius norm.
2. Sweep δ ∈ {0, 0.01, 0.05, 0.1, 0.2} controlling departure from orthogonality.
3. Fix σ = 0.1, N = 5, r = 16.
4. Measure alignment residual and rotation recovery error.
5. 100 trials per configuration.

**This is the most important synthetic experiment.** Real LoRA adapters are NOT exact rotations of a shared matrix. This experiment quantifies how gracefully GPA degrades when its core orthogonality assumption is violated. The δ = 0.1–0.2 range is where we expect real adapters to fall.

**Key output:** A curve of alignment residual vs. δ. If the residual increases gracefully (linearly or sub-linearly), GPA is viable. If it explodes at δ > 0.05, the project may need to pivot to a modified GPA with relaxed constraints.

#### Step A.5 — Experiment 4: Structured LoRA-Like Ground Truth (Day 3–4, ~4 hours)

Create `scripts/synthetic_exp4_structured.py`:

**Protocol (from proposal Section 3.4):**

1. Generate A* with geometrically decaying singular values (mimicking trained LoRA).
2. Add task-specific rank-1 perturbations P_i (simulating task-specific directions).
3. Apply random rotations Q_i and noise.
4. Test whether GPA can separate shared from task-specific components.
5. Measure subspace overlap via principal angles between span(C) and span(A*).

**Key output:** A table showing that GPA's consensus C captures the dominant shared subspace even in the presence of task-specific perturbations. This is the strongest motivation for using GPA on real LoRA adapters.

#### Step A.6 — Generate All Synthetic Figures (Day 4, ~2 hours)

Create `scripts/plot_synthetic.py` using matplotlib:

1. **Figure: Rotation Recovery Heatmap** — error vs. (σ, N) for r = 16.
2. **Figure: Convergence Curves** — log residual vs. iteration for multiple (N, r).
3. **Figure: Non-Orthogonal Robustness** — residual vs. δ with error bars.
4. **Figure: Structured Ground Truth** — subspace overlap vs. perturbation strength.

Save all figures as PDF to `results/figures/`. These are dissertation-ready.

---

### Track B: Infrastructure & Adapter Training (Days 1–5, ~30 hours)

This track runs in parallel with Track A on the single GPU. The RTX 6000 Ada's 48 GB VRAM is massive overkill for Qwen2.5-1.5B (~0.77 GB in 4-bit), so VRAM constraints are eliminated. The trade-off is sequential training (one adapter at a time), compensated by much larger batch sizes.

#### Step B.1 — Environment Setup (Day 1, ~3 hours)

```bash
conda create -n lora-merge python=3.10 -y
conda activate lora-merge

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.46.0 \
            datasets==3.0.0 \
            peft==0.13.0 \
            bitsandbytes==0.44.1 \
            accelerate==1.0.0 \
            evaluate \
            scikit-learn \
            numpy \
            scipy \
            wandb \
            tqdm
```

**Why cu124:** The RTX 6000 Ada uses the AD102 chip (Ada Lovelace architecture) which works best with CUDA 12.x. The old plan used `cu118` for the RTX 2080s — this must be updated. Verify your system's CUDA driver version with `nvidia-smi` and ensure it supports CUDA 12.4+.

Verify GPU access:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
props = torch.cuda.get_device_properties(0)
print(f"GPU 0: {props.name}, {props.total_mem / 1e9:.2f} GB")
```

You should see 1× RTX 6000 Ada Generation with ~48 GB. Verify LAPACK backend for NumPy/SciPy (same as before).

Project directory structure (unchanged):

```bash
mkdir -p ~/gpa-lora-merge/{configs,scripts,adapters,results,logs,data}
```

#### Step B.2 — Hardware Profiling (Day 1, ~1 hour)

With 48 GB VRAM, you no longer need to carefully sweep batch sizes to fit within an 8 GB budget. A single profiling run confirms everything works:

- Load quantized Qwen2.5-1.5B, measure base VRAM (~0.77 GB expected).
- Attach LoRA (r=16, target all attention + MLP projections), run a forward + backward pass at batch_size=32, seq_len=256.
- Expected peak VRAM: ~3–5 GB, leaving ~43 GB headroom.
- Record in `results/vram_profile.json` for the dissertation.

**Note:** You no longer need the batch size × seq_len sweep table. A single configuration (batch=32, seq=256) will work for all tasks. If you want to speed up training further, try batch=64 — you have the VRAM for it.

#### Step B.3 — Data Preparation & Training Script (Day 2, ~4 hours)

- Create `scripts/data.py` with TASK_CONFIG for all 5 GLUE tasks.
- Handle MNLI's `validation_matched` split, CoLA's Matthew's Correlation metric.
- Create `scripts/train.py` with CLI args, QLoRA config, HuggingFace Trainer integration.
- **Remove all GPU index selection logic** — there is only one GPU. Use `device_map="auto"` or `device_map={"": 0}` throughout.

#### Step B.4 — Smoke Test: SST-2 (Day 2, ~1 hour)

Run a 50-step sanity check on SST-2. With batch_size=32 this will be very fast. Verify: loss decreases, no OOM, adapter saves and reloads correctly.

#### Step B.5 — Full Adapter Training (Day 2–4, ~6 hours)

Since you have a single GPU, adapters train **sequentially**. However, the RTX 6000 Ada's higher throughput and larger batch sizes make each run significantly faster than it would have been on an RTX 2080.

Create `scripts/run_all.sh` — a **sequential** training script:

```bash
#!/bin/bash
# Train all 5 QLoRA adapters sequentially on the single RTX 6000 Ada.
set -e
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "Starting sequential training at $TIMESTAMP"

python train.py --task sst2 \
    --output_dir ../adapters/sst2 \
    --epochs 3 --lr 2e-4 --per_device_batch 32 --grad_accum 1 \
    --max_length 128 --eval_steps 500 \
    2>&1 | tee ../logs/sst2_${TIMESTAMP}.log

python train.py --task mnli \
    --output_dir ../adapters/mnli \
    --epochs 2 --lr 2e-4 --per_device_batch 32 --grad_accum 1 \
    --max_length 256 --eval_steps 1000 \
    2>&1 | tee ../logs/mnli_${TIMESTAMP}.log

python train.py --task qnli \
    --output_dir ../adapters/qnli \
    --epochs 3 --lr 2e-4 --per_device_batch 32 --grad_accum 1 \
    --max_length 256 --eval_steps 500 \
    2>&1 | tee ../logs/qnli_${TIMESTAMP}.log

python train.py --task cola \
    --output_dir ../adapters/cola \
    --epochs 5 --lr 1e-4 --per_device_batch 32 --grad_accum 1 \
    --max_length 128 --eval_steps 100 \
    2>&1 | tee ../logs/cola_${TIMESTAMP}.log

python train.py --task rte \
    --output_dir ../adapters/rte \
    --epochs 10 --lr 1e-4 --per_device_batch 32 --grad_accum 1 \
    --max_length 256 --eval_steps 50 \
    2>&1 | tee ../logs/rte_${TIMESTAMP}.log

echo "All training runs complete."
```

**Note:** No `--gpu` flag needed — there is only one GPU. The `grad_accum` is set to 1 since batch_size=32 is already a reasonable effective batch size with 48 GB VRAM. Increase to batch_size=64 if training is too slow.

Estimated training times on the RTX 6000 Ada:

| Task | Epochs | Batch | Est. Time |
|------|--------|-------|-----------|
| SST-2 | 3 | 32 | ~30 min |
| MNLI | 2 | 32 | ~2.5 hr |
| QNLI | 3 | 32 | ~1.5 hr |
| CoLA | 5 | 32 | ~10 min |
| RTE | 10 | 32 | ~10 min |
| **Total** | | | **~5 hours** |

**Tip:** Launch `run_all.sh` at the end of Day 2 and let it run overnight. All 5 adapters will be ready by morning on Day 3. Meanwhile, continue working on Track A synthetic experiments on CPU.

#### Step B.6 — Post-Training Validation (Day 4, ~3 hours)

- Run standalone evaluation on all 5 adapters.
- Verify performance targets: SST-2 92–95%, MNLI 82–87%, QNLI 88–92%, CoLA 0.50–0.65, RTE 70–78%.
- Extract and inspect LoRA matrices: confirm r = 16, d_in = 1536 (attention) for all adapters.
- Save parameter mapping JSON for Week 2.

**Contingency:** If any single training run fails (e.g., NaN loss), debug and restart — you have the VRAM headroom to experiment with different learning rates without worrying about OOM. If total training time overruns, reduce MNLI to 1 epoch or drop it entirely for a valid N = 4 proof-of-concept.

### Week 1 Deliverables Checklist

| # | Deliverable | Location | Priority |
|---|-------------|----------|----------|
| 1 | GPA core implementation | `scripts/gpa.py` | **HIGH** |
| 2 | Synthetic Exp 1–4 results (JSON) | `results/synthetic_exp{1,2,3,4}.json` | **HIGH** |
| 3 | Synthetic validation figures (4 figures) | `results/figures/` | **HIGH** |
| 4 | Working conda environment | `lora-merge` env | HIGH |
| 5 | VRAM profiling result (single config, batch=32) | `results/vram_profile.json` | HIGH |
| 6 | 5 trained QLoRA adapters | `adapters/{sst2,mnli,qnli,cola,rte}/` | HIGH |
| 7 | Per-task eval metrics | `adapters/*/eval_metrics.json` | HIGH |
| 8 | LoRA shape verification + mapping JSON | `configs/lora_param_mapping.json` | HIGH |
| 9 | Training scripts + sequential launcher | `scripts/{train,evaluate,data}.py`, `run_all.sh` | MEDIUM |

**Go/No-Go Gate (end of Week 1):** If Experiment 3 shows GPA degrades catastrophically at δ ≥ 0.1, immediately schedule a meeting with your supervisor to discuss whether to proceed with modified GPA or pivot the project focus. Do NOT wait until Week 3 to discover this.

---

## Week 2 (7 Apr – 13 Apr): Baseline Merging, GPA Pipeline on Real Adapters & Begin Writing

**Hours budget:** ~50 hours  
**Goal:** Implement all merging baselines, apply GPA to real LoRA adapters for the first time, and begin the dissertation.

### Phase 1: Baseline Merging Implementations (Days 1–3, ~20 hours)

#### Step 2.1 — Task Arithmetic Baseline (Day 1, ~4 hours)

Create `scripts/merge_task_arithmetic.py`:

- Compute task vectors: τ_i = θ_adapter_i (the LoRA weights themselves serve as the task vectors since they represent Δ from the pretrained model).
- Merge: τ_merged = λ Σ_i τ_i, where λ is swept over {0.1, 0.2, ..., 1.0}.
- Reconstruct merged adapter and evaluate on all 5 tasks.

#### Step 2.2 — TIES-Merging Baseline (Day 1–2, ~6 hours)

Create `scripts/merge_ties.py`:

Implement the full TIES pipeline operating on the raw (unaligned) LoRA weight matrices:

1. **Trim:** Zero out the bottom k% of parameters by magnitude (sweep k ∈ {10, 20, 30}).
2. **Elect Sign:** For each parameter position, elect the sign (positive or negative) that has the highest total magnitude across adapters.
3. **Disjoint Merge:** Average only the parameter values that agree with the elected sign.
4. Apply scaling coefficient λ.

This is the most important baseline — it's what GPA+TIES will be compared against.

#### Step 2.3 — DARE Baseline (Day 2, ~4 hours)

Create `scripts/merge_dare.py`:

1. For each adapter, randomly drop parameters with probability p (sweep p ∈ {0.1, 0.5, 0.9}).
2. Rescale remaining parameters by 1/(1-p).
3. Apply Task Arithmetic or TIES-Merging to the sparsified adapters.

#### Step 2.4 — LR-KnOTS Baseline (Day 2–3, ~6 hours)

Create `scripts/merge_lr_knots.py`:

This is the key baseline from the proposal (Section 1.6.5):

1. Column-wise concatenate all A factors: A_concat = [A_1; A_2; ...; A_N] ∈ ℝ^{Nr × d}.
2. Perform SVD: A_concat = UΣV^T.
3. Extract per-adapter components V^(i) from the right singular vectors.
4. Apply TIES-Merging to the V^(i) matrices.
5. Reconstruct the merged adapter.

**Note:** This baseline requires materialising the concatenated matrix (Nr × d), which for N=5, r=16, d=1536 is only 80×1536 — trivially small. The memory advantage of GPA over KnOTS only matters at scale (large d, large N), but the algorithmic comparison is still meaningful.

### Phase 2: GPA Alignment on Real Adapters (Days 3–4, ~12 hours)

#### Step 2.5 — Apply GPA to Real LoRA A Matrices (Day 3, ~6 hours)

Create `scripts/gpa_align_adapters.py`:

```python
"""
Apply GPA alignment to real QLoRA adapters.

For each LoRA layer (e.g., layers.0.self_attn.q_proj.lora_A):
  1. Load the A matrices from all N adapters.
  2. Run GPA to find rotations Q_i and consensus C.
  3. Compute aligned factors: A_tilde_i = Q_i @ A_i, B_tilde_i = B_i @ Q_i^T.
  4. Save aligned adapters.
"""
```

Process each LoRA module independently (28 layers × 7 modules = 196 GPA problems). Per the computational analysis (Section 2), total wall-clock time should be ~80–280 ms for all layers.

**Important diagnostics to record per layer:**
- Number of GPA iterations to convergence.
- Final alignment residual.
- Frobenius norm of A_i before and after alignment (should be preserved since Q_i is orthogonal).
- Verify functional invariance: ||B̃_i Ã_i - B_i A_i||_F should be ≈ 0 (up to floating-point).

#### Step 2.6 — GPA+TIES Factor-Space Merging (Day 3–4, ~6 hours)

Create `scripts/merge_gpa_ties.py`:

Implement the primary method from proposal Section 1.6.3:

1. Run GPA alignment (from Step 2.5) to get Ã_i = Q_i A_i and B̃_i = B_i Q_i^T.
2. Apply TIES-Merging to the **aligned A factors** {Ã_1, ..., Ã_N} → Ã_merged.
3. Apply simple **averaging** to the aligned B factors: B̃_merged = (1/N) Σ_i B̃_i.
4. Reconstruct merged adapter: ΔW_merged = B̃_merged × Ã_merged.
5. Scale by λ and add to pretrained weights.

### Phase 3: Evaluation Framework & Begin Dissertation (Days 4–5, ~18 hours)

#### Step 2.7 — Unified Evaluation Script (Day 4, ~4 hours)

Create `scripts/evaluate_merged.py`:

- Accept a merged adapter directory as input.
- Evaluate on all 5 GLUE tasks (or the subset that was trained).
- Output a JSON with per-task metrics and average.
- Support evaluating with different λ scaling coefficients.

#### Step 2.8 — Run Initial Comparison (Day 4–5, ~6 hours)

Run all methods and record results:

| Method | Description |
|--------|-------------|
| Task Arithmetic | Simple average of task vectors |
| TIES-Merging | On raw (unaligned) LoRA weights |
| DARE + TIES | Sparsified then merged |
| LR-KnOTS + TIES | SVD concatenation baseline |
| **GPA + TIES** | Our method: factor-space merging |

This gives the first data point on whether GPA helps. Don't over-optimise hyperparameters yet — that's Week 4.

#### Step 2.9 — Begin Dissertation: Introduction & Related Work (Day 5, ~8 hours)

Start writing in parallel. Target ~3,000 words across:

- **Introduction:** Problem statement, motivation, contributions.
- **Related Work:** Model merging (Task Arithmetic, TIES, DARE, KnOTS), Procrustes analysis (Gower, Ling, Pizarro & Bartoli), representation alignment (Multi-Way Alignment paper).

Writing early forces you to articulate the contribution clearly before the experiments are complete.

### Week 2 Deliverables Checklist

| # | Deliverable | Location |
|---|-------------|----------|
| 1 | Task Arithmetic merging script + results | `scripts/merge_task_arithmetic.py` |
| 2 | TIES-Merging script + results | `scripts/merge_ties.py` |
| 3 | DARE script + results | `scripts/merge_dare.py` |
| 4 | LR-KnOTS script + results | `scripts/merge_lr_knots.py` |
| 5 | GPA alignment on real adapters | `scripts/gpa_align_adapters.py` |
| 6 | GPA+TIES factor-space merging | `scripts/merge_gpa_ties.py` |
| 7 | Unified evaluation framework | `scripts/evaluate_merged.py` |
| 8 | Initial comparison table (all methods) | `results/initial_comparison.json` |
| 9 | Dissertation draft: Introduction + Related Work | `dissertation/chapters/` |

---

## Week 3 (14 Apr – 20 Apr): Scale-Aware Enhancements, Full Experiment Matrix, Methods Section & Hyperparameter Tuning

**Hours budget:** ~50 hours  
**Goal:** Implement supervisor-recommended scale-aware enhancements to the GPA+TIES pipeline, then systematically evaluate all methods (including enhanced variants) across all hyperparameter configurations, and write the Methods chapter.

> **Supervisor feedback (received end of Week 2):** The Week 1–2 norm analysis and CoLA collapse were identified as key diagnostic findings. The supervisor recommended three targeted enhancements to address scale bias in the GPA+TIES pipeline: (1) Directional GPA via pre-normalisation of A factors, (2) Scale-aware TIES merging on rescaled aligned factors, and (3) Inverse-norm-weighted B̃ averaging. These are implemented as enhancements to the primary method in Phase 0 below, then evaluated alongside the baseline methods in the Phase 1 hyperparameter sweep. They are treated as **primary method improvements** (not Week 4 ablations) because they address a fundamental flaw — scale bias — that is the most likely cause of the CoLA collapse and narrow GPA advantage observed in Week 2.

### Phase 0: Implement Scale-Aware GPA+TIES Enhancements (Day 1, ~5 hours) — **TOP PRIORITY**

These three modifications are small code changes to `scripts/merge_gpa_ties.py` and `scripts/gpa.py`, but they are the most important work of Week 3 because they directly target the failure modes identified in the Week 1–2 analysis.

#### Step 3.0a — Directional GPA Alignment via Pre-Normalised A Factors (Day 1, ~2 hours)

**Problem:** Standard GPA minimises $\sum_i \|Q_i A_i - C\|_F^2$. While the consensus update step (the mean) weights each adapter equally, the rotation update step is influenced by scale: the Frobenius residual for a larger-norm $A_i$ dominates the overall objective. With MNLI's A-norm at 3.80 vs. RTE's at 2.36, MNLI exerts ~2.6× more geometric pull on the consensus during rotation fitting. This means the "shared subspace" GPA finds is biased toward MNLI's coordinate system.

**Solution:** Normalise each $A_i$ to unit Frobenius norm before running GPA, then apply the recovered rotations to the original (unnormalised) matrices. This decouples directional alignment from scale.

**Implementation:** Add a `normalise` flag to `scripts/gpa.py` and a wrapper in `scripts/merge_gpa_ties.py`:

```python
def gpa_align_normalised(matrices, **kwargs):
    """
    Directional GPA: normalise inputs to unit Frobenius norm,
    run GPA on the unit-norm matrices to get pure directional rotations,
    then apply those rotations to the original matrices.
    
    This ensures alignment focuses on shared subspace structure
    rather than being biased by adapters with larger magnitudes.
    """
    # Step 1: Store original norms
    norms = [np.linalg.norm(A, 'fro') for A in matrices]
    
    # Step 2: Normalise to unit Frobenius norm
    normed = [A / n for A, n in zip(matrices, norms)]
    
    # Step 3: Run standard GPA on normalised matrices
    rotations, consensus_normed, residuals = gpa_align(normed, **kwargs)
    
    # Step 4: Apply rotations to ORIGINAL matrices (not the normalised ones)
    # This preserves the learned magnitude information while using
    # the scale-unbiased rotations.
    aligned_A = [rotations[i] @ matrices[i] for i in range(len(matrices))]
    consensus = np.mean(aligned_A, axis=0)
    
    return rotations, consensus, residuals, norms
```

**Verification checklist:**
- Confirm that `||Q_i A_i||_F == ||A_i||_F` for all i (orthogonal rotation preserves norms).
- Confirm functional invariance still holds: `||B̃_i Ã_i - B_i A_i||_F ≈ 0`.
- Compare the rotation matrices $Q_i$ from normalised vs. unnormalised GPA — they should differ, especially for layers with high norm disparity (early MLP layers where MNLI peaks near 9.0).
- Log per-layer alignment residuals for both variants for later comparison.

#### Step 3.0b — Scale-Aware TIES Merging on Rescaled Aligned Factors (Day 1, ~1.5 hours)

**Problem:** After GPA alignment, TIES operates on the aligned $\tilde{A}_i$ matrices. The TIES trim step keeps the top-$k\%$ of parameters by magnitude and zeros out the rest. When adapters have very different scales, "important" parameters in a small-norm adapter (CoLA, combined A+B = 2.82) may have magnitudes below "redundant" parameters in a large-norm adapter (MNLI, combined A+B = 6.58). The TIES sign election step similarly uses total magnitude to determine the elected sign — MNLI's sign preference will almost always dominate. This is the most likely cause of CoLA's features being systematically discarded during merging.

**Solution:** Rescale each $\tilde{A}_i$ to unit Frobenius norm **before** the TIES trim and elect steps, then rescale back after the disjoint merge. This ensures each adapter's parameters compete on a level playing field during conflict resolution.

**Implementation:** Modify the TIES pipeline in `scripts/merge_gpa_ties.py`:

```python
def ties_merge_scale_aware(aligned_A_list, trim_percentage=20):
    """
    Scale-aware TIES: normalise aligned A factors before trim/elect,
    so that small-norm adapters (CoLA, RTE) have a fair chance of
    surviving the conflict resolution against large-norm adapters (MNLI).
    
    After TIES merging, the merged result is rescaled to the average
    norm of the original aligned factors.
    """
    # Step 1: Store original norms and compute target scale
    norms = [np.linalg.norm(A, 'fro') for A in aligned_A_list]
    avg_norm = np.mean(norms)
    
    # Step 2: Rescale each Ã_i to unit norm for fair TIES comparison
    rescaled = [A / n for A, n in zip(aligned_A_list, norms)]
    
    # Step 3: Apply standard TIES (trim, elect sign, disjoint merge) 
    # on the rescaled matrices — now CoLA's features are on equal
    # footing with MNLI's during magnitude-based trim and sign election
    merged_unit = ties_merge(rescaled, trim_percentage=trim_percentage)
    
    # Step 4: Rescale merged result to average original norm
    merged = merged_unit * avg_norm
    
    return merged
```

**Key design decision — rescaling target:** The merged result is rescaled to the *average* of the original norms, not the sum. This is because the TIES disjoint merge already takes the mean of sign-consistent values, so the output should represent a single adapter's worth of scale. The average norm is the most neutral choice. An alternative (rescale to median norm) could be tested as a micro-ablation but is low priority.

**Verification:** After running the sweep, check per-method CoLA prediction distributions. If scale-aware TIES recovers non-degenerate CoLA predictions at λ values where the baseline GPA+TIES still collapses, this confirms that scale bias during TIES was the root cause.

#### Step 3.0c — Inverse-Norm-Weighted B̃ Averaging (Day 1, ~1.5 hours)

**Problem:** The primary method averages the aligned B factors: $\tilde{B}_{\text{merged}} = \frac{1}{N} \sum_i \tilde{B}_i$. With MNLI's B-norm at 2.78 vs. RTE's 0.34 (an 8× spread), this average is heavily MNLI-dominated. The output projection of the merged model is essentially MNLI's output projection with minor perturbations from other tasks.

**Solution:** Use inverse-norm weighting: $w_i \propto 1 / \|\tilde{B}_i\|_F^\alpha$, normalised so $\sum w_i = 1$. The exponent $\alpha$ controls the strength of rebalancing:
- $\alpha = 0$: Recovers simple averaging (no rebalancing).
- $\alpha = 0.5$: Moderate rebalancing (square-root dampening).
- $\alpha = 1.0$: Full inverse-norm weighting.

**Implementation:** Modify B̃ averaging in `scripts/merge_gpa_ties.py`:

```python
def weighted_B_average(aligned_B_list, alpha=1.0):
    """
    Inverse-norm-weighted averaging of aligned B factors.
    
    Gives more weight to adapters with smaller B-norms to counteract
    the dominance of large-norm adapters (MNLI) in the output projection.
    
    Args:
        aligned_B_list: List of N arrays, each shape (d_out, r).
        alpha: Weighting exponent.
            0.0 = simple average (baseline).
            0.5 = moderate rebalancing.
            1.0 = full inverse-norm weighting.
    
    Returns:
        B_merged: Weighted average, shape (d_out, r).
    """
    norms = [np.linalg.norm(B, 'fro') for B in aligned_B_list]
    
    # Compute inverse-norm weights
    raw_weights = [1.0 / (n ** alpha + 1e-8) for n in norms]
    total = sum(raw_weights)
    weights = [w / total for w in raw_weights]
    
    # Weighted average
    B_merged = sum(w * B for w, B in zip(weights, aligned_B_list))
    
    return B_merged
```

**Sweep:** Test $\alpha \in \{0.0, 0.5, 1.0\}$ alongside the λ sweep. Since $\alpha = 0.0$ recovers the baseline, this is a strict superset of the current method.

**Caution noted:** The B-norms carry genuine information about task adaptation magnitude. Aggressively upweighting small-B-norm adapters (CoLA, RTE) may amplify noise from their relatively modest fine-tuning updates. The $\alpha = 0.5$ setting provides a compromise. If $\alpha = 1.0$ degrades performance on tasks where the baseline was already reasonable (SST-2, MNLI), prefer $\alpha = 0.5$.

#### Summary of Enhanced GPA+TIES Variants for the Sweep

The three enhancements are **composable** and should be tested both individually and in combination. The full variant matrix for GPA+TIES is:

| Variant Label | Directional GPA | Scale-Aware TIES | B̃ Weighting (α) | Notes |
|---------------|:-:|:-:|:-:|---|
| `GPA+TIES` (baseline) | ✗ | ✗ | 0.0 | Existing method from Week 2 |
| `dGPA+TIES` | ✓ | ✗ | 0.0 | Tests alignment fix alone |
| `dGPA+saTIES` | ✓ | ✓ | 0.0 | Tests alignment + TIES fix |
| `dGPA+saTIES+wB(0.5)` | ✓ | ✓ | 0.5 | Moderate B rebalancing |
| `dGPA+saTIES+wB(1.0)` | ✓ | ✓ | 1.0 | Full B rebalancing |

Each variant is swept over the same λ × k grid as the baseline GPA+TIES (10 × 3 = 30 configs per variant), giving **150 additional configurations** for the GPA+TIES family. Since the merging step is CPU-milliseconds, the bottleneck is GPU evaluation only. Budget ~3 hours additional evaluation time for these variants.

**If time is constrained:** Prioritise `dGPA+saTIES+wB(1.0)` (the full enhancement stack) and `GPA+TIES` (the baseline). The intermediate variants (`dGPA+TIES`, `dGPA+saTIES`) help isolate which enhancement contributes most but are lower priority than the full comparison.

### Phase 1: Comprehensive Hyperparameter Sweep (Days 1–3, ~25 hours)

#### Step 3.1 — Scaling Coefficient λ Sweep (Day 1, ~8 hours)

> **Informed by Step 2.8 findings:** The initial comparison at λ=1.0 showed that factor-space methods (GPA+TIES, LR-KnOTS+TIES) have evaluation losses of 2.0–2.7, far higher than Task Arithmetic's 0.7–1.1. This strongly suggests λ=1.0 is too aggressive for these methods. The sweep grid below is therefore biased toward the low end for factor-space methods, with finer resolution in the 0.05–0.5 range where their optimal λ is most likely to fall.

Use **two λ grids** depending on method family:

- **Delta-space methods** (Task Arithmetic, raw TIES, DARE+TIES): λ ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0} — 10 values.
- **Factor-space methods** (GPA+TIES, LR-KnOTS+TIES): λ ∈ {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0} — 10 values, with finer granularity in the 0.05–0.5 range.

Sweep configurations:

- Task Arithmetic: λ sweep (10 configs).
- TIES-Merging: λ × k sweep (10 × 3 = 30 configs, but fast since no GPU inference needed for merging itself — only for evaluation).
- DARE + TIES: λ × p × k sweep.
- LR-KnOTS + TIES: λ × k sweep (10 × 3 = 30 configs, finer λ grid).
- GPA + TIES (baseline): λ × k sweep (10 × 3 = 30 configs, finer λ grid).
- GPA + TIES (enhanced variants): λ × k sweep for each of the 4 new variants from Step 3.0 (up to 4 × 30 = 120 additional configs). **If time-constrained**, run only `dGPA+saTIES+wB(1.0)` (30 configs) and compare against the baseline.

Automate with a grid sweep script. Evaluation is the bottleneck (each evaluation run requires a forward pass through the merged model on each task's validation set), so batch evaluations efficiently on the single GPU.

#### Step 3.1b — CoLA Prediction Distribution Diagnostic (Day 1, ~1 hour)

> **Informed by Step 2.8 findings:** Every merged method scored Matthews correlation 0.0 on CoLA at λ=1.0, despite the single-task adapter achieving 0.6522. This is the clearest failure mode in the initial comparison and must be understood early — before writing the Methods chapter — to determine whether CoLA collapse is a λ-scaling issue or a deeper structural problem.
>
> **Informed by supervisor feedback:** The scale-aware enhancements (Step 3.0) are specifically designed to rescue CoLA. This diagnostic is the primary test of whether they succeed.

During or immediately after the λ sweep, log the **prediction class distribution** for CoLA at each λ value for every method:

- For each (method, λ) configuration, count the number of predictions in each class (acceptable / unacceptable).
- Record whether the merged model is producing near-constant predictions (e.g., >95% one class).
- Identify the λ threshold at which CoLA predictions become non-degenerate (if any).
- **New:** Compare the CoLA recovery threshold between baseline GPA+TIES and the enhanced variants. If `dGPA+saTIES+wB` recovers non-degenerate CoLA predictions at higher λ values than the baseline, this is strong evidence that scale bias was the root cause of the collapse.

This costs almost nothing — just add a few lines to the evaluation loop to record `predictions.argmax(-1).bincount()`. The output directly informs:

1. Whether CoLA recovers at lower λ (in which case it's a scaling issue, not a structural one).
2. How to frame CoLA results in the dissertation — as a failure of the specific λ=1.0 configuration vs. a fundamental limitation of merging on this task.
3. Whether similar diagnostics are needed for QNLI and RTE (which showed near-invariant scores across methods at λ=1.0).

Save results to `results/hp_sweep/cola_prediction_distributions.json`.

#### Step 3.2 — Full Results Table (Day 2–3, ~8 hours)

Create the main results table for the dissertation. For each method at its best hyperparameters:

| Method | SST-2 | MNLI | QNLI | CoLA | RTE | **Avg** |
|--------|-------|------|------|------|-----|---------|
| Individual (oracle) | — | — | — | — | — | — |
| Simple Averaging | | | | | | |
| Task Arithmetic | | | | | | |
| TIES-Merging | | | | | | |
| DARE + TIES | | | | | | |
| LR-KnOTS + TIES | | | | | | |
| **GPA + TIES** (baseline) | | | | | | |
| **dGPA + saTIES + wB** (enhanced) | | | | | | |

Report best λ, k, and α for each method. Include standard deviation if running multiple seeds. If intermediate variants (`dGPA+TIES`, `dGPA+saTIES`) were tested, include them in a supplementary ablation table showing which enhancement contributed most.

#### Step 3.3 — Statistical Significance (Day 3, ~4 hours) — **IN PROGRESS (16 April)**

Run each primary method with 3 different random seeds and report mean ± std. Currently running on the three primary methods: `GPA+TIES`, `TIES`, and `LR-KnOTS+TIES`, each at their best λ from the Week 3 sweep.

**Addition under refined plan:** Also run 3 seeds on the headline enhanced variant `dGPA + saTIES + wB(0.5)` at its best setting (λ=0.25). This is a small extra cost (~30 min evaluation per seed × 3 seeds ≈ 1.5 hours) and is needed because the headline gain (+0.008 vs. baseline GPA, +0.005 vs. raw TIES) is small enough that seed noise could plausibly swamp it.

The proposal's success criterion requires ≥1% average accuracy improvement, consistent across at least 3 of 5 tasks.

**Honest framing:** If the enhanced GPA gain over raw TIES is within one standard deviation of zero, report that directly. This does not weaken the dissertation — it sharpens claim C2 (alignment and merge quality are separable) and claim C3 (scale is the binding constraint). Save the seed-variance band in `results/seed_variance.json` for use in the Week 4 experiments chapter.

### Phase 2: Dissertation Methods Chapter (Days 3–5, ~20 hours)

#### Step 3.4 — Write Methods Section (~15 hours)

Target ~4,000–5,000 words covering:

- **GPA Algorithm:** Full mathematical derivation of the alternating optimization, orthogonality constraints, convergence properties. Reference Gower (1975) and Ling (2023).
- **GPA for LoRA Factor Alignment:** How GPA is applied to the A matrices, the rotation/counter-rotation scheme (Equation 3 from the proposal), and why merging must happen in the aligned factor space.
- **Scale-Aware Enhancements (new — from supervisor feedback):** Present the three enhancements as principled solutions to the scale-bias problem diagnosed in the Week 1 adapter analysis:
  - *Directional GPA via pre-normalisation:* Motivate from the GPA objective's implicit norm-weighting. Explain that the rotation update step's residual for larger-norm $A_i$ dominates the objective. Show how normalising to unit Frobenius norm before GPA produces scale-unbiased rotations, while applying those rotations to original matrices preserves learned magnitudes. Contrast with LR-KnOTS, whose single SVD on concatenated factors is even more heavily norm-biased (as noted in proposal Section 1.6.5).
  - *Scale-aware TIES:* Motivate from the TIES paper's own analysis that the trim step depends on relative magnitudes across adapters. Explain that when adapters have drastically different scales (8× B-norm spread), the trim threshold conflates "redundant in a large adapter" with "important in a small adapter." Rescaling $\tilde{A}_i$ before trim/elect ensures each adapter's parameters compete on a level playing field.
  - *Inverse-norm B̃ weighting:* Present as addressing the output-side scale imbalance. Discuss the caution that B-norms carry genuine information and the $\alpha$ exponent as a tunable compromise.
- **Post-Alignment Merging:** The TIES-on-Ã, average-B̃ pipeline. Why consensus replacement is insufficient (Section 1.6.1). Why reconstruction defeats alignment (Section 1.6.2).
- **LR-KnOTS Baseline:** How it differs from GPA (single SVD vs. iterative, concatenation vs. equal-weight alignment).
- **Computational Complexity:** Include the cost analysis from proposal Section 2. Note that the scale-aware enhancements add negligible overhead (a few vector norm computations per layer).
- **TIES Sparsity in Factor Space (new):** Discuss the Step 2.8 observation that TIES-family methods do not produce sparse merged adapters in practice. Despite trimming each source tensor (e.g., `trim_percentage=20`), the disjoint merge step repopulates positions across tasks, leaving the final merged adapter almost fully dense (nonzero fractions ~0.999). This means that DARE+TIES's benefit likely comes from regularisation during sparsification rather than from producing a compact sparse adapter. This observation affects how the TIES pipeline should be described and is worth noting as a methodological insight.

#### Step 3.5 — Create Methodology Figures (~5 hours)

- **Pipeline Diagram:** A figure showing the full enhanced pipeline: Train adapters → Extract A, B → **Normalise A to unit norm** → GPA alignment (directional) → Apply rotations to original A, B → **Rescale Ã to unit norm** → TIES on rescaled Ã (trim, elect, merge) → **Rescale merged Ã** → **Inverse-norm-weighted B̃ average** → Reconstruct → Evaluate. Annotate the three new enhancement steps distinctly (e.g., dashed boxes or colour coding) to make clear what is new vs. the baseline GPA+TIES pipeline.
- **GPA Visualisation:** A 2D illustration showing how GPA rotates misaligned subspaces into a common coordinate system. Optionally add a panel showing the difference between norm-biased and directional alignment (two adapters of different scales converging to a consensus that is skewed toward the larger one vs. an unbiased consensus).

### Week 3 Deliverables Checklist

| # | Deliverable | Location |
|---|-------------|----------|
| 0 | Scale-aware GPA+TIES implementation (3 enhancements) | `scripts/merge_gpa_ties.py`, `scripts/gpa.py` |
| 1 | Hyperparameter sweep results (all methods incl. enhanced variants) | `results/hp_sweep/` |
| 1b | CoLA prediction distribution diagnostic (baseline vs. enhanced) | `results/hp_sweep/cola_prediction_distributions.json` |
| 2 | Main results table (all methods × all tasks, incl. best enhanced variant) | `results/main_results.json` |
| 2b | Enhancement ablation table (contribution of each enhancement) | `results/enhancement_ablation.json` |
| 3 | Best hyperparameters per method | `results/best_hparams.json` |
| 4 | Dissertation draft: Methods chapter (incl. scale-aware enhancements, TIES sparsity discussion) | `dissertation/chapters/methods.md` |
| 5 | Pipeline diagram (annotated with enhancement steps) + GPA illustration | `dissertation/figures/` |

---

## Week 4 (21 Apr – 27 Apr): Focused Ablations, CKA Analysis & Experiments Chapter

**Hours budget:** ~45 hours (down from original ~50; freed time redirected to Week 5 writing).
**Goal:** Produce the diagnostic ablations and analyses that directly support C1–C5, then draft the Experiments & Results chapter.

> **Re-prioritisation rationale:** The original Week 4 listed seven ablations. Under the refined framing (C1–C5), they are not equally valuable. Ablations that do not support any core claim have been cut; ablations that directly probe the identified bottleneck (post-alignment compression, scale handling) have been elevated. Net effect: ~11 hours of ablation work cut, redirected into extra seed runs (Step 3.3), a new methodological comparison table (Step 4.X), and Week 5 writing buffer.

### Ablation Prioritisation Matrix

| Step | Ablation | Supports | Priority | Change from Original |
|---|---|---|---|---|
| 4.1 | Vary N (number of merged adapters) | C2, C4 | **HIGH** | Elevated — clearest remaining test of proposal hypothesis |
| 4.4-reduced | TIES on B̃ (single new config) | C3 | **MEDIUM** | Reduced scope — only the novel variant |
| 4.5 | Task Arithmetic in aligned space | C2, C4 | **MEDIUM** | Reframed as direct test of C2 |
| 4.6 | CKA before/after alignment | C1, C2 | **HIGH** | Unchanged |
| 4.7 | Per-layer alignment residual + norm correlation | C1, C3 | **HIGH** | Expanded to correlate with B-norm disparity |
| 4.X | **NEW:** Qualitative methodological comparison table | C1, C4 | **MEDIUM** | Added (~1 hour) |
| 4.2 | Vary LoRA rank | C4 (weakly) | **LOW** | Narrowed scope, conditional on time |
| 4.3 | Consensus replacement vs. TIES | — | **CUT** | No core claim support |
| 4.5b | Float16 vs. NF4 quantization | — | **CUT** | No core claim support |

### Phase 1: Finish Statistical Significance (Day 1, ~4 hours)

#### Step 3.3-completion — Finish Seed Runs (Day 1)

Finish the 3-seed runs for the three primary methods (`GPA+TIES`, `TIES`, `LR-KnOTS+TIES`) plus the headline enhanced variant `dGPA + saTIES + wB(0.5)` at their respective best λ settings from the Week 3 sweep. Report mean ± std on the average metric and per-task. Save to `results/seed_variance.json`.

### Phase 2: Core Ablations (Days 1–3, ~16 hours)

#### Step 4.1 — Vary Number of Merged Adapters (Day 1–2, ~6 hours) — **HIGH PRIORITY**

Unchanged in protocol but elevated in priority because this is the single remaining experiment most likely to produce a clean positive result for GPA.

Test merging subsets N ∈ {2, 3, 4, 5}. For each N, evaluate `GPA+TIES` (at its best λ), the best enhanced GPA variant, `TIES` (at its best λ), and `LR-KnOTS+TIES` (at its best λ). Use the best hyperparameters from Week 3 rather than re-sweeping.

- For N=2: all 10 pairs.
- For N=3: all 10 triples.
- For N=4: all 5 subsets of size 4.
- For N=5: the single full set (already done).

**Key question:** Does GPA's advantage over raw TIES grow with N? This is the clearest remaining test of claim C4 and was specifically called out in the proposal's hypothesis.

**Interpretation ladder:**
- If GPA's advantage grows monotonically with N: strong support for C4, and a clean figure for the dissertation.
- If GPA's advantage is flat across N: evidence that multi-way alignment is not providing the compounding benefit the proposal predicted — itself an informative finding to report.
- If GPA's advantage shrinks with N: informative negative result that fits the established narrative of compression being the bottleneck.

Any of these outcomes is usable. Plot as mean primary score vs. N with error bars (use the Step 3.3 seed variance where available; for N < 5, run 2 seeds per subset to get a rough variance estimate).

**Output:** `results/ablation_N/` + `dissertation/figures/ablation_N.pdf` (performance vs. N for 4 methods, with error bars).

#### Step 4.4-reduced — TIES on B̃ (Day 2, ~2 hours) — **MEDIUM PRIORITY**

The original Step 4.4 tested four B̃ merge strategies. Three (simple average, α=0.5, α=1.0) are already covered by the Week 3 enhancement ablation. Only the **TIES-on-B̃** variant is genuinely new.

Run one additional configuration: `dGPA + saTIES_on_A + saTIES_on_B` at the best λ from Week 3. This directly tests whether nonlinear conflict resolution on the output side outperforms reweighting.

**Key question:** Is output-side scale imbalance better addressed by reweighting (inverse-norm) or by conflict resolution (TIES on B̃)? If TIES-on-B beats reweighting, update the recommended pipeline. If not, the existing `wB(0.5)` remains the headline.

**Output:** one row added to the enhancement ablation table and a single paragraph in the experiments chapter.

#### Step 4.5 — Task Arithmetic in Aligned Space (Day 2, ~3 hours) — **MEDIUM PRIORITY**

This ablation gains new importance under the refined framing. It directly tests claim C2: does alignment matter when the merge is linear rather than TIES-style nonlinear?

Apply simple weighted averaging (Task Arithmetic style) to the aligned Ã factors, keeping the unaligned B̃ averaging. Compare:

- Unaligned TA (already have: best avg 0.451 at λ=1.0).
- GPA-aligned TA (new).
- Enhanced-GPA-aligned TA (new).

**Expected result based on KnOTS paper and Week 3 findings:** alignment provides little to no benefit for linear merges. If confirmed, this is clean evidence for C2 — alignment only pays off under nonlinear conflict resolution, which is where the downstream fragility also lives.

**Secondary benefit:** addresses the fairness concern around Task Arithmetic by showing that even when TA gets the aligned subspace, the compactness/compression gap is the decisive factor.

**Output:** `results/ablation_ta_aligned/` + one row in the main results table comparing unaligned TA, GPA-aligned TA, and enhanced-GPA-aligned TA.

#### Step 4.2-deferred — Vary LoRA Rank (Day 2–3, up to ~3 hours) — **LOW PRIORITY, CONDITIONAL**

Only run this if Step 4.1 finishes by end of Day 2 with time to spare. Otherwise, defer.

If run, scope is reduced: train adapters only at r = 32 (skip r = 8), only on SST-2 and CoLA (the two most informative tasks — SST-2 because it drives GPA's current gain, CoLA because it is the collapse case). Four new training runs, ~30 minutes total on the RTX 6000 Ada, plus evaluation.

**Narrow hypothesis:** if higher rank reduces the compression bottleneck, the gap between Task Arithmetic (rank-80 merged) and factor-space methods (rank-r merged) should shrink at higher r.

**If time-constrained or if Step 4.1 overruns: cut entirely.** The rank ablation is nice-to-have but does not anchor any of C1–C5.

#### Cut: Steps 4.3 (consensus replacement) and 4.5b (float16 vs. NF4)

Note in the dissertation's "future work" section that the consensus-replacement ablation and a quantization study are natural extensions, but were not pursued here because they do not probe the identified bottleneck (post-alignment compression and scale handling).

### Phase 3: CKA & Layer Analysis (Day 3–4, ~8 hours) — **HIGH PRIORITY**

Both remain unchanged in protocol from the original plan but gain importance because they are the primary evidence for C1 and C2 on real adapters (as opposed to synthetic data).

#### Step 4.6 — CKA Before/After Alignment (Day 3, ~4 hours)

Compute pairwise CKA between LoRA A matrices before and after GPA alignment; plot as heatmaps.

**Framing for interpretation:** this figure serves a specific argumentative purpose. It should show that GPA successfully increases pairwise similarity (supporting C1 on real adapters, not just synthetic) while the downstream merge still underperforms (supporting C2). If the CKA increases substantially but performance does not, that is exactly the figure the dissertation needs.

**Output:** `results/cka/` + `dissertation/figures/cka_before_after.pdf`.

#### Step 4.7 — Per-Layer Alignment Residual & Norm Correlation (Day 4, ~4 hours) — **EXPANDED**

In addition to recording per-layer convergence iterations and residuals, also correlate:

- Final alignment residual vs. **per-layer B-norm disparity** (Week 1 showed B-norm imbalance is largest in early MLP layers — does GPA struggle more there?).
- Final alignment residual vs. module family (attention vs. MLP).
- Final alignment residual vs. layer depth.

This analysis supports C3 directly by linking the Week 1 diagnostic (B-norm imbalance is structured by layer and module family) to the Week 2/3 alignment behaviour.

**Output:** `results/layer_analysis/` + `dissertation/figures/layer_residual_heatmap.pdf`.

### Phase 4: Qualitative Methodological Comparison (Day 4, ~1 hour) — **NEW**

#### Step 4.X — Method Properties Table

Construct a single table comparing the alignment stages of all compared methods on formal properties. No new experiments required — this is a synthesis of the literature and of your own implementation.

| Property | TIES | DARE+TIES | Task Arithmetic | LR-KnOTS+TIES | **GPA+TIES** |
|---|---|---|---|---|---|
| Has alignment stage | No | No | No | Yes (single-pass SVD) | **Yes (iterative)** |
| Iterative refinement | — | — | — | No | **Yes** |
| Formal convergence guarantee | — | — | — | N/A (closed-form) | **Yes (Gower 1975, Ling 2023)** |
| Equal-contribution by construction | No | No | No | No (norm-biased) | **Yes (in consensus step)** |
| Produces interpretable rotations | No | No | No | No (concatenation) | **Yes (per-adapter Q_i)** |
| Preserves output rank | No (rank-Nr) | No (rank-Nr) | No (rank-Nr) | Yes (rank-r) | **Yes (rank-r)** |
| Alignment overhead | — | — | — | One SVD | **~3 iterations, ms-scale** |

This table appears in the Methods chapter (not Results), supporting claim C4 by making GPA's unique methodological position visible without overstating empirical performance. Nearly impossible to contest because every entry is factual.

### Phase 5: Experiments & Results Chapter (Days 4–5, ~16 hours)

#### Step 4.8 — Write Experiments & Results (~12 hours)

Target ~5,000–6,000 words. The content list is largely unchanged but the **structure and framing** are revised to match the established claims.

**Recommended section ordering:**

1. **Experimental Setup** (unchanged).
2. **Synthetic Validation** — presents Experiments 1–4 as supporting C1. Lead with the main result; defer parameter sweeps to appendix.
3. **Real Adapter Structure (Week 1 analysis)** — positions the norm imbalance findings as the empirical foundation for C3 and motivates the Week 3 scale-aware enhancements.
4. **Main Results** — the Week 3 sweep table, with seed variance from Step 3.3. **Frame honestly:** state up front that enhanced GPA does not clear the proposal's 1pp threshold over raw TIES, but does improve modestly within the compact factor-space family.
5. **Enhancement Ablation** — the Week 3 ablation table, framed as primary evidence for C3 (only scale-aware B-weighting helps).
6. **CoLA Collapse & Rerun** — gets its own section. This is the most informative single result; treat it accordingly. Include the `max_iter=300` rerun as the key falsification result.
7. **N-Ablation** — the Week 4 result. Whatever it shows, it is a direct test of the proposal's multi-way hypothesis.
8. **Task-Arithmetic-in-Aligned-Space Ablation** — presents as evidence for C2.
9. **CKA & Per-Layer Analysis** — real-adapter evidence for C1 and C3.

**Tone guidance:** honest-positive. State the proposal's 1pp threshold is not met. Follow immediately with what the results *do* establish (C1–C5) and why these are substantive contributions. Do not apologise for the result; do not inflate it. Examiners respond well to clear-eyed analysis of partial results.

#### Step 4.9 — Generate All Experiment Figures (~4 hours)

Reduced figure list (tied directly to the claims):

1. Synthetic Experiment 1 heatmap (rotation recovery) — C1.
2. Synthetic Experiment 2 convergence curves — C1, supports the method-properties table.
3. Synthetic Experiment 3 non-orthogonal robustness — C1.
4. Synthetic Experiment 4 structured LoRA-like — C1.
5. Week 1 adapter norm ranking / heatmap — C3 motivation.
6. Main results bar chart (best per method, with seed error bars) — C4, C5.
7. Enhancement ablation bar chart — C3.
8. N-ablation lines (GPA vs. TIES vs. LR-KnOTS across N) — C2, C4.
9. CKA before/after heatmap pair — C1, C2.
10. Per-layer residual heatmap — C3.
11. CoLA prediction distribution across λ (from Week 3 diagnostic) — C5.

All figures must follow `dissertation_figure_style_guide.md`. Budget is 4 hours because most figures are already generated from Weeks 1–3; this step is mainly styling and polishing.

#### Step 4.10 — Begin Discussion Chapter Draft (~4 hours) — **NEW, MOVED FORWARD FROM WEEK 5**

Start drafting the Discussion chapter in parallel with the Experiments chapter. The core narrative (C1–C5) is already established, so the Discussion can be scaffolded now rather than waiting for Week 5. Aim for ~1,500 words of rough draft covering sections 1–4 of the Week 5 Discussion outline (interpretation against four outcomes; C1, C2, C3 contributions). Polish and complete in Week 5.

This front-loading is essential because the refined plan budgets only 5 days in Week 5 for assembly + submission; Discussion writing cannot start cold on 28 April.

### Week 4 Deliverables Checklist (Revised)

| # | Deliverable | Location | Status |
|---|---|---|---|
| 1 | Completed 3-seed results for primary methods + headline enhanced variant | `results/seed_variance.json` | **Must** |
| 2 | N-ablation results & figure | `results/ablation_N/`, `dissertation/figures/ablation_N.pdf` | **Must** |
| 3 | TIES-on-B̃ single-config result | appended to enhancement ablation table | **Should** |
| 4 | TA-in-aligned-space result | `results/ablation_ta_aligned/` | **Should** |
| 5 | CKA before/after analysis | `results/cka/` + figure | **Must** |
| 6 | Per-layer residual + norm-correlation analysis | `results/layer_analysis/` + figure | **Must** |
| 7 | Qualitative methodological comparison table | in Methods chapter draft | **Should** |
| 8 | Rank ablation (r = 32 only, SST-2 + CoLA only) | `results/ablation_rank/` | **Conditional** |
| 9 | Experiments & Results chapter draft (~5,000–6,000 words) | `dissertation/chapters/experiments.md` | **Must** |
| 10 | Discussion chapter rough draft (~1,500 words, sections 1–4) | `dissertation/chapters/discussion.md` | **Should** |
| 11 | All figures in final style | `dissertation/figures/` | **Must** |

**Cut (vs. original Week 4 plan):** consensus-replacement ablation; float16 vs. NF4 ablation; three of four B̃ merge strategy variants.

---

## Week 5 (28 Apr – 1 May): Discussion, Conclusion, Final Assembly, Submission

**Hours budget:** ~50 hours.
**Goal:** Complete and submit by 1 May.
**Key change vs. original plan:** The Discussion chapter is partially drafted during Week 4 Step 4.10 in parallel, because the core narrative (C1–C5) is already established. Week 5 is polishing and assembly, not discovery.

> **Zero new experiments in Week 5.** Any gap identified during writing must be addressed by reframing, not by running more code.

### Phase 1: Discussion Chapter (Days 1–2, ~15 hours)

#### Step 5.1 — Write / Complete Discussion (~10 hours)

Target ~3,500–4,000 words. Revised outline reflecting what the results actually support:

1. **Interpretation of Results against the Proposal's Four Outcomes (~500 words).**
   The results map to **Outcome 3** (primary hypothesis only weakly confirmed): the enhanced GPA variant improves over raw TIES on average, but the gain is small, below the 1pp threshold, and concentrated in SST-2. The secondary hypothesis (GPA vs. LR-KnOTS) is narrowly supported. State this directly.

2. **Contribution 1 — GPA as a Validated Alignment Primitive (~600 words).**
   Anchor in synthetic Experiments 1–4 and the `max_iter=300` rerun. Argue that the alignment problem in low-rank factor space is solvable, and GPA is a principled, reproducible way of solving it. Cite Gower (1975) and Ling (2023) for the formal backing. Tie in the qualitative methodological comparison table from the Methods chapter.

3. **Contribution 2 — Separating Alignment Quality from Merge Quality (~700 words).**
   The central intellectual contribution. Synthetic success + real-task gap + rerun falsification of the convergence explanation = strong evidence that the community has been conflating two problems. Reference the TA-in-aligned-space ablation (alignment does not help linear merges) and the N-ablation (whatever it shows) to sharpen this.

4. **Contribution 3 — Scale Imbalance as the Binding Constraint (~700 words).**
   Chain the Week 1 empirical diagnosis (B-norm imbalance, MLP vs. attention, layer-local structure) to the Week 3 ablation (only scale-aware B-weighting helps) to the per-layer analysis (high residual correlates with high norm disparity, if the Week 4 analysis shows this). This is the most actionable finding for future work.

5. **Contribution 4 — CoLA as a Diagnostic (~400 words).**
   Universal collapse across all methods and all scales. The rerun shows it is not a convergence issue. The non-zero DARE+TIES CoLA at a non-selected λ shows it is not absolutely impossible. Argue that CoLA exposes a task-specific fragility that compact merging methods should be expected to address, not to average over.

6. **Fairness of Comparison (~300 words).**
   Task Arithmetic's rank-80 merged adapter vs. factor-space rank-16 outputs is not like-for-like. Within the compact family, enhanced GPA is competitive. This is the honest framing of claim C4.

7. **Computational & Practical Properties of GPA (~300 words).**
   Short subsection (not a headline contribution) noting GPA's convergence guarantees, equal-contribution property, and interpretable rotations, drawing on the methodological comparison table. Explicitly **not** framed as a speed story; framed as principled properties that make GPA easier to reason about and extend.

8. **Limitations (~400 words).**
   Single model family (Qwen2.5-1.5B). Single dataset (GLUE subset). Orthogonality assumption violated to unknown degree in real adapters. Inverse-norm α is a hyperparameter. Single seed on most Week 3 configurations (acknowledge and state which configurations were re-run with seeds). Fixed rank r=16 as primary experimental regime (with narrow r=32 check if Step 4.2 was run).

9. **Future Work (~400 words).**
   Directly follows from the contributions. Scale-aware merge stages are the next frontier (C3 → actionable next step). Adaptive per-layer α. Extension to heterogeneous architectures. Application to vision LoRA. Probing task-specific fragility beyond CoLA.

#### Step 5.2 — Write Conclusion (~2 hours)

~600–800 words. Three paragraphs:

1. What the dissertation set out to test and the result (C1 yes, C2–C5 as positive substantive findings, full proposal hypothesis only weakly confirmed).
2. The most useful single takeaway for the field: alignment and merge are separable; future factor-space work should prioritise the merge stage.
3. The immediate next research step the dissertation enables.

#### Step 5.3 — Write Abstract (~1 hour)

~250 words. Must be written last, after Discussion is final. Structure:

- Problem (1 sentence).
- Approach (GPA + scale-aware enhancements, 2 sentences).
- Key results (synthetic validation strong; real-task partial; scale is the bottleneck, 2 sentences).
- Contribution framing (separating alignment from merge quality, 1 sentence).
- Implication (1 sentence).

#### Step 5.4 — Positive-Framing Pass (~2 hours) — **REPLACES ORIGINAL NEGATIVE-RESULT PROTOCOL**

The results are not negative. They are partial-positive with a clear intellectual contribution. The original plan's "negative result protocol" step was written for a harsher outcome than occurred. Replace it with a **positive-framing pass**:

- Read the full Discussion and Conclusion in one sitting.
- For each paragraph, check: does it lead with what the results *show*, or does it lead with what they *fail to show*? Revise any paragraph that leads with a failure framing to instead lead with the positive finding, with the limitation following.
- Example: "The enhanced GPA variant does not exceed raw TIES by the 1pp threshold set in the proposal" → "Enhanced GPA modestly exceeds raw TIES on average, with the gain concentrated in SST-2; it does not clear the proposal's 1pp threshold, which we return to in Section X."
- Confirm that every limitation paragraph is followed by a constructive statement (future work hook, or re-grounding in what the evidence *does* support).

This is not spin. It is ensuring that the reader finishes each section knowing what the dissertation established, not what it did not.

### Phase 2: Full Dissertation Assembly (Days 2–3, ~15 hours)

#### Step 5.5 — Compile and Cross-Reference (~4 hours)

- Assemble all chapters into the final document.
- Number all figures, tables, and equations consistently.
- Ensure all citations are correct and complete.
- Add table of contents, list of figures, list of tables.

#### Step 5.6 — Final Figures and Tables (~4 hours)

- Review every figure for readability, axis labels, legends, colour consistency per `dissertation_figure_style_guide.md`.
- Ensure all tables have captions and are referenced in the text.
- Convert all matplotlib figures to PDF with consistent styling.

#### Step 5.7 — Proofread and Revise (~7 hours)

- Full read-through for clarity, grammar, and logical flow.
- Check that every claim is supported by evidence (a figure, table, or citation) AND ties back to C1–C5.
- Verify all numbers in tables match the actual experimental results.
- Remove any placeholder text or TODOs.
- Confirm no framing overreaches beyond what the data support.

### Phase 3: Submission (Day 4–5, ~5 hours)

#### Step 5.8 — Code Cleanup and Documentation (~3 hours)

- Clean up all scripts, add docstrings and usage instructions.
- Create a `README.md` for the codebase.
- Ensure the codebase is reproducible: include exact commands to replicate every result.

#### Step 5.9 — Final Submission (~2 hours)

- Export dissertation as PDF.
- Verify formatting against university requirements.
- Submit before the 1 May deadline.
- Archive the complete codebase and results.

### Week 5 Deliverables Checklist

| # | Deliverable | Location |
|---|-------------|----------|
| 1 | Discussion chapter (final) | `dissertation/chapters/discussion.md` |
| 2 | Conclusion | `dissertation/chapters/conclusion.md` |
| 3 | Abstract | `dissertation/abstract.md` |
| 4 | Complete assembled dissertation | `dissertation/dissertation.pdf` |
| 5 | All figures finalised | `dissertation/figures/` |
| 6 | Clean, documented codebase | `gpa-lora-merge/` + `README.md` |
| 7 | **SUBMITTED DISSERTATION** | University submission portal |

---

## Summary of Changes from Original Plan

| Aspect | Original Plan | Revised Plan |
|--------|--------------|--------------|
| Hardware | 6× RTX 2080 (8 GB each) | **1× RTX 6000 Ada (48 GB)** |
| Synthetic GPA validation | Week 2 | **Week 1, Days 1–4** (parallel with infra) |
| GPA implementation start | Week 3 | **Week 1, Day 1** (core algorithm for synthetic) |
| Adapter training | Parallel across 6 GPUs | **Sequential on 1 GPU** (larger batches compensate) |
| Baseline merging (TIES, DARE, TA) | Week 2 | Week 2 (unchanged) |
| GPA on real adapters | Week 3 | **Week 2, Days 3–4** (one week earlier) |
| Dissertation writing begins | Week 3 | **Week 2, Day 5** (one week earlier) |
| Full experiment matrix | Week 4 | **Week 3** (one week earlier) |
| Ablations + CKA | Week 5 | **Week 4** |
| Dissertation synthesis | Week 6 | **Week 5** (must finish by 1 May) |
| Total duration | 6 weeks | **5 weeks** (compressed to meet deadline) |

### Summary of Weeks 4–5 Refinement (added 16 April 2026, post-Week-3)

| Aspect | Pre-Refinement | Post-Refinement | Rationale |
|---|---|---|---|
| Week 4 ablation count | 7 | 5 kept, 2 cut, 1 added | Anchor every experiment to one of C1–C5 |
| Consensus replacement ablation | Included | **Cut** | Does not support any core claim |
| Float16 vs. NF4 ablation | Optional | **Cut** | Does not support any core claim |
| B̃ merge strategy ablation | 4 variants | 1 new variant (TIES-on-B̃) | Other variants already covered by Week 3 enhancement ablation |
| Rank ablation | r ∈ {8, 32}, 3 tasks | r = 32 only, 2 tasks, conditional | Narrow hypothesis, not anchoring any core claim |
| Methodological comparison table | Not present | **Added** (~1 hour) | Supports C1, C4 without overreaching on performance |
| Discussion draft start | Week 5 | **Week 4 Step 4.10** (rough draft, sections 1–4) | Core narrative (C1–C5) already established; protects Week 5 buffer |
| Step 5.4 | Negative-result protocol | **Positive-framing pass** | Results are partial-positive, not negative |
| Experiments chapter ordering | Flat list | Restructured to tell C1→C5 story | Reader encounters claims in order of increasing specificity |
| Step 3.3 seed coverage | 3 primary methods | 3 primary methods **+ enhanced GPA** | Headline gain small enough to need variance bars |

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GPA fails on synthetic data (Exp 3, δ > 0.1) | Medium | **Critical** | Detected in Week 1 Day 3. Pivot to analysing failure modes. Dissertation becomes a negative-result study. |
| Adapter training fails (OOM, NaN) | **Very Low** | Medium | 48 GB VRAM eliminates OOM risk for this model. NaN: debug learning rate. Smoke test on Day 2 catches issues early. |
| Sequential training overruns | Low | Medium | The RTX 6000 Ada is fast enough that all 5 adapters train in ~5 hours. Run overnight if needed. Fallback: drop MNLI for N = 4. |
| GPA+TIES ≤ TIES on real adapters | Medium | Medium | Follow negative result protocol (Section 4.2 of proposal). Synthetic results still provide value. |
| Scale-aware enhancements don't rescue CoLA | Medium | Medium | The λ sweep may still rescue CoLA at low λ. If CoLA remains collapsed even with scale-aware enhancements at all λ, the diagnosis shifts from "scale bias" to "fundamental task incompatibility during merging" — this is itself an informative finding for the dissertation. |
| Inverse-norm B̃ weighting degrades large-task performance | Medium | Low | The α exponent provides a dial. If α=1.0 hurts MNLI/SST-2, use α=0.5 as a compromise, or report the trade-off curve explicitly. |
| Enhanced sweep evaluation time overruns | Low | Medium | Prioritise the full enhancement stack (`dGPA+saTIES+wB(1.0)`) vs. baseline. Drop intermediate variants if time is tight. Merging is CPU-milliseconds; only evaluation is costly. |
| Writing takes longer than planned | High | High | Writing starts Week 2 (not Week 3). Discussion pre-written with conditional framing. |
| CUDA 12.x compatibility issues | Low | Medium | Test `bitsandbytes` and `peft` on CUDA 12.4 in the smoke test. If issues arise, try `cu121` instead. |
| University formatting issues | Low | Medium | Check formatting requirements in Week 4, not Week 5. |
| *(Added after Week 3)* Seed variance swamps enhanced-GPA gain over raw TIES | Medium | Low | Report directly. The dissertation's central claims (C1–C5) do not depend on this specific margin. |
| *(Added after Week 3)* N-ablation shows GPA advantage does not grow with N | Medium | Low | Report as finding that contradicts proposal expectation; fits the established narrative that alignment is not the bottleneck. |
| *(Added after Week 3)* TA-in-aligned-space unexpectedly improves TA | Low | Low | Would strengthen C2 in a surprising direction (alignment does help linear merges contra KnOTS); report as informative finding. |
| *(Added after Week 3)* Per-layer residual shows no correlation with norm disparity | Low | Medium | Weakens C3 slightly. Fall back to Week 1 empirical evidence for C3 and note the per-layer behaviour as a future-work direction. |
| *(Added after Week 3)* Week 5 writing overruns | Medium | High | Discussion drafting begins in Week 4 Step 4.10 (sections 1–4 of outline). Hold Days 4–5 of Week 5 explicitly for submission. |
