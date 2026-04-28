# Adapter Analysis Figures

This note describes the figure bundle generated from:

- `configs/lora_param_mapping.json`
- `results/adapter_norm_analysis.json`

These figures are intended to support the dissertation hypothesis that real LoRA adapters differ in scale and structure before any alignment-aware merging is applied, which motivates GPA-based processing in Week 2.

## How to regenerate

Run:

```bash
python scripts/plot_adapter_analysis.py
```

Optional flags:

- `--mapping-input` to override the mapping JSON path
- `--analysis-input` to override the norm-analysis JSON path
- `--output-dir` to change the figure destination
- `--dpi` to change PNG resolution

The script writes PDF and PNG outputs plus `results/figures/adapter_analysis_manifest.json`.

## Figures

### `adapter_norm_ranking`

What it shows:
- grouped bars for average `LoRA A` and `LoRA B` Frobenius norms per task
- tasks ordered by overall norm magnitude

Why it matters:
- this is the clearest direct visualization of the norm imbalance called out in Step `B.6`
- it supports the claim that independently trained adapters do not contribute at comparable scale

How to use it in the dissertation:
- introduce it as baseline descriptive evidence that raw adapters are not balanced contributors before merging
- connect it to the motivation for alignment-aware or equal-contribution merging

### `adapter_norm_attention_vs_mlp`

What it shows:
- separate `LoRA A` and `LoRA B` panels
- attention-family averages versus MLP-family averages for each task

Why it matters:
- shows that the imbalance is structured across module families rather than being only a single task-wide scale effect
- helps motivate later layerwise alignment analysis

How to use it in the dissertation:
- use it to argue that adapter differences are architectural as well as global
- connect it to the expectation that some module families may benefit more from factor-space alignment than others

### `adapter_layer_heatmap_A`

What it shows:
- per-layer `LoRA A` norm heatmaps across all tasks
- split into attention and MLP panels using the canonical layer ordering from `configs/lora_param_mapping.json`

Why it matters:
- reveals whether larger-norm tasks are consistently larger across the network or only in selected regions
- provides stronger evidence that raw factor-space magnitude varies locally, not just globally

How to use it in the dissertation:
- use it as the strongest descriptive figure for heterogeneous pre-alignment structure
- cite it when motivating later GPA, CKA, and residual analyses

### `adapter_depth_trends`

What it shows:
- mean `LoRA A` norm versus transformer depth for each task
- separate attention and MLP panels

Why it matters:
- shows whether tasks concentrate adaptation at different depths
- suggests where alignment pressure may differ across the network

How to use it in the dissertation:
- use it to frame later questions about whether deeper layers are harder to align or more task-specific
- pair it with future per-layer GPA diagnostics

### `adapter_perf_vs_norm`

What it shows:
- exploratory scatter of combined average adapter norm against normalized standalone validation performance

Why it matters:
- asks whether stronger standalone adapters are simply the largest ones
- if the relationship is weak, that supports the idea that norm magnitude alone is not a sufficient summary of adapter usefulness

How to use it in the dissertation:
- treat this as a secondary or appendix figure unless the trend is especially clear

## Expected output files

- `results/figures/adapter_norm_ranking.pdf`
- `results/figures/adapter_norm_ranking.png`
- `results/figures/adapter_norm_attention_vs_mlp.pdf`
- `results/figures/adapter_norm_attention_vs_mlp.png`
- `results/figures/adapter_layer_heatmap_A.pdf`
- `results/figures/adapter_layer_heatmap_A.png`
- `results/figures/adapter_depth_trends.pdf`
- `results/figures/adapter_depth_trends.png`
- `results/figures/adapter_perf_vs_norm.pdf`
- `results/figures/adapter_perf_vs_norm.png`
- `results/figures/adapter_analysis_manifest.json`

## Dissertation narrative

The core narrative should use the first four figures in sequence:

1. `adapter_norm_ranking` shows overall task-level imbalance.
2. `adapter_norm_attention_vs_mlp` shows that the imbalance is structured by module family.
3. `adapter_layer_heatmap_A` shows the imbalance is layer-local and heterogeneous.
4. `adapter_depth_trends` shows that tasks exhibit distinct depth profiles even before any merging.

Together, these figures provide a clean motivation for testing whether GPA-based alignment can reduce the impact of raw factor-scale differences before nonlinear merging methods such as TIES are applied.
