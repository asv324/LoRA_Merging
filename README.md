# LoRA Merging Dissertation Artifacts

This repository contains a reproducible subset of code, logs, results, and dissertation artifacts for experiments on compact factor-space LoRA adapter merging.

The project compares task arithmetic, TIES, DARE+TIES, LR-KnOTS, and GPA-enhanced merging methods on Qwen2.5 GLUE adapters. It also includes synthetic experiments used to validate the Generalized Procrustes Analysis (GPA) alignment primitive.

## Repository Layout

- `Documentation/` - implementation notes, analysis plans, and artifact documentation.
- `dissertation/` - generated dissertation chapter material, result tables, and manifests.
- `logs/` - selected experiment logs.
- `results/` - saved JSON outputs from synthetic experiments, sweeps, ablations, and evaluations.
- `scripts/` - training, merging, evaluation, analysis, and plotting scripts.
- `tests/` - focused unit and regression tests for the scripts.
- `Dockerfile` - CUDA/PyTorch environment with the pinned ML and analysis dependencies.
- `test.py` - quick CUDA/GPU availability check.

## Environment

The recommended environment is the provided Docker image:

```bash
docker build -t lora-merging .
docker run --gpus all -it --rm -v "$PWD:/workspace" -w /workspace lora-merging bash
```

Then verify GPU access:

```bash
python test.py
```

The Docker image installs PyTorch with CUDA, Hugging Face Transformers, PEFT, bitsandbytes, datasets/evaluate, SciPy, scikit-learn, pandas, matplotlib, and related utilities.

## Common Commands

Run the test suite:

```bash
python -m unittest discover tests
```

Run a small training job:

```bash
python scripts/train.py --task sst2 --output_dir adapters/sst2 --max_train_samples 128 --max_eval_samples 128
```

Evaluate a merged adapter:

```bash
python scripts/evaluate_merged.py --adapter_dir merged_adapters/example --output_path results/example_eval.json
```

Regenerate dissertation result artifacts from saved outputs:

```bash
python scripts/generate_dissertation_results_artifacts.py
```

## Notes

Some scripts expect trained adapter directories such as `adapters/` or `merged_adapters/`. These large model artifacts are not part of this subset and must be generated locally or supplied separately before rerunning full training, merging, or evaluation pipelines.

The committed `results/`, `logs/`, and `dissertation/` artifacts are intended to make the experimental record inspectable without rerunning every GPU-heavy job.
