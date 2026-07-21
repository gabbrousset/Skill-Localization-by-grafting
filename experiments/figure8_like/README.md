# Figure 8-like Grafting Masks

The paper's Figure 8 is a multi-task analysis of task-specific grafting regions:
rows are task masks, columns are evaluation tasks, and panel (a) reports the
asymmetric overlap `|gamma_i intersect gamma_j| / |gamma_j|`.

This repository does not include the original multi-task training or plotting
pipeline. The scripts here provide a reproducible Figure 8-like workflow using
the checked-in single-task fine-tuning and grafting code:

1. Fine-tune one checkpoint per task and seed.
2. Train one zero-base graft mask per task and seed.
3. Compute a pairwise mask-overlap CSV from the saved graft mask checkpoints.

The default task list matches the Figure 8 tasks that are present in the local
LM-BFF data tree:

```text
SST-2 cr SNLI MNLI QNLI QQP MRPC
```

The paper also includes AG News (`ag_news`/AGN). Add it to `TASKS` only after
preparing `../data/k-shot/ag_news/<K>-<SEED>/`.

## Local or interactive run

From the repository root:

```bash
bash experiments/figure8_like/run_mask_grid.sh
```

Useful overrides:

```bash
TASKS="SST-2 cr" SEEDS="100" K=16 bash experiments/figure8_like/run_mask_grid.sh
RUN_FINETUNE=false bash experiments/figure8_like/run_mask_grid.sh
RUN_GRAFT=false bash experiments/figure8_like/run_mask_grid.sh
FORCE=true bash experiments/figure8_like/run_mask_grid.sh
RUN_ARGS="--no_cuda --no_predict" GRAFT_ARGS="--no_cuda" TASKS="SST-2" SEEDS="0" FT_MAX_STEP=1 GRAFT_EPOCHS=1 MODEL=sshleifer/tiny-distilroberta-base MODELNAME=sshleifer/tiny-distilroberta-base bash experiments/figure8_like/run_mask_grid.sh
```

Defaults:

```text
K=16
SEEDS="100 13 21 42 87"
MODEL=roberta-base
TYPE=prompt
FT_MAX_STEP=1000
GRAFT_EPOCHS=10
MASK_BASE=zero
SPARSITY_LEVEL=0
```

Outputs:

```text
ckpt_paths/figure8_like/finetuned/<TASK>/k<K>/seed-<SEED>/roberta-base/
ckpt_paths/figure8_like/masks/<TASK>/k<K>/seed-<SEED>/roberta-base/mask_zero.pt
ckpt_paths/figure8_like/graft_eval/<TASK>/k<K>/seed-<SEED>/roberta-base/
```

Slashes in `MODELNAME` are converted to `__` for output paths.

## Slurm run

Submit one task/seed job at a time:

```bash
bash experiments/figure8_like/submit_slurm_mask_grid.sh
```

Override Slurm resources without editing the script:

```bash
SBATCH_ARGS="--partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=24:00:00" \
bash experiments/figure8_like/submit_slurm_mask_grid.sh
```

If the base model is already cached at `model_files/roberta-base`, the script
uses that local path and disables Hugging Face resolution. Otherwise it lets
`run_experiment.sh` download a compatible local model cache.

## Overlap CSV

After masks are generated:

```bash
uv run python tools/mask_overlap.py \
  --mask-dir ckpt_paths/figure8_like/masks \
  --output-csv ckpt_paths/figure8_like/masks/overlap.csv
```

The output includes `overlap_fraction`, computed as:

```text
|row_mask intersect column_mask| / |column_mask|
```

Only compare masks trained from the same model architecture and trainable
parameter setting.
