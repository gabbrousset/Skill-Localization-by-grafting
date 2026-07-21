## Task-Specific Skill Localization in Fine-tuned Language Models

This is the code repository for the paper [Task-Specific Skill Localization in Fine-tuned Language Models](https://arxiv.org/abs/2302.06600), which appeared at ICML 2023.

## Overview

Pre-trained language models can be fine-tuned to solve diverse NLP tasks, including in few-shot settings. This repository contains the fine-tuning and grafting code used to localize small task-specific parameter subsets in fine-tuned language models.

The main workflow is:

1. Prepare an environment with `uv`.
2. Prepare LM-BFF-style original data.
3. Generate k-shot splits.
4. Fine-tune a model with `run_experiment.sh` or `run_task.sh`.
5. Run grafting with `run_graft_experiment.sh` or `run_graft_task.sh`.

## Repository Setup

Clone the repository and enter it:

```bash
git clone git@github.com:gabbrousset/Skill-Localization-by-grafting.git
cd Skill-Localization-by-grafting
```

Install [uv](https://docs.astral.sh/uv/) if needed. Then create the locked Python 3.8 environment:

```bash
uv sync --locked
```

On clusters where `$HOME/.cache` is not writable, use a writable cache directory:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --locked
```

On Apple Silicon macOS, use the same command. The lockfile resolves `torch==2.4.1`
from PyPI so uv can install the native macOS arm64 wheel instead of the Linux CUDA
wheel:

```bash
uv sync --locked
```

This repository is still a Python 3.8 research-code environment with several old
compiled dependencies. If native Apple Silicon installation fails later on a
package such as `numpy`, `scipy`, `scikit-learn`, `pandas`, `sentencepiece`, or
`tokenizers`, use an x86_64/Rosetta Python 3.8 environment on that Mac, or run the
full experiments on a Linux GPU machine. Full `roberta-base` fine-tuning and
grafting are expected to run on Linux GPU nodes.

Check that the Python entrypoints import correctly:

```bash
uv run python -c "import run, Grafting; print('entrypoint imports ok')"
```

The shell scripts use `uv run --project . python` automatically when `uv` is available. To use an already activated Conda or virtualenv environment instead, run with:

```bash
PYTHON_BIN=python bash run_task.sh
```

The original Conda environment remains available for legacy reproduction:

```bash
conda env create -n icl_as_ft --file skills_environment.yml
conda activate icl_as_ft
```

## Data Setup

Download the original datasets using the LM-BFF instructions: https://github.com/princeton-nlp/LM-BFF#prepare-the-data

Recommended layout from the parent directory:

```text
skill-inference/
  data/
    original/
      SST-2/
      sst-5/
      ...
  Skill-Localization-by-grafting/
```

From inside `Skill-Localization-by-grafting`, generate k-shot splits for all supported tasks:

```bash
for task in SST-2 sst-5 mr cr mpqa subj trec CoLA MRPC QQP STS-B MNLI SNLI QNLI RTE; do
  uv run python tools/generate_k_shot_data.py \
    --data_dir ../data/original \
    --output_dir ../data \
    --task "$task" || exit 1
done
```

The command above creates the default seeds `100`, `13`, `21`, `42`, and `87`. The default `run_task.sh` uses `SST-2` seed `0`, so also generate that split:

```bash
uv run python tools/generate_k_shot_data.py \
  --data_dir ../data/original \
  --output_dir ../data \
  --task SST-2 \
  --seed 0
```

The expected k-shot layout is:

```text
../data/k-shot/<TASK>/<K>-<SEED>/
```

For example:

```text
../data/k-shot/SST-2/16-0/train.tsv
../data/k-shot/SST-2/16-0/dev.tsv
../data/k-shot/SST-2/16-0/test.tsv
```

The run scripts automatically use `../data/k-shot` if it exists. Otherwise they use `data/k-shot` inside the repository. Override paths explicitly when needed:

```bash
DATA_ROOT=/path/to/k-shot
LOG_ROOT=/path/to/log_files
CKPT_ROOT=/path/to/ckpt_paths
MODEL_CACHE_DIR=/path/to/model_cache
```

Runtime outputs are ignored by git:

```text
ckpt_paths/
log_files/
model_files/
wandb/
runs/
```

## Model Download Compatibility

This code uses `transformers==3.4.0`. Some built-in pretrained-weight URLs in that old Transformers version are stale. The shell scripts therefore call `tools/cache_hf_model.py` to download base models into `MODEL_CACHE_DIR` and pass a local model path to Python.

This happens automatically for model names such as `roberta-base`, `gpt2`, and namespaced smoke-test models. Disable it only when you already pass local model paths:

```bash
RESOLVE_HF_MODELS=false bash run_experiment.sh
```

## Quick Smoke Test

Run this CPU smoke test before launching a full experiment. It uses a tiny RoBERTa-compatible checkpoint and performs one SST-2 training step plus validation:

```bash
TAG=smoke \
TYPE=prompt \
TASK=SST-2 \
K=16 \
BS=2 \
LR=1e-3 \
SEED=0 \
modelseed=0 \
uselmhead=1 \
useCLS=0 \
max_step=1 \
fixhead=True \
fixembeddings=True \
MODEL=sshleifer/tiny-distilroberta-base \
train_bias_only=False \
MODELNAME=sshleifer/tiny-distilroberta-base \
bash run_experiment.sh --no_predict --save_at_last --logging_steps 1 --eval_steps 1 --no_cuda
```

Expected result: the command exits with status `0`, writes a checkpoint under `ckpt_paths/log_noembed_SGD_graft/`, and appends validation metrics to `log_files/log_noembed_SGD_graft`.

## Full Fine-tuning

For the default SST-2 prompt experiment with `roberta-base`, run:

```bash
bash run_task.sh
```

`run_task.sh` currently uses:

```text
TASK=SST-2
K=16
SEED=0
MODEL=roberta-base
TYPE=prompt
max_step=1000
LR=1e-3
```

A custom single-task run looks like this:

```bash
TAG=exp \
TYPE=prompt \
TASK=SST-2 \
K=16 \
BS=2 \
LR=1e-3 \
SEED=0 \
modelseed=0 \
uselmhead=1 \
useCLS=0 \
max_step=1000 \
fixhead=True \
fixembeddings=True \
MODEL=roberta-base \
train_bias_only=False \
MODELNAME=roberta-base \
bash run_experiment.sh
```

To launch the same RoBERTa prompt setup for the default seeds across all generated tasks:

```bash
tasks=(SST-2 sst-5 mr cr mpqa subj trec CoLA MRPC QQP STS-B MNLI SNLI QNLI RTE)
seeds=(100 13 21 42 87)

for task in "${tasks[@]}"; do
  for seed in "${seeds[@]}"; do
    TAG=exp \
    TYPE=prompt \
    TASK="$task" \
    K=16 \
    BS=2 \
    LR=1e-3 \
    SEED="$seed" \
    modelseed=0 \
    uselmhead=1 \
    useCLS=0 \
    max_step=1000 \
    fixhead=True \
    fixembeddings=True \
    MODEL=roberta-base \
    train_bias_only=False \
    MODELNAME=roberta-base \
    bash run_experiment.sh || exit 1
  done
done
```

Full `roberta-base` runs should be launched on a GPU node. The tiny smoke test is only for verifying setup.

## Fine-tuning Arguments

Important environment variables accepted by `run_experiment.sh`:

* `TYPE`: `finetune`, `prompt`, or `autoregressive`
* `TASK`: task name, for example `SST-2`, `sst-5`, `MRPC`, `MNLI`
* `K`: number of training examples per class
* `BS`: batch size
* `LR`: learning rate
* `SEED`: k-shot data seed
* `modelseed`: model initialization/training seed
* `MODEL`: base model name or local model path, for example `roberta-base` or `gpt2`
* `MODELNAME`: output-name version of the model; defaults to `MODEL`
* `uselmhead`: use language-model head, usually `1` for prompt runs
* `useCLS`: use CLS linear head
* `fixembeddings`: freeze embeddings
* `fixhead`: freeze LM head
* `train_bias_only`: train only bias parameters
* `max_step`: maximum training steps

## Grafting

Grafting requires:

1. A pretrained base model, for example `roberta-base`.
2. A fine-tuned checkpoint created by the fine-tuning step.
3. The same task/data settings used for fine-tuning.

Find a fine-tuned checkpoint:

```bash
find ckpt_paths/log_noembed_SGD_graft -mindepth 1 -maxdepth 1 -type d | sort
```

Edit `model_path` in `run_graft_task.sh` to point to the fine-tuned checkpoint, then run:

```bash
bash run_graft_task.sh
```

A custom grafting command:

```bash
model_path="ckpt_paths/log_noembed_SGD_graft/SST-2-prompt-16-0-roberta-base-<trial>-2-1e-3"

TAG=exp \
TYPE=prompt \
TASK=SST-2 \
K=16 \
LR=1e7 \
SEED=0 \
MODEL="$model_path" \
uselmhead=1 \
useCLS=0 \
num_train_epochs=10 \
mask_path=highest_movement \
sparsitylevel=1e-7 \
pretrained_model=roberta-base \
fixhead=True \
fixembeddings=True \
truncate_head=True \
train_bias_only=False \
no_train=False \
checkpoint_location=ckpt_paths/graft_mask_SST-2.pt \
bash run_graft_experiment.sh
```

To evaluate an existing saved graft mask instead of training one, set:

```bash
no_train=True
checkpoint_location=/path/to/saved_mask.pt
```

## Grafting Arguments

Important grafting variables:

* `MODEL`: fine-tuned checkpoint path
* `pretrained_model`: base model name or local base model path
* `mask_path`: `highest_movement` or a path to an existing mask
* `sparsitylevel`: basepatch sparsity level
* `checkpoint_location`: where to save or load the trained mask
* `no_train`: `False` to train a mask, `True` to load `checkpoint_location`

## Troubleshooting

If `uv` tries to use a parent workspace with the wrong Python version, run commands from this repository root. This repository declares its own uv workspace in `pyproject.toml`.

If `$HOME/.cache/uv` is read-only, use:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --locked
```

If a run fails with missing data, check:

```bash
find ../data/k-shot -maxdepth 3 -type d | sort | head
```

If old Transformers cannot load `roberta-base` directly, leave `RESOLVE_HF_MODELS` enabled so `tools/cache_hf_model.py` downloads a local compatible layout.

If uv fails on macOS Apple Silicon with an error like `torch==2.4.1+cu121` has no
wheel for `macosx_..._arm64`, pull the latest repository changes and rerun
`uv sync --locked`. Older lockfiles forced the Linux CUDA PyTorch index for every
platform; the current lock uses PyPI so uv can select the macOS arm64 torch wheel.

If a full `roberta-base` run is killed during model loading, use a GPU job with enough memory. The CPU smoke test uses a tiny model to validate setup only.

If W&B creates unwanted local logs, the wrappers set `WANDB_DISABLED=true` and `WANDB_MODE=disabled` by default. Override those variables if you want W&B logging.

## Citation

Please cite the paper if you make use of this code:

```bibtex
@article{panigrahi2023task,
  title={Task-Specific Skill Localization in Fine-tuned Language Models},
  author={Panigrahi, Abhishek and Saunshi, Nikunj and Zhao, Haoyu and Arora, Sanjeev},
  journal={arXiv preprint arXiv:2302.06600},
  year={2023}
}
```
