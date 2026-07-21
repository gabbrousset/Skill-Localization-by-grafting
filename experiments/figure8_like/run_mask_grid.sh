#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

TASKS=${TASKS:-"SST-2 cr SNLI MNLI QNLI QQP MRPC"}
SEEDS=${SEEDS:-"100 13 21 42 87"}
K=${K:-16}
TYPE=${TYPE:-prompt}
MODELNAME=${MODELNAME:-roberta-base}
MODELNAME_SAFE=${MODELNAME//\//__}
MODEL=${MODEL:-roberta-base}

if [ -d "$REPO_ROOT/model_files/roberta-base" ] && [ "$MODEL" = "roberta-base" ]; then
  MODEL="$REPO_ROOT/model_files/roberta-base"
  RESOLVE_HF_MODELS=${RESOLVE_HF_MODELS:-false}
else
  RESOLVE_HF_MODELS=${RESOLVE_HF_MODELS:-true}
fi

DATA_ROOT=${DATA_ROOT:-../data/k-shot}
CKPT_ROOT=${CKPT_ROOT:-ckpt_paths}
LOG_ROOT=${LOG_ROOT:-log_files}
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-model_files}
FT_ROOT=${FT_ROOT:-$CKPT_ROOT/figure8_like/finetuned}
MASK_ROOT=${MASK_ROOT:-$CKPT_ROOT/figure8_like/masks}
GRAFT_OUTPUT_ROOT=${GRAFT_OUTPUT_ROOT:-$CKPT_ROOT/figure8_like/graft_eval}

FT_BS=${FT_BS:-2}
FT_LR=${FT_LR:-1e-3}
FT_MAX_STEP=${FT_MAX_STEP:-1000}
MODEL_SEED=${MODEL_SEED:-0}

GRAFT_LR=${GRAFT_LR:-1e7}
GRAFT_EPOCHS=${GRAFT_EPOCHS:-10}
MASK_BASE=${MASK_BASE:-zero}
SPARSITY_LEVEL=${SPARSITY_LEVEL:-0}

RUN_FINETUNE=${RUN_FINETUNE:-true}
RUN_GRAFT=${RUN_GRAFT:-true}
FORCE=${FORCE:-false}
RUN_ARGS=${RUN_ARGS:-}
GRAFT_ARGS=${GRAFT_ARGS:-}

mkdir -p "$FT_ROOT" "$MASK_ROOT" "$GRAFT_OUTPUT_ROOT"

read -r -a TASK_ARRAY <<< "$TASKS"
read -r -a SEED_ARRAY <<< "$SEEDS"

for task in "${TASK_ARRAY[@]}"; do
  for seed in "${SEED_ARRAY[@]}"; do
    data_dir="$DATA_ROOT/$task/$K-$seed"
    if [ ! -d "$data_dir" ]; then
      echo "Skipping missing data directory: $data_dir" >&2
      continue
    fi

    ft_dir="$FT_ROOT/$task/k$K/seed-$seed/$MODELNAME_SAFE"
    mask_dir="$MASK_ROOT/$task/k$K/seed-$seed/$MODELNAME_SAFE"
    mask_file="$mask_dir/mask_${MASK_BASE}.pt"
    graft_output_dir="$GRAFT_OUTPUT_ROOT/$task/k$K/seed-$seed/$MODELNAME_SAFE"

    if [ "$RUN_FINETUNE" = "true" ]; then
      if [ "$FORCE" != "true" ] && [ -f "$ft_dir/pytorch_model.bin" ]; then
        echo "Fine-tuned checkpoint exists, skipping: $ft_dir"
      else
        echo "Fine-tuning $task k=$K seed=$seed -> $ft_dir"
        TAG=figure8_like \
        TYPE="$TYPE" \
        TASK="$task" \
        K="$K" \
        BS="$FT_BS" \
        LR="$FT_LR" \
        SEED="$seed" \
        modelseed="$MODEL_SEED" \
        uselmhead=1 \
        useCLS=0 \
        max_step="$FT_MAX_STEP" \
        fixhead=True \
        fixembeddings=True \
        MODEL="$MODEL" \
        MODELNAME="$MODELNAME" \
        train_bias_only=False \
        OUTPUT_DIR="$ft_dir" \
        DATA_ROOT="$DATA_ROOT" \
        CKPT_ROOT="$CKPT_ROOT" \
        LOG_ROOT="$LOG_ROOT" \
        MODEL_CACHE_DIR="$MODEL_CACHE_DIR" \
        RESOLVE_HF_MODELS="$RESOLVE_HF_MODELS" \
        bash run_experiment.sh $RUN_ARGS
      fi
    fi

    if [ "$RUN_GRAFT" = "true" ]; then
      if [ ! -f "$ft_dir/pytorch_model.bin" ]; then
        echo "Missing fine-tuned checkpoint for grafting: $ft_dir" >&2
        exit 1
      fi

      if [ "$FORCE" != "true" ] && [ -f "$mask_file" ]; then
        echo "Graft mask exists, skipping: $mask_file"
      else
        echo "Training graft mask $task k=$K seed=$seed -> $mask_file"
        TYPE="$TYPE" \
        TASK="$task" \
        K="$K" \
        LR="$GRAFT_LR" \
        SEED="$seed" \
        MODEL="$ft_dir" \
        uselmhead=1 \
        useCLS=0 \
        num_train_epochs="$GRAFT_EPOCHS" \
        mask_path="$MASK_BASE" \
        sparsitylevel="$SPARSITY_LEVEL" \
        pretrained_model="$MODEL" \
        fixhead=True \
        fixembeddings=True \
        truncate_head=True \
        train_bias_only=False \
        no_train=False \
        checkpoint_location="$mask_file" \
        DATA_ROOT="$DATA_ROOT" \
        LOG_ROOT="$LOG_ROOT" \
        MODEL_CACHE_DIR="$MODEL_CACHE_DIR" \
        GRAFT_OUTPUT_DIR="$graft_output_dir" \
        RESOLVE_HF_MODELS="$RESOLVE_HF_MODELS" \
        bash run_graft_experiment.sh $GRAFT_ARGS
      fi
    fi
  done
done

echo "Mask root: $MASK_ROOT"
echo "To compute pairwise overlap:"
echo "uv run python tools/mask_overlap.py --mask-dir $MASK_ROOT --output-csv $MASK_ROOT/overlap.csv"
