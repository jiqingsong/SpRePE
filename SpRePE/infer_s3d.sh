#!/usr/bin/env bash
set -euo pipefail

export MODULUS_DISTRIBUTED_INITIALIZATION_METHOD="OPENMPI"
export WANDB_MODE=offline
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

PYTHON="path/to/python"
INFER="makani/inference.py"

# 固定参数
FIXED_CONFIG="vit_73ch"
FIXED_RUNNUM="runnum"
RUNNUM=(
  "vit_sprepe"
)

GPU_IDS=(0)
YAMLS=(
  "config/vit_s3d.yaml"
)

CKPTS=(
  'path/to/training_checkpoints'
)

BASE_PORT=14050
LOG_ROOT="logs_s3d_infer/$(date +%Y%m%d-%H%M%S)_$FIXED_RUNNUM"
mkdir -p "$LOG_ROOT"


for i in "${!YAMLS[@]}"; do
  YAML="${YAMLS[$i]}"
  GPU="${GPU_IDS[$(( i % ${#GPU_IDS[@]} ))]}"
  PORT=$((BASE_PORT + i))
  LOG="$LOG_ROOT/$(basename "$YAML" .yaml)_gpu${GPU}.log"
  FIXED_RUNNUM="$FIXED_RUNNUM"
  CKPT="${CKPTS[$i]}"
  echo "[LAUNCH] yaml=$YAML | --config=$FIXED_CONFIG | --run_num=$FIXED_RUNNUM | GPU=$GPU | MASTER_PORT=$PORT | CKPT=$CKPT"

  CUDA_VISIBLE_DEVICES="$GPU" MASTER_PORT="$PORT" \
  mpirun -np 1 --allow-run-as-root \
    -x CUDA_VISIBLE_DEVICES -x MASTER_PORT -x WANDB_MODE -x MODULUS_DISTRIBUTED_INITIALIZATION_METHOD \
    "$PYTHON" -u "$INFER" \
      --yaml_config="$YAML" \
      --config="$FIXED_CONFIG" \
      --checkpointing_level=3 \
      --batch_size=1 \
      --run_num="$FIXED_RUNNUM" \
      --checkpoint_path="$CKPT" \
      >"$LOG" 2>&1 &
done

wait
echo "All jobs finished. Logs -> $LOG_ROOT"
