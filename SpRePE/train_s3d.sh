#!/usr/bin/env bash
set -euo pipefail

# ===== 固定环境 =====
export MODULUS_DISTRIBUTED_INITIALIZATION_METHOD="OPENMPI"
export WANDB_MODE=offline
# optional: 控制CPU并发，避免过度竞争
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

PYTHON="path/to/python"
TRAIN="makani/train.py"

# 固定参数
FIXED_CONFIG=""
FIXED_RUNNUM=""

GPU_IDS=(0)

YAMLS=(
  'config/vit_s3d.yaml'
)

# 端口与日志
BASE_PORT=14899
LOG_ROOT="logs_s3d/$(date +%Y%m%d-%H%M%S)_$FIXED_RUNNUM"
mkdir -p "$LOG_ROOT"

# ===== 并行启动 =====
for i in "${!YAMLS[@]}"; do
  YAML="${YAMLS[$i]}"
  GPU="${GPU_IDS[$(( i % ${#GPU_IDS[@]} ))]}"
  PORT=$((BASE_PORT + i))
  LOG="$LOG_ROOT/$(basename "$YAML" .yaml)_gpu${GPU}.log"

  echo "[LAUNCH] yaml=$YAML | --config=$FIXED_CONFIG | --run_num=$FIXED_RUNNUM | GPU=$GPU | MASTER_PORT=$PORT"

  # 每个进程绑定到一张独立 GPU，并使用独立端口
  CUDA_VISIBLE_DEVICES="$GPU" MASTER_PORT="$PORT" \
  mpirun -np 1 --allow-run-as-root \
    -x CUDA_VISIBLE_DEVICES -x MASTER_PORT -x WANDB_MODE -x MODULUS_DISTRIBUTED_INITIALIZATION_METHOD \
    "$PYTHON" -u "$TRAIN" \
      --yaml_config="$YAML" \
      --config="$FIXED_CONFIG" \
      --checkpointing_level=3 \
      --batch_size=16 \
      --run_num="$FIXED_RUNNUM" \
      >"$LOG" 2>&1 &
done

wait
echo "All jobs finished. Logs -> $LOG_ROOT"
