#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3,4,5
export MODULUS_DISTRIBUTED_INITIALIZATION_METHOD="OPENMPI"
export WANDB_MODE=offline
export MASTER_PORT=12395
mpirun -np 4 --allow-run-as-root  python -u makani/train.py --yaml_config="config/vit_era5.yaml" --config="vit_73ch" --checkpointing_level=3 --batch_size=4 --amp_mode=bf16 --run_num="runnum" #
