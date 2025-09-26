#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export MASTER_PORT=12095
export WANDB_MODE=offline
export MODULUS_DISTRIBUTED_INITIALIZATION_METHOD="OPENMPI"
mpirun -np 1 --allow-run-as-root python -u makani/inference.py --yaml_config="config/vit_era5_test.yaml"  --batch_size=1 --config="vit_73ch" --checkpoint_path='path/to/training_checkpoints' --run_num="runnum"
