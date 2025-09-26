#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export MODULUS_DISTRIBUTED_INITIALIZATION_METHOD="OPENMPI"
export WANDB_MODE=offline
export MASTER_PORT=12397
mpirun -np 1 --allow-run-as-root  python -u makani/train.py --yaml_config="config/vit_mnist.yaml" --config="vit_73ch" --checkpointing_level=3 --batch_size=32 --run_num="runnum" #
