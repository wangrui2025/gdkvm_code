#!/bin/zsh

export CUDA_VISIBLE_DEVICES=0,1,2,3

export MASTER_ADDR=locahost
export MASTER_PORT=10123
export HYDRA_FULL_ERROR=1

NUM_GPUS=4

export OMP_NUM_THREADS=$((12 * NUM_GPUS))

export PYTHONPATH=/data:$PYTHONPATH

# launch distributed training
torchrun --nproc-per-node=$NUM_GPUS --master-port=$MASTER_PORT train.py