#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_SOCKET_IFNAME=bond5
export NCCL_BLOCKING_WAIT=1
export NCCL_IB_DISABLE=1
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200
export OMP_NUM_THREADS=8

accelerate launch \
  --config_file accelerate_fsdp.yaml \
  --main_process_port 29500 \
  train.py
