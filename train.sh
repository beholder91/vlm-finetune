#!/usr/bin/env bash
# train.sh
set -e

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONFAULTHANDLER=1  # 增加错误跟踪能力

accelerate launch \
  --config_file deepspeed_zero3.yaml \
  --num_processes=4 \
  train.py
