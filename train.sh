#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export NCCL_DEBUG=INFO # 设置NCCL调试信息
export TORCH_NCCL_TRACE_BUFFER_SIZE=1024000
export NCCL_P2P_DISABLE=1
# export NCCL_SOCKET_IFNAME=bond5
# export NCCL_BLOCKING_WAIT=1
# export NCCL_IB_DISABLE=1
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200
export OMP_NUM_THREADS=1


accelerate launch \
  --config_file accelerate_fsdp.yaml \
  --main_process_port 29500 \
  train.py
