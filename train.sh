# NCCL 通信调试
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_STACK=1

# DeepSpeed 日志
export DEEPSPEED_LOG_LEVEL=debug

# Accelerate 日志
export ACCELERATE_LOG_LEVEL=debug

# Transformers 日志
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_LOAD_WARNINGS=1

accelerate launch --config_file deepspeed_zero3.yaml train.py
