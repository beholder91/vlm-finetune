ACCELERATE_LOG_LEVEL=DEBUG \
TRANSFORMERS_VERBOSITY=debug \
DEEPSPEED_LOGS=debug \
NCCL_DEBUG=INFO \
DS_DEBUG=1 \
accelerate launch --config_file deepspeed_zero3.yaml train.py