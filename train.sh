# # 禁用IB通信
# export NCCL_IB_DISABLE=1
# # 使用bond5网卡
# export NCCL_SOCKET_IFNAME=bond5

# # NCCL 通信调试
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_STACK=1

# # DeepSpeed 日志
# export DEEPSPEED_LOG_LEVEL=debug

# # Accelerate 日志
# export ACCELERATE_LOG_LEVEL=debug

# # Transformers 日志
# export TRANSFORMERS_VERBOSITY=error
# export HF_HUB_DISABLE_LOAD_WARNINGS=1

# accelerate launch --config_file deepspeed_zero3.yaml --num_processes=4 train.py

# 调试信息
export NCCL_DEBUG=INFO

# —— 重新启用高性能互联通道 —— 
# 允许使用 InfiniBand 插件，启用 P2P/NVLink，避免降级到 Socket
export NCCL_IB_DISABLE=0                     # 允许 IB 通信，默认值 0 :contentReference[oaicite:5]{index=5}
unset NCCL_SOCKET_IFNAME                     # 取消强制指定网卡，让 NCCL 自动选择最优接口 :contentReference[oaicite:6]{index=6}

# —— 启用 GPU Direct RDMA —— 
# 通过 C2C 或 NVLink 网卡直连，减少 CPU 拷贝开销
export NCCL_NET_GDR_C2C=1                    # 开启 C2C 直连 RDMA，默认值 0 :contentReference[oaicite:7]{index=7}
export NCCL_NET_GDR_READ=1                   # 开启 RDMA 读，默认 NVLink 平台为 1 :contentReference[oaicite:8]{index=8}

# —— 确保 P2P 与共享内存 —— 
export NCCL_P2P_DISABLE=0                    # 启用 GPU 间 P2P，默认值 0 :contentReference[oaicite:9]{index=9}
export NCCL_SHM_DISABLE=0                    # 启用共享内存，默认值 0 :contentReference[oaicite:10]{index=10}

# —— 可选：指定通信算法 —— 
export NCCL_ALGO=Ring                         # 使用环形算法（Ring），均衡带宽 :contentReference[oaicite:11]{index=11}

# 加速与 DeepSpeed 日志级别
export DEEPSPEED_LOG_LEVEL=debug
export ACCELERATE_LOG_LEVEL=debug
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_LOAD_WARNINGS=1

# 启动命令：4 卡单机训练
accelerate launch \
  --config_file deepspeed_zero3.yaml \
  --num_processes=4 \
  train.py
