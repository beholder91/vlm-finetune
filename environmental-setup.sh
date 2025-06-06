pip install --upgrade pip

mamba install -c nvidia cuda-nvcc=12.8 cuda-libraries-dev=12.8 ninja -y

pip install --no-cache-dir --force-reinstall accelerate==1.6.0

# pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
#         --extra-index-url https://download.pytorch.org/whl/cu124

# DS_BUILD_OPS=0 pip install deepspeed==0.16.7 --no-deps

# pip install einops hjson msgpack ninja nvidia-ml-py py-cpuinfo "pydantic>=2" typing-inspection annotated-types

pip install transformers datasets torchvision pymupdf wandb