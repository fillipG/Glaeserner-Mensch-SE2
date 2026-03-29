FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python point to python3 (3.10 on Ubuntu 22.04)
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip for *this* python
RUN python -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .

# Install PyTorch CUDA + deps for the SAME python
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

COPY script.py .

ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

CMD ["python", "script.py"]
