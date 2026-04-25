FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/data/hf-cache
ENV WANDB_DIR=/data/wandb
ENV WANDB_CACHE_DIR=/data/wandb-cache
ENV WANDB_CONFIG_DIR=/data/wandb-config
ENV HOME=/tmp/home

# The CUDA base image ships no Python. Install 3.12 via deadsnakes,
# then bootstrap pip with get-pip.py against that exact interpreter.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | /usr/bin/python3.12

# Single Python everywhere: python, python3, pip all resolve to 3.12.
RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

# Sanity check — surfaces version mismatches in build logs immediately.
RUN python3 --version && python3 -m pip --version && python --version

WORKDIR /app

COPY pyproject.toml requirements.txt ./
RUN python3 -m pip install --no-cache-dir 'torch>=2.4.0,<2.7.0' --index-url https://download.pytorch.org/whl/cu124
RUN python3 -m pip install --no-cache-dir bitsandbytes wandb

COPY . .
RUN python3 -m pip install --no-cache-dir -e ".[training]"

RUN mkdir -p /data /data/hf-cache /data/wandb /data/wandb-cache /data/wandb-config /tmp/home \
    && chmod -R 777 /data /tmp/home

# /app was COPYed as root; non-root runtime user needs write access for
# evaluation/_results, training/_runs, and any other in-tree artifact paths.
RUN chmod -R 777 /app

RUN chmod +x scripts/space_run_layers_6_7.sh

CMD ["./scripts/space_run_layers_6_7.sh"]
