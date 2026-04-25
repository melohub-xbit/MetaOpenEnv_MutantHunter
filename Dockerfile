FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/data/hf-cache
ENV WANDB_DIR=/data/wandb
ENV WANDB_CACHE_DIR=/data/wandb-cache
ENV WANDB_CONFIG_DIR=/data/wandb-config
ENV HOME=/tmp/home

# Install only what's needed; use whatever python3 ships with the base
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Verify python3 exists; print version
RUN python3 --version && python3 -m pip --version

# Make `python` resolve to whatever `python3` is
RUN ln -sf $(which python3) /usr/bin/python

WORKDIR /app

COPY pyproject.toml requirements.txt ./
RUN python3 -m pip install --no-cache-dir 'torch>=2.4.0,<2.7.0' --index-url https://download.pytorch.org/whl/cu124
RUN python3 -m pip install --no-cache-dir bitsandbytes wandb

COPY . .
RUN python3 -m pip install --no-cache-dir -e ".[training]"

RUN mkdir -p /data /data/hf-cache /data/wandb /data/wandb-cache /data/wandb-config /tmp/home \
    && chmod -R 777 /data /tmp/home

RUN chmod +x scripts/space_run_layers_6_7.sh

CMD ["./scripts/space_run_layers_6_7.sh"]
