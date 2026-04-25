FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/data/hf-cache
ENV WANDB_DIR=/data/wandb

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir bitsandbytes wandb

COPY . .
RUN pip install --no-cache-dir -e ".[training]"

RUN mkdir -p /data && chmod +x scripts/space_run_layers_6_7.sh

CMD ["./scripts/space_run_layers_6_7.sh"]
