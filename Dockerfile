FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/data/hf-cache
ENV WANDB_DIR=/data/wandb

# Python 3.12 via deadsnakes (Ubuntu 22.04 ships 3.10 by default).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

WORKDIR /app

COPY pyproject.toml requirements.txt ./

# Use a torch version that has cu124 wheels (2.4.0–2.6.0 confirmed available).
RUN pip install --no-cache-dir 'torch>=2.4.0,<2.7.0' --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir bitsandbytes wandb

COPY . .
RUN pip install --no-cache-dir -e ".[training]"

RUN mkdir -p /data /data/hf-cache /data/wandb /data/wandb-cache /data/wandb-config /tmp/home \
    && chmod -R 777 /data /tmp/home \
    && chmod +x scripts/space_run_layers_6_7.sh

CMD ["./scripts/space_run_layers_6_7.sh"]
