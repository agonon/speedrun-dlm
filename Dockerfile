FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /speedrun-dlm/requirements.txt
WORKDIR /speedrun-dlm

ARG TORCH_VERSION=2.10.0+cu128
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

RUN python3 -m pip install --upgrade pip --break-system-packages && \
    python3 -m pip install --break-system-packages \
      --index-url "$TORCH_INDEX_URL" \
      "torch==$TORCH_VERSION" && \
    grep -v '^torch' requirements.txt > /tmp/requirements-no-torch.txt && \
    python3 -m pip install --break-system-packages -r /tmp/requirements-no-torch.txt

COPY . /speedrun-dlm

CMD ["bash"]
ENTRYPOINT []
