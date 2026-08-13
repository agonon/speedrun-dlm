FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04@sha256:24c8e3581ea6330038b0d374920721983312627f8adbfcf390bdb4b399d280ed

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

# Fetch Ubuntu packages from Canonical's public APT snapshot dated
# 2026-08-12 12:00 UTC so the installed system packages are reproducible.
RUN sed -i -E \
      's#^URIs: http://(archive\.ubuntu\.com/ubuntu|security\.ubuntu\.com/ubuntu|ports\.ubuntu\.com/ubuntu-ports)/?$#URIs: https://snapshot.ubuntu.com/ubuntu/20260812T120000Z/#' \
      /etc/apt/sources.list.d/ubuntu.sources && \
    rm -f /etc/apt/sources.list.d/cuda.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt constraints.txt /speedrun-dlm/
WORKDIR /speedrun-dlm

ARG TORCH_VERSION=2.10.0+cu128
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG PIP_VERSION=26.2.1

RUN python3 -m pip install --ignore-installed --break-system-packages "pip==$PIP_VERSION" && \
    python3 -m pip install --break-system-packages \
      --index-url "$TORCH_INDEX_URL" \
      --constraint constraints.txt \
      "torch==$TORCH_VERSION" && \
    grep -v '^torch' requirements.txt > /tmp/requirements-no-torch.txt && \
    python3 -m pip install --break-system-packages \
      --constraint constraints.txt \
      --requirement /tmp/requirements-no-torch.txt

COPY . /speedrun-dlm

CMD ["bash"]
ENTRYPOINT []
