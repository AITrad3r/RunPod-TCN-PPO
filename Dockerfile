# GPU-enabled Serverless image for RunPod
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    build-essential git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Symlink python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Workdir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copy project
COPY . /app

# Default environment
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    TORCH_CUDA_ARCH_LIST="8.0+PTX;8.6;8.9;9.0" \
    PYTHONPATH=/app

# Health/diag
RUN python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
print('Torch version:', torch.__version__)
print('Device count:', torch.cuda.device_count())
PY

# RunPod Serverless requires a handler entrypoint
CMD ["python", "-u", "handler.py"]
