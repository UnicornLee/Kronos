# Kronos Web UI on Apple Silicon (linux/arm64).
# NOTE: a macOS container has no MPS access — PyTorch inside the container runs
# on CPU. The Kronos models are tiny (4.1M / 24.7M / 102.3M params) so CPU
# inference is fast enough; use the native (non-Docker) install if you want MPS.
FROM python:3.11-slim-bookworm

ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/models/hf \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    PORT=7070

WORKDIR /app

# curl is used by the HEALTHCHECK below
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt /app/requirements-docker.txt
# torch first, from the official CPU-only index: the plain PyPI linux/aarch64
# wheel of torch >= 2.7 drags in ~1.5 GB of NVIDIA CUDA runtime packages that
# are useless inside a macOS container (no GPU passthrough there).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.9.1+cpu" \
 && pip install --no-cache-dir -r /app/requirements-docker.txt

COPY . /app
# the container runs with WORKDIR /app/webui, so the entrypoint has to sit there
COPY entrypoint.py /app/webui/entrypoint.py

WORKDIR /app/webui
EXPOSE 7070

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://127.0.0.1:7070/api/available-models || exit 1

CMD ["python", "entrypoint.py"]
