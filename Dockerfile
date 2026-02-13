# ============================================================
# Chatterbox TTS Server for NVIDIA Jetson
# JetPack 6.x (L4T r36.x), aarch64, CUDA 12.6
#
# Self-contained — no host bind-mounts needed.
# CUDA runtime libs included via nvidia/cuda base image.
# PyTorch from Jetson AI Lab (CUDA ARM64 wheels).
#
# Build:  docker build -t chatterbox-tts-jetson .
# Run:    docker run -d --name chatterbox-tts \
#           --runtime nvidia --network host \
#           --restart unless-stopped \
#           -e HF_TOKEN=hf_your_token_here \
#           -v chatterbox-models:/app/hf_cache \
#           -v chatterbox-outputs:/app/outputs \
#           chatterbox-tts-jetson
#
# First run downloads ~1.9GB model from HuggingFace (public, but token required by library).
# Web UI: http://<jetson-ip>:8004/
# API docs: http://<jetson-ip>:8004/docs
# ============================================================

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/app/hf_cache
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/libcudss/12:${LD_LIBRARY_PATH}

# ---- System dependencies + Python 3.10 (Ubuntu 22.04 default) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    cmake \
    ninja-build \
    libsndfile1 \
    ffmpeg \
    libopenblas0 \
    git \
    curl \
    cuda-cupti-12-6 \
    libcudss0-cuda-12 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# ---- PyTorch for Jetson (CUDA 12.6 ARM64 wheels) ----
# Downloaded directly from Jetson AI Lab — pip index resolution
# fails because these use legacy linux_aarch64 platform tags.
RUN curl -fSL -o /tmp/torch-2.9.1-cp310-cp310-linux_aarch64.whl \
      https://pypi.jetson-ai-lab.io/jp6/cu126/+f/02f/de421eabbf626/torch-2.9.1-cp310-cp310-linux_aarch64.whl && \
    curl -fSL -o /tmp/torchaudio-2.9.1-cp310-cp310-linux_aarch64.whl \
      https://pypi.jetson-ai-lab.io/jp6/cu126/+f/d12/bede7113e6b00/torchaudio-2.9.1-cp310-cp310-linux_aarch64.whl && \
    curl -fSL -o /tmp/torchvision-0.24.1-cp310-cp310-linux_aarch64.whl \
      https://pypi.jetson-ai-lab.io/jp6/cu126/+f/d5b/caaf709f11750/torchvision-0.24.1-cp310-cp310-linux_aarch64.whl && \
    pip install --no-cache-dir \
      /tmp/torch-2.9.1-cp310-cp310-linux_aarch64.whl \
      /tmp/torchaudio-2.9.1-cp310-cp310-linux_aarch64.whl \
      /tmp/torchvision-0.24.1-cp310-cp310-linux_aarch64.whl && \
    rm -f /tmp/*.whl

# ---- Chatterbox TTS (--no-deps to skip pinned torch==2.6.0) ----
RUN pip install --no-cache-dir --no-deps chatterbox-tts==0.1.6

# ---- Remaining Python dependencies ----
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    "numpy>=1.24.0,<2.0.0" \
    soundfile \
    librosa \
    safetensors \
    descript-audio-codec \
    huggingface-hub \
    tokenizers \
    transformers \
    PyYAML \
    python-multipart \
    requests \
    Jinja2 \
    watchdog \
    aiofiles \
    unidecode \
    inflect \
    tqdm \
    hf_transfer \
    pydub \
    audiotsm \
    praat-parselmouth \
    einops \
    encodec

# ---- Chatterbox transitive deps (missed by --no-deps) ----
RUN pip install --no-cache-dir \
    resemble-perth \
    s3tokenizer \
    omegaconf \
    conformer \
    diffusers \
    pyloudnorm

# ---- Application ----
WORKDIR /app
COPY . .
RUN mkdir -p model_cache outputs logs hf_cache voices

EXPOSE 8004

CMD ["python3", "server.py"]
