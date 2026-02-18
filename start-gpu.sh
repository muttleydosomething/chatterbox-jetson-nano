#!/bin/bash
# Start Chatterbox TTS with GPU acceleration on Jetson Orin Nano
#
# Fixes applied:
# 1. Mount host Jetson-native cuBLAS/cuDNN/cuFFT (container ships SBSA builds)
# 2. STFT/ISTFT monkey-patch for non-power-of-2 FFT sizes via sitecustomize.py
# 3. Mount config.yaml (writable), server.py, engine.py, and voices/ for runtime customization
# 4. PYTORCH_NO_CUDA_MEMORY_CACHING=1 — disables caching allocator to prevent memory fragmentation
#    on Jetson unified memory. Without this, consecutive synthesis requests fragment physical RAM
#    causing CUDA ENOMEM after 3-4 utterances. Small speed cost, major stability gain.
# 5. --restart no — chatterbox-watchdog.service handles crash recovery. On CUDA OOM the process
#    exits with code 1; the watchdog drops page caches from the host (SYS_ADMIN available there)
#    then restarts the container with clean contiguous memory. Docker's built-in restart would
#    skip the cache drop, causing the reload to OOM on fragmented RAM and loop indefinitely.

CUBLAS_HOST=/usr/local/cuda/targets/aarch64-linux/lib
CUBLAS_CTR=/usr/local/cuda/targets/sbsa-linux/lib
CUDNN_HOST=/lib/aarch64-linux-gnu
CUDNN_CTR=/usr/lib/aarch64-linux-gnu

docker run -d --name chatterbox-tts \
  --runtime nvidia --network host \
  --restart no \
  -e HF_TOKEN=${HF_TOKEN:?Set HF_TOKEN environment variable or edit this script} \
  -e PYTHONPATH=/app/patches \
  -e PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
  -v chatterbox-models:/app/hf_cache \
  -v chatterbox-outputs:/app/outputs \
  -v ~/chatterbox-jetson/config.yaml:/app/config.yaml \
  -v ~/chatterbox-jetson/server.py:/app/server.py \
  -v ~/chatterbox-jetson/engine.py:/app/engine.py \
  -v ~/chatterbox-jetson/voices:/app/voices \
  -v /mnt/nas/music/_tts:/app/outputs/permanent \
  -v ~/chatterbox-jetson/patches:/app/patches:ro \
  -v ~/chatterbox-jetson/ui/index.html:/app/ui/index.html:ro \
  -v ~/chatterbox-jetson/ui/favicon.png:/app/ui/favicon.png:ro \
  -v ${CUBLAS_HOST}/libcublas.so.12.6.1.4:${CUBLAS_CTR}/libcublas.so.12.6.4.1:ro \
  -v ${CUBLAS_HOST}/libcublasLt.so.12.6.1.4:${CUBLAS_CTR}/libcublasLt.so.12.6.4.1:ro \
  -v ${CUBLAS_HOST}/libcufft.so.11.2.6.59:${CUBLAS_CTR}/libcufft.so.11.3.0.4:ro \
  -v ${CUBLAS_HOST}/libcufftw.so.11.2.6.59:${CUBLAS_CTR}/libcufftw.so.11.3.0.4:ro \
  -v ${CUDNN_HOST}/libcudnn.so.9.3.0:${CUDNN_CTR}/libcudnn.so.9.5.1:ro \
  -v ${CUDNN_HOST}/libcudnn_adv.so.9.3.0:${CUDNN_CTR}/libcudnn_adv.so.9.5.1:ro \
  -v ${CUDNN_HOST}/libcudnn_cnn.so.9.3.0:${CUDNN_CTR}/libcudnn_cnn.so.9.5.1:ro \
  -v ${CUDNN_HOST}/libcudnn_engines_precompiled.so.9.3.0:${CUDNN_CTR}/libcudnn_engines_precompiled.so.9.5.1:ro \
  -v ${CUDNN_HOST}/libcudnn_engines_runtime_compiled.so.9.3.0:${CUDNN_CTR}/libcudnn_engines_runtime_compiled.so.9.5.1:ro \
  -v ${CUDNN_HOST}/libcudnn_graph.so.9.3.0:${CUDNN_CTR}/libcudnn_graph.so.9.5.1:ro \
  -v ${CUDNN_HOST}/libcudnn_heuristic.so.9.3.0:${CUDNN_CTR}/libcudnn_heuristic.so.9.5.1:ro \
  -v ${CUDNN_HOST}/libcudnn_ops.so.9.3.0:${CUDNN_CTR}/libcudnn_ops.so.9.5.1:ro \
  chatterbox-tts-jetson
