# Chatterbox TTS on NVIDIA Jetson

**The first GPU-accelerated Chatterbox TTS container for Jetson Orin Nano.**

Text-to-speech with voice cloning, running entirely on a $249 edge AI device. ~4 seconds per utterance on GPU, with an OpenAI-compatible API. Pairs with [Whisper STT](https://github.com/muttleydosomething/whisper-stt-jetson) for a complete voice I/O stack on Jetson.

| | |
|---|---|
| **Hardware** | NVIDIA Jetson Orin Nano (8GB) |
| **JetPack** | 6.x (L4T r36.x) |
| **CUDA** | 12.6 |
| **PyTorch** | 2.9.1 (Jetson AI Lab wheels) |
| **Model** | Chatterbox Turbo (~1.9GB, fp16) |
| **Image size** | 5.92 GB |
| **Inference** | ~4s warm / ~28s cold per utterance |
| **License** | Apache 2.0 |

---

## Why This Exists

Running Chatterbox TTS on Jetson should be straightforward — it's a CUDA device with PyTorch support. In practice, it took solving **11 distinct technical problems**, most of which have no documentation anywhere.

The most significant discovery: **NVIDIA's own `nvidia/cuda` Docker images ship the wrong CUDA libraries for Jetson.** They include SBSA (Server Base System Architecture) builds of cuBLAS, cuDNN, and cuFFT — compiled for server-class ARM64 GPUs like Grace-Hopper, not for Jetson's Orin iGPU. These libraries load without any error but **silently fail** when called, producing cryptic errors like `CUBLAS_STATUS_ALLOC_FAILED` even with gigabytes of free memory.

This project provides a complete, working solution: a Docker image, runtime patches, and a startup script that handles all of it.

---

## Quick Start

### Prerequisites

- NVIDIA Jetson Orin Nano (8GB recommended) running JetPack 6.x
- Docker with `nvidia-container-runtime`
- A [HuggingFace token](https://huggingface.co/settings/tokens) (free — the model is public but the library requires authentication)

### 1. Get the image

**Option A: Pull pre-built (recommended)**

```bash
docker pull ghcr.io/muttleydosomething/chatterbox-tts-jetson:latest
docker tag ghcr.io/muttleydosomething/chatterbox-tts-jetson:latest chatterbox-tts-jetson
```

**Option B: Build from source (~35 min)**

```bash
git clone https://github.com/muttleydosomething/chatterbox-jetson-nano.git
cd chatterbox-jetson
docker build -t chatterbox-tts-jetson .
```

Building from source takes ~35 minutes (most of that is `praat-parselmouth` compiling C++ from source).

### 2. Start with GPU acceleration

```bash
./start-gpu.sh
```

On first run, edit `start-gpu.sh` and replace `YOUR_HF_TOKEN_HERE` with your HuggingFace token. The script handles all the SBSA library replacements automatically.

First startup downloads ~1.9GB of model weights. Subsequent starts use the cached weights.

### 3. Use it

- **Web UI:** `http://<jetson-ip>:8004/`
- **API docs:** `http://<jetson-ip>:8004/docs`
- **OpenAI-compatible endpoint:** `POST http://<jetson-ip>:8004/v1/audio/speech`

```bash
# Generate speech
curl -X POST http://localhost:8004/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "chatterbox-turbo", "input": "Hello from the Jetson.", "voice": "Alexander.wav"}' \
  --output speech.wav
```

---

## The SBSA Problem (and Why Your GPU "Doesn't Work")

If you've tried running CUDA workloads inside Docker on Jetson and hit any of these errors, you've been bitten by SBSA:

| Error | Cause |
|-------|-------|
| `CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)` | SBSA cuBLAS |
| `GET was unable to find an engine to execute this computation` | SBSA cuDNN |
| `cuFFT error: CUFFT_INVALID_SIZE` / `CUFFT_INTERNAL_ERROR` | SBSA cuFFT |

**What's happening:** The `nvidia/cuda` Docker images on Docker Hub are built for two ARM64 targets — `sbsa-linux` (server) and `aarch64-linux` (embedded/Jetson). The images default to SBSA libraries, which are binary-incompatible with Jetson's GPU.

**The fix:** Mount the host's Jetson-native libraries into the container, replacing the SBSA versions:

```bash
# cuBLAS: host aarch64-linux → replaces container sbsa-linux
-v /usr/local/cuda/targets/aarch64-linux/lib/libcublas.so.12.6.1.4:/usr/local/cuda/targets/sbsa-linux/lib/libcublas.so.12.6.4.1:ro

# cuDNN: host native → replaces container SBSA build
-v /lib/aarch64-linux-gnu/libcudnn.so.9.3.0:/usr/lib/aarch64-linux-gnu/libcudnn.so.9.5.1:ro

# cuFFT: host aarch64-linux → replaces container sbsa-linux
-v /usr/local/cuda/targets/aarch64-linux/lib/libcufft.so.11.2.6.59:/usr/local/cuda/targets/sbsa-linux/lib/libcufft.so.11.3.0.4:ro
```

The `start-gpu.sh` script does all of this automatically, mounting cuBLAS, cuBLASLt, cuFFT, cuFFTW, and all 8 cuDNN sub-libraries.

> **Note:** The exact `.so` version numbers depend on your L4T / JetPack version. The versions in `start-gpu.sh` are for JetPack 6.2.2 (L4T 36.5.0). If you're on a different version, check your host libraries at `/usr/local/cuda/targets/aarch64-linux/lib/` and `/lib/aarch64-linux-gnu/` and update the paths accordingly.

---

## Additional Jetson Workarounds

### cuFFT Non-Power-of-2 Limitation

Even with the correct native cuFFT library, Jetson Orin rejects non-power-of-2 FFT sizes (e.g., `n_fft=1920` used by audio processing). The `patches/sitecustomize.py` file transparently routes all FFT operations (`torch.stft`, `torch.istft`, `torch.fft.rfft`, `torch.fft.irfft`) to CPU while keeping everything else on GPU.

### FP16 Memory Optimization

The Turbo model in fp32 uses ~4.3GB — too much for the 8GB Orin's unified memory (shared between CPU and GPU) once you account for inference buffers. The `sitecustomize.py` patch automatically converts the model to fp16 after loading (~2.4GB), using `torch.amp.autocast` for seamless mixed-precision inference. No quality loss on speech output.

Both patches are loaded automatically via the `PYTHONPATH=/app/patches` environment variable in `start-gpu.sh`.

---

## CPU Fallback

If you don't need GPU acceleration (or want to run alongside other GPU workloads), edit `config.yaml`:

```yaml
tts_engine:
  device: cpu   # Change from 'cuda' to 'cpu'
```

Then use the basic start command (no library mounts needed):

```bash
docker run -d --name chatterbox-tts \
  --runtime nvidia --network host \
  --restart unless-stopped \
  -e HF_TOKEN=YOUR_HF_TOKEN_HERE \
  -v chatterbox-models:/app/hf_cache \
  -v chatterbox-outputs:/app/outputs \
  chatterbox-tts-jetson
```

CPU mode runs at ~42-69 seconds per utterance — functional but about 17x slower than GPU.

---

## API Reference

### OpenAI-Compatible: `POST /v1/audio/speech`

```json
{
  "model": "chatterbox-turbo",
  "input": "Hello from the Jetson. [laugh] That's pretty cool.",
  "voice": "Alexander.wav",
  "response_format": "wav"
}
```

All three fields (`model`, `input`, `voice`) are required. Voice names need the `.wav` extension.

### Custom Endpoint: `POST /tts`

```json
{
  "text": "Hello from the Jetson.",
  "predefined_voice_id": "Alexander.wav",
  "output_format": "wav"
}
```

Only `text` is required.

### Other Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /get_predefined_voices` | List available voices |
| `GET /api/model-info` | Model details |
| `GET /docs` | Interactive API documentation |
| `GET /` | Web UI |

### Paralinguistic Tags

The Turbo model supports expressive tags in the input text:

`[laugh]` `[chuckle]` `[sigh]` `[gasp]` `[cough]` `[clear throat]` `[sniff]` `[groan]` `[shush]`

Example: `"And then I realized [laugh] it was the wrong library architecture the whole time."`

---

## Project Structure

```
chatterbox-jetson/
├── Dockerfile              # Complete build recipe
├── start-gpu.sh            # GPU startup with SBSA library mounts
├── config.yaml             # Server configuration (writable at runtime)
├── patches/
│   └── sitecustomize.py    # FFT CPU routing + fp16 conversion + autocast
├── server.py               # Chatterbox TTS Server (FastAPI)
├── engine.py               # TTS engine wrapper
├── config.py               # Server config loader
├── models.py               # Model definitions
├── utils.py                # Utilities
├── static/                 # Web UI static assets
├── ui/                     # Web UI templates
├── reference_audio/        # Sample reference audio for voice cloning
└── voices/                 # Predefined voice files (28 built-in voices)
```

### Runtime File Mounts

The `start-gpu.sh` script mounts three files/directories as writable, so you can customize them without rebuilding the image:

- **`config.yaml`** — Server settings. Editable via the web UI at `http://<jetson-ip>:8004/` (settings page).
- **`server.py`** — The FastAPI server. Mount a modified copy to change defaults (e.g., default voice) without rebuilding.
- **`voices/`** — Voice reference WAV files. Drop new `.wav` files here and they appear in the web UI and API immediately.

---

## Configuration

`config.yaml`:

```yaml
server:
  host: 0.0.0.0
  port: 8004
model:
  repo_id: chatterbox-turbo
tts_engine:
  device: cuda              # 'cuda' for GPU, 'cpu' for CPU fallback
generation_defaults:
  temperature: 0.8
  exaggeration: 0.5         # Expressiveness (0.0 = neutral, 1.0 = dramatic)
  cfg_weight: 0.5
```

---

## Compatibility

Tested on:
- **Hardware:** NVIDIA Jetson Orin Nano 8GB Super
- **JetPack:** 6.2.2 (L4T 36.5.0)
- **CUDA:** 12.6, Driver 540.5.0
- **Kernel:** 5.15.185-tegra

Should work on other Jetson Orin variants (Orin NX, AGX Orin) with JetPack 6.x. The library version numbers in `start-gpu.sh` may need adjusting for different JetPack releases — check your host's `/usr/local/cuda/targets/aarch64-linux/lib/` for the correct versions.

**Not compatible with:** Jetson Nano (original), Jetson TX2, Jetson Xavier — these use older JetPack versions with different CUDA toolkits.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUBLAS_STATUS_ALLOC_FAILED` | SBSA cuBLAS loaded. Use `start-gpu.sh` instead of plain `docker run` |
| `GET was unable to find an engine` | SBSA cuDNN loaded. Check cuDNN bind mounts in `start-gpu.sh` |
| `CUFFT_INTERNAL_ERROR` | SBSA cuFFT and/or non-power-of-2 sizes. Ensure both cuFFT mounts and `patches/sitecustomize.py` are active |
| OOM during inference | Model loaded in fp32. Ensure `PYTHONPATH=/app/patches` is set so fp16 conversion activates |
| Model download hangs | Check your `HF_TOKEN` is valid. The model is public but the library requires authentication |
| `No CUDA runtime found` | Missing `--runtime nvidia` in docker run |
| Slow inference (~60s+) | Running in CPU mode. Check `config.yaml` has `device: cuda` and use `start-gpu.sh` |

---

## Performance

| Mode | Warm Inference | Cold Inference | Model Size |
|------|---------------|----------------|------------|
| GPU fp16 | ~4 seconds | ~28 seconds | ~2.4 GB |
| GPU fp32 | OOM | OOM | ~4.3 GB |
| CPU | ~42-69 seconds | ~42-69 seconds | ~4.3 GB |

Cold inference includes voice embedding computation on the first request. Subsequent requests with the same voice use cached embeddings.

---

## Problems We Solved

Building this required solving 11 technical problems with no existing documentation:

1. **PyTorch ARM64 CUDA wheels** — pip can't resolve Jetson AI Lab wheels; must download directly
2. **Wrong base image** — `python:3.10-slim` has no CUDA; Jetson runtime doesn't mount toolkit libs
3. **Bloated community images** — `dustynv/l4t-pytorch` is 6.4GB+; NVIDIA's CUDA runtime is 3.18GB
4. **SBSA cuBLAS** — silently fails on Jetson, reports fake "alloc failed" errors
5. **SBSA cuDNN** — silently fails on Jetson, reports "no engine found" errors
6. **SBSA cuFFT** — silently fails on Jetson, plus Orin rejects non-power-of-2 sizes even with native lib
7. **Broken upstream package** — `chatterbox-v2` git fork has broken metadata; use official PyPI
8. **Missing CUDA extras** — CUPTI and cuDSS not in CUDA runtime image, cuDSS installs to non-standard path
9. **C++ compilation** — `praat-parselmouth` needs cmake + ninja + python3.10-dev (~33 min build)
10. **Transitive dependencies** — 6 packages missed by `--no-deps` install, found iteratively
11. **FP16 memory optimization** — fp32 model OOMs during attention; auto-convert to fp16 with autocast

---

## Running Alongside Whisper STT

This project is designed to run alongside [Whisper STT](https://github.com/muttleydosomething/whisper-stt-jetson) on the same Jetson, creating a complete voice I/O stack:

| Port | Service | Endpoint | Function |
|------|---------|----------|----------|
| 8004 | Chatterbox TTS | `POST /v1/audio/speech` | Text to speech |
| 8005 | Whisper STT | `POST /v1/audio/transcriptions` | Speech to text |

### Startup Order Matters

**Chatterbox must load first.** It uses PyTorch which needs a large contiguous GPU allocation (~5.5 GiB in fp16). If Whisper grabs GPU memory first, Chatterbox's `cudaMalloc` will fail with `NVML_SUCCESS == r INTERNAL ASSERT FAILED`.

The [Whisper STT repo](https://github.com/muttleydosomething/whisper-stt-jetson) includes a `borg-ai-services.sh` boot orchestration script and a systemd unit that handles this automatically — starting Chatterbox first, waiting for model load, then starting Whisper.

---

## See Also

- **[Whisper STT on Jetson](https://github.com/muttleydosomething/whisper-stt-jetson)** — GPU-accelerated speech-to-text using whisper.cpp. Runs alongside Chatterbox on the same Orin Nano for a complete voice I/O stack (TTS on port 8004, STT on port 8005). Includes boot orchestration scripts for reliable startup.

---

## Acknowledgments

- [Resemble AI](https://github.com/resemble-ai/chatterbox) for the Chatterbox TTS model
- [NVIDIA Jetson AI Lab](https://www.jetson-ai-lab.com/) for the PyTorch ARM64 CUDA wheels
- [Chatterbox TTS Server](https://github.com/devnen/chatterbox-tts-server) for the FastAPI server application
- Built with [Claude Code](https://claude.ai/code) (Claude Opus 4.6) — the SBSA discovery, fp16 patches, and FFT workarounds were developed collaboratively between human and AI over an extended debugging session

---

## Support This Project

If this saved you hours of debugging SBSA library issues on Jetson, consider buying me a coffee. The money helps fund continued development and pays for the AI tools used to build this.

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-support-yellow?style=for-the-badge&logo=buy-me-a-coffee)](https://buymeacoffee.com/muttleydosomething)

---

## License

Copyright 2026 Simon (muttleydosomething)

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
