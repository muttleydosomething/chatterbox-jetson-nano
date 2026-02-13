"""Jetson Orin Nano compatibility patches for Chatterbox TTS.

Fixes:
1. cuFFT: Orin rejects certain FFT sizes. Route STFT/ISTFT/RFFT to CPU.
2. Memory: Convert model to fp16 after loading to fit in 8GB unified memory.
"""
import torch
import gc

# === FFT Patches ===
_original_stft = torch.stft
_original_istft = torch.istft
_original_rfft = torch.fft.rfft
_original_irfft = torch.fft.irfft

def _patched_stft(input, n_fft, *args, **kwargs):
    if input.is_cuda:
        device = input.device
        window = kwargs.get("window", None)
        if window is not None and window.is_cuda:
            kwargs["window"] = window.cpu()
        result = _original_stft(input.cpu(), n_fft, *args, **kwargs)
        return result.to(device)
    return _original_stft(input, n_fft, *args, **kwargs)

def _patched_istft(input, n_fft, *args, **kwargs):
    if input.is_cuda:
        device = input.device
        window = kwargs.get("window", None)
        if window is not None and window.is_cuda:
            kwargs["window"] = window.cpu()
        result = _original_istft(input.cpu(), n_fft, *args, **kwargs)
        return result.to(device)
    return _original_istft(input, n_fft, *args, **kwargs)

def _patched_rfft(input, *args, **kwargs):
    if input.is_cuda:
        device = input.device
        result = _original_rfft(input.cpu(), *args, **kwargs)
        return result.to(device)
    return _original_rfft(input, *args, **kwargs)

def _patched_irfft(input, *args, **kwargs):
    if input.is_cuda:
        device = input.device
        result = _original_irfft(input.cpu(), *args, **kwargs)
        return result.to(device)
    return _original_irfft(input, *args, **kwargs)

torch.stft = _patched_stft
torch.istft = _patched_istft
torch.fft.rfft = _patched_rfft
torch.fft.irfft = _patched_irfft


# === FP16 Model Conversion ===
# Monkey-patch from_pretrained to convert model to fp16 after loading.
# Saves ~1.4GB on 8GB Jetson unified memory.

def _patch_chatterbox_fp16():
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
    except ImportError:
        return

    _original_from_pretrained = ChatterboxTurboTTS.from_pretrained

    @classmethod
    def _fp16_from_pretrained(cls, device):
        # Load normally (fp32)
        model = _original_from_pretrained.__func__(cls, device)

        # Convert all sub-models to fp16
        model.t3 = model.t3.half()
        model.s3gen = model.s3gen.half()
        model.ve = model.ve.half()
        if model.conds is not None:
            for field_name in vars(model.conds):
                val = getattr(model.conds, field_name)
                if isinstance(val, torch.Tensor) and val.is_floating_point():
                    setattr(model.conds, field_name, val.half())

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model

    ChatterboxTurboTTS.from_pretrained = _fp16_from_pretrained

    # Also wrap generate to use autocast for dtype handling
    _original_generate = ChatterboxTurboTTS.generate

    def _autocast_generate(self, *args, **kwargs):
        if self.device == "cuda" or str(self.device).startswith("cuda"):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                return _original_generate(self, *args, **kwargs)
        return _original_generate(self, *args, **kwargs)

    ChatterboxTurboTTS.generate = _autocast_generate

# Apply the patch when chatterbox is first imported
import importlib
import sys

class _ChatterboxPatchFinder:
    """Meta path finder that patches chatterbox after import."""
    _patched = False

    def find_module(self, name, path=None):
        if name == "chatterbox.tts_turbo" and not self._patched:
            return self
        return None

    def load_module(self, name):
        # Let normal import happen first
        sys.meta_path.remove(self)
        mod = importlib.import_module(name)
        _patch_chatterbox_fp16()
        self.__class__._patched = True
        return mod

sys.meta_path.insert(0, _ChatterboxPatchFinder())
