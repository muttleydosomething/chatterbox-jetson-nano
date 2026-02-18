"""Jetson Orin Nano compatibility patches for Chatterbox TTS.

Fixes:
1. cuFFT: Orin rejects certain FFT sizes. Route STFT/ISTFT/RFFT to CPU.
2. Memory: Convert model to fp16 after loading to fit in 8GB unified memory.
3. SDPA: Multilingual model uses output_attentions=True, incompatible with SDPA.
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
        cpu_in = input.cpu()
        if cpu_in.dtype == torch.float16:
            cpu_in = cpu_in.float()
        window = kwargs.get("window", None)
        if window is not None and window.is_cuda:
            kwargs["window"] = window.cpu().float() if window.dtype == torch.float16 else window.cpu()
        result = _original_stft(cpu_in, n_fft, *args, **kwargs)
        return result.to(device)
    return _original_stft(input, n_fft, *args, **kwargs)

def _patched_istft(input, n_fft, *args, **kwargs):
    if input.is_cuda:
        device = input.device
        cpu_in = input.cpu()
        if cpu_in.dtype == torch.float16:
            cpu_in = cpu_in.float()
        window = kwargs.get("window", None)
        if window is not None and window.is_cuda:
            kwargs["window"] = window.cpu().float() if window.dtype == torch.float16 else window.cpu()
        result = _original_istft(cpu_in, n_fft, *args, **kwargs)
        return result.to(device)
    return _original_istft(input, n_fft, *args, **kwargs)

def _patched_rfft(input, *args, **kwargs):
    if input.is_cuda:
        device = input.device
        cpu_in = input.cpu()
        if cpu_in.dtype == torch.float16:
            cpu_in = cpu_in.float()
        result = _original_rfft(cpu_in, *args, **kwargs)
        return result.to(device)
    if input.dtype == torch.float16:
        return _original_rfft(input.float(), *args, **kwargs)
    return _original_rfft(input, *args, **kwargs)

def _patched_irfft(input, *args, **kwargs):
    if input.is_cuda:
        device = input.device
        cpu_in = input.cpu()
        if cpu_in.dtype == torch.complex32 or (hasattr(torch, 'complex32') and cpu_in.dtype == torch.complex32):
            cpu_in = cpu_in.to(torch.complex64)
        result = _original_irfft(cpu_in, *args, **kwargs)
        return result.to(device)
    return _original_irfft(input, *args, **kwargs)

torch.stft = _patched_stft
torch.istft = _patched_istft
torch.fft.rfft = _patched_rfft
torch.fft.irfft = _patched_irfft


# === SDPA Fix for Multilingual Model ===
# The multilingual model uses output_attentions=True which is incompatible
# with SDPA (Scaled Dot-Product Attention) on Jetson. Switch to eager.
def _patch_llama_sdpa():
    try:
        from chatterbox.models.t3 import llama_configs
        if llama_configs.LLAMA_520M_CONFIG_DICT.get("attn_implementation") == "sdpa":
            llama_configs.LLAMA_520M_CONFIG_DICT["attn_implementation"] = "eager"
    except (ImportError, AttributeError):
        pass


# === Selective FP16 for Turbo Model: t3 + ve only ===
# Model sizes:  t3=815MB fp32 / 408MB fp16 (saves 407MB)
#               s3gen=507MB fp32 (stays fp32 — audio processing incompatible with fp16)
#               ve=3MB fp32 / 1MB fp16 (saves 2MB)
# Total savings: ~409MB. Free RAM: ~183MB -> ~592MB with Whisper running.
#
# Architecture of fp16/fp32 boundaries:
#   ve  (fp16) -> float16 speaker embedding -> T3Cond -> t3 (fp16) OK
#   t3  (fp16) -> integer speech tokens (no float dtype) -> s3gen OK
#   s3gen (fp32) -> all audio processing in float32
#
# Two patches required:
# 1. VoiceEncoder.forward: cuDNN LSTM requires input dtype to match weight dtype.
#    ve.embeds_from_mels computes float32 mels but feeds them to fp16 LSTM.
#    Fix: cast mels to model dtype (fp16) before LSTM.
# 2. T3.prepare_conditioning: ALL T3Cond tensors must match t3's dtype.
#    emotion_adv is created as float32 (torch.ones); speaker_emb from ve is fp16
#    (after VoiceEncoder patch), but other paths may produce float32 tensors.
#    Fix: cast all T3Cond tensors to t3's dtype before cond_enc.

def _patch_ve_forward_dtype():
    """Patch VoiceEncoder.forward to cast mels to model dtype before LSTM."""
    try:
        from chatterbox.models.voice_encoder.voice_encoder import VoiceEncoder
    except ImportError:
        return
    _orig_forward = VoiceEncoder.forward
    def _dtype_cast_forward(self, mels):
        dtype = next(self.parameters()).dtype
        if mels.dtype != dtype:
            mels = mels.to(dtype)
        return _orig_forward(self, mels)
    VoiceEncoder.forward = _dtype_cast_forward


def _patch_t3_cond_cast():
    """Patch T3.prepare_conditioning to cast T3Cond tensors to model dtype."""
    try:
        from chatterbox.models.t3.t3 import T3
        from chatterbox.models.t3.inference.t3_hparams import T3Cond
    except ImportError:
        try:
            from chatterbox.models.t3.t3 import T3, T3Cond
        except ImportError:
            return
    _orig_prepare = T3.prepare_conditioning
    def _cast_prepare_conditioning(self, t3_cond):
        dtype = next(self.parameters()).dtype
        # Cast all float tensors in T3Cond to model dtype
        speaker_emb = t3_cond.speaker_emb
        if speaker_emb is not None and speaker_emb.dtype != dtype:
            speaker_emb = speaker_emb.to(dtype)
        emotion_adv = getattr(t3_cond, 'emotion_adv', None)
        if emotion_adv is not None and isinstance(emotion_adv, torch.Tensor) and emotion_adv.dtype != dtype:
            emotion_adv = emotion_adv.to(dtype)
        # Rebuild T3Cond with cast tensors (handles both dataclass and namedtuple)
        try:
            new_cond = t3_cond.__class__(
                speaker_emb=speaker_emb,
                cond_prompt_speech_tokens=t3_cond.cond_prompt_speech_tokens,
                emotion_adv=emotion_adv,
            )
        except Exception:
            new_cond = t3_cond  # fallback: use original
        return _orig_prepare(self, new_cond)
    T3.prepare_conditioning = _cast_prepare_conditioning


def _patch_chatterbox_fp16():
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
    except ImportError:
        return

    _original_from_pretrained = ChatterboxTurboTTS.from_pretrained

    @classmethod
    def _fp16_from_pretrained(cls, device):
        model = _original_from_pretrained.__func__(cls, device)

        # Apply dtype-cast patches before converting weights
        _patch_ve_forward_dtype()
        _patch_t3_cond_cast()

        model.t3 = model.t3.half()
        # model.s3gen stays fp32
        model.ve = model.ve.half()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model

    ChatterboxTurboTTS.from_pretrained = _fp16_from_pretrained


def _patch_turbo_max_tokens():
    """Cap Turbo T3 max_new_tokens to text length instead of hardcoded 1000.

    ChatterboxTurboTTS.generate() passes max_new_tokens=1000 to T3.inference().
    On Jetson unified memory, each autoregressive step appends to the KV cache;
    1000 steps create a large peak CUDA allocation per chunk. After 20-30 chunks
    the NvMap free list fragments so badly that a fresh large allocation fails even
    when total free RAM appears sufficient (torch.cuda.memory_allocated() shows 0 MB
    but NvMapMemAllocInternalTagged returns error 12).

    Fix: scale max_new_tokens to ~4× text length, capped at 750. For a 150-char
    chunk that is 600 tokens — well above the ~250 observed tokens per chunk —
    whilst cutting peak KV cache size by ~40% vs the hardcoded 1000.

    Applied per-call (temporary class patch) so the multilingual model's own
    max_new_tokens values (set by _patch_multilingual_max_tokens) are unaffected.
    """
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
    except ImportError:
        return

    _original_generate = ChatterboxTurboTTS.generate

    def _scaled_generate(self, text, audio_prompt_path=None, **kwargs):
        max_tokens = max(300, min(750, int(len(text) * 4)))

        t3_cls = type(self.t3)
        orig_t3_inference = t3_cls.inference

        def _capped_inference(t3_self, *args, max_new_tokens=1000, **kw):
            return orig_t3_inference(
                t3_self, *args,
                max_new_tokens=min(max_tokens, max_new_tokens),
                **kw,
            )

        t3_cls.inference = _capped_inference
        try:
            return _original_generate(
                self, text, audio_prompt_path=audio_prompt_path, **kwargs
            )
        finally:
            t3_cls.inference = orig_t3_inference

    ChatterboxTurboTTS.generate = _scaled_generate


def _patch_multilingual_fp16():
    # NOTE: Multilingual model produces wonky audio with fp16 conversion.
    # Skip fp16 for now — runs in fp32. Uses more memory but sounds correct.
    pass


# === Multilingual Babble Fix ===
# The multilingual model hardcodes max_new_tokens=1000, which causes it to
# generate nonsensical babbling after the text has been spoken. We patch
# the generate method to scale max_new_tokens based on text length.
def _patch_multilingual_max_tokens():
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        import torch.nn.functional as F
    except ImportError:
        return

    _original_generate = ChatterboxMultilingualTTS.generate

    def _capped_generate(self, text, language_id, audio_prompt_path=None,
                         exaggeration=0.5, cfg_weight=0.5, temperature=0.8,
                         repetition_penalty=2.0, min_p=0.05, top_p=1.0):
        from chatterbox.mtl_tts import punc_norm, drop_invalid_tokens, T3Cond

        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None

        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(
            text, language_id=language_id.lower() if language_id else None
        ).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        # Scale max_new_tokens based on text length instead of hardcoded 1000
        # ~2 speech tokens per character, minimum 150, max 750
        max_tokens = max(150, min(750, len(text) * 2))

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=max_tokens,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            speech_tokens = speech_tokens[0]
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()

            # Trim trailing audio after speech ends.
            # Two-pass: (1) hard cap based on estimated speech duration,
            # (2) energy-based trim to remove trailing breathing/noise.
            import numpy as np
            # Pass 1: Duration cap — ~0.4s per word + 1s buffer
            word_count = len(text.split())
            max_duration_s = word_count * 0.4 + 1.0
            max_samples = int(max_duration_s * self.sr)
            if len(wav) > max_samples:
                wav = wav[:max_samples]
            # Pass 2: Energy-based trim for remaining trailing noise
            frame_size = int(self.sr * 0.05)  # 50ms frames
            threshold = 0.01
            end = len(wav)
            while end > frame_size:
                frame = wav[end - frame_size:end]
                rms = np.sqrt(np.mean(frame ** 2))
                if rms > threshold:
                    end = min(len(wav), end + int(self.sr * 0.1))
                    break
                end -= frame_size // 2
            wav = wav[:end]

            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    ChatterboxMultilingualTTS.generate = _capped_generate



# Apply patches when chatterbox modules are first imported
import importlib
import sys

class _ChatterboxPatchFinder:
    """Meta path finder that patches chatterbox after import."""
    _turbo_patched = False
    _multilingual_patched = False
    _llama_patched = False

    def find_module(self, name, path=None):
        if name == "chatterbox.tts_turbo" and not self._turbo_patched:
            return self
        if name == "chatterbox.mtl_tts" and not self._multilingual_patched:
            return self
        if name == "chatterbox.models.t3.llama_configs" and not self._llama_patched:
            return self
        return None

    def load_module(self, name):
        # Let normal import happen first
        sys.meta_path.remove(self)
        mod = importlib.import_module(name)
        sys.meta_path.insert(0, self)

        if name == "chatterbox.tts_turbo":
            _patch_chatterbox_fp16()
            _patch_turbo_max_tokens()
            self.__class__._turbo_patched = True

        elif name == "chatterbox.mtl_tts":
            _patch_multilingual_fp16()
            _patch_multilingual_max_tokens()
            self.__class__._multilingual_patched = True
        elif name == "chatterbox.models.t3.llama_configs":
            _patch_llama_sdpa()
            self.__class__._llama_patched = True

        return mod

sys.meta_path.insert(0, _ChatterboxPatchFinder())
