"""Pure MLX streaming audio encoder for Raon-SpeechChat duplex inference.

Replaces the previous PyTorch-wrapping path. Loads the Voxtral Realtime
encoder + Raon's input adaptor entirely in MLX. ``encode_frame(pcm)`` is
called once per duplex step (every 80ms / 1920 samples at 24kHz) and
returns thinker-space embeddings ``[1, n_adapter_tokens, 4096]`` ready for
``duplex_step`` to splice into the LLM input.

Streaming pipeline per call:
    pcm @24k -> resample -> StreamingMelCausal -> StreamingConvStem
    -> StreamingEncoder -> StreamingFrameStack -> InputAdaptor (5120 -> 4096)

The encoder uses a bounded ``RotatingKVCache(max_size=sliding_window)`` per
layer, matching the sliding-window-attention semantics the model was
trained with — this is the key behavioral difference vs the old PyTorch
``DynamicCache`` path, which let the encoder's KV grow unboundedly.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from scipy.signal import resample_poly

from .voxtral_mlx import (
    AudioEncoder,
    EncoderConfig,
    StreamingConvStem,
    StreamingEncoder,
    StreamingFrameStack,
    StreamingMelCausal,
)
from .voxtral_mlx.audio import mel_filters
from .voxtral_mlx.weights import (
    load_audio_encoder_config,
    load_encoder_weights,
    load_input_adaptor_weights,
)

logger = logging.getLogger(__name__)

SAMPLES_PER_FRAME_24K = 1920  # 80ms @ 24kHz
SAMPLES_PER_FRAME_16K = 1280  # 80ms @ 16kHz
INPUT_SR = 24000
ENCODER_SR = 16000


class _RaonInputAdaptor(nn.Module):
    """Raon's input adaptor: 5120 -> 4096 -> GELU -> 4096 -> RMSNorm.

    Weights loaded from ``input_adaptor.{proj.0,proj.2,post_norm}`` in the
    Raon HF checkpoint. Both linears are bias-free; post_norm is a RMSNorm.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj_0 = nn.Linear(in_dim, out_dim, bias=False)
        self.proj_2 = nn.Linear(out_dim, out_dim, bias=False)
        self.post_norm = nn.RMSNorm(out_dim, eps=1e-6)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu(self.proj_0(x))
        x = self.proj_2(x)
        return self.post_norm(x)


class StreamingAudioEncoder:
    """Pure-MLX streaming audio encoder. Singleton-cached per model_path.

    Public API matches the previous PyTorch-wrapping implementation:
        load() -> idempotent weight load
        reset() -> reset streaming state at session start
        encode_frame(pcm: mx.array) -> mx.array of shape [1, T_adapter, 4096]
    """

    def __init__(self, model_path: str, dtype: mx.Dtype = mx.float32) -> None:
        self._model_path = model_path
        self._dtype = dtype

        self._encoder: AudioEncoder | None = None
        self._input_adaptor: _RaonInputAdaptor | None = None

        self._smel: StreamingMelCausal | None = None
        self._sconv: StreamingConvStem | None = None
        self._senc: StreamingEncoder | None = None
        self._sframe: StreamingFrameStack | None = None

        self._mel_filters_mx: mx.array | None = None
        self._loaded = False
        self._encoder_type = "voxtral_mlx"

    # ----- weight loading -----------------------------------------------------

    def load(self) -> None:
        if self._loaded:
            return

        logger.info("Loading MLX Voxtral encoder from %s ...", self._model_path)

        ae_cfg = load_audio_encoder_config(self._model_path)
        config = EncoderConfig.from_hf(ae_cfg)
        logger.info(
            "encoder cfg: dim=%d layers=%d heads=%d head_dim=%d sw=%d ds=%d",
            config.dim, config.n_layers, config.n_heads, config.head_dim,
            config.sliding_window, config.downsample_factor,
        )

        encoder = AudioEncoder(config)
        enc_weights = load_encoder_weights(self._model_path, dtype=self._dtype)
        encoder.load_weights(list(enc_weights.items()))

        in_dim = config.dim * config.downsample_factor
        out_dim = 4096  # Raon thinker hidden_size
        adaptor = _RaonInputAdaptor(in_dim, out_dim)
        ad_weights = load_input_adaptor_weights(self._model_path, dtype=self._dtype)
        adaptor.load_weights(list(ad_weights.items()))

        self._encoder = encoder
        self._input_adaptor = adaptor

        fb = mel_filters(
            sample_rate=ENCODER_SR,
            n_fft=400,
            n_mels=config.num_mel_bins,
            f_min=0.0,
            f_max=8000.0,
            norm="slaney",
            mel_scale="slaney",
        ).astype(mx.float32)
        self._mel_filters_mx = fb

        self._build_streaming_state()
        self._loaded = True
        logger.info("MLX Voxtral encoder ready.")

    # ----- streaming session lifecycle ----------------------------------------

    def _build_streaming_state(self) -> None:
        assert self._encoder is not None and self._mel_filters_mx is not None
        self._smel = StreamingMelCausal(self._mel_filters_mx)
        self._sconv = StreamingConvStem(self._encoder)
        self._senc = StreamingEncoder(self._encoder)
        self._sframe = StreamingFrameStack(
            downsample_factor=self._encoder.config.downsample_factor
        )

    def reset(self) -> None:
        if not self._loaded:
            self.load()
        self._build_streaming_state()
        logger.info("MLX streaming encoder state reset.")

    # ----- per-frame inference -----------------------------------------------

    def encode_frame(self, pcm: Any) -> mx.array:
        """Encode one duplex audio frame.

        Args:
            pcm: 1920 samples of 24kHz mono PCM, as ``mx.array`` (any rank
                with last-axis = samples) or ``np.ndarray``.

        Returns:
            mx.array of shape ``[1, n_adapter_tokens, 4096]``. Typically
            ``n_adapter_tokens`` is 0 or 1 per call at steady state — the
            very first call may emit 0 adapter tokens because the causal
            STFT needs to accumulate enough samples past the n_fft//2
            left-pad.
        """
        if self._smel is None:
            self.reset()
        assert self._smel is not None
        assert self._sconv is not None
        assert self._senc is not None
        assert self._sframe is not None
        assert self._input_adaptor is not None

        if isinstance(pcm, mx.array):
            pcm_np = np.array(pcm, copy=False).reshape(-1).astype(np.float32)
        else:
            pcm_np = np.asarray(pcm, dtype=np.float32).reshape(-1)

        if INPUT_SR != ENCODER_SR:
            pcm_np = resample_poly(pcm_np, up=2, down=3).astype(np.float32)

        mel_chunk = self._smel.append(pcm_np)
        if mel_chunk is None or mel_chunk.shape[1] == 0:
            return mx.zeros(
                (1, 0, self._input_adaptor.proj_0.weight.shape[0]),
                dtype=self._dtype,
            )

        conv_out = self._sconv.step(mel_chunk)
        if conv_out.shape[0] == 0:
            return mx.zeros(
                (1, 0, self._input_adaptor.proj_0.weight.shape[0]),
                dtype=conv_out.dtype,
            )

        encoded = self._senc.step(conv_out)
        stacked = self._sframe.step(encoded)
        if stacked.shape[0] == 0:
            return mx.zeros(
                (1, 0, self._input_adaptor.proj_0.weight.shape[0]),
                dtype=stacked.dtype,
            )

        adapter_in = stacked.astype(self._dtype)
        adapter_out = self._input_adaptor(adapter_in)  # [n, 4096]
        return adapter_out[None, :, :]


# ---- module-level singleton (matches the previous public surface) -----------

_encoder_lock = threading.Lock()
_encoder_cache: dict[str, StreamingAudioEncoder] = {}


def get_streaming_encoder(model_path: str) -> StreamingAudioEncoder:
    """Get or create a cached streaming encoder for ``model_path``."""
    with _encoder_lock:
        if model_path not in _encoder_cache:
            encoder = StreamingAudioEncoder(model_path)
            encoder.load()
            _encoder_cache[model_path] = encoder
        return _encoder_cache[model_path]


__all__ = ["StreamingAudioEncoder", "get_streaming_encoder", "SAMPLES_PER_FRAME_24K"]
