"""Streaming primitives for the Voxtral Realtime encoder, with causal-mode mel.

Vendored & adapted from Blaizzy/mlx-audio (mlx_audio/stt/models/voxtral_realtime/streaming.py)
under MIT license. Differences vs upstream:

  * ``StreamingMelCausal`` replaces ``StreamingMel``: causal STFT (left-only
    pad on first chunk, no right-reflect), per-frame running-max log
    normalization, drop-last-frame at chunk boundary, leftover-waveform
    stft cache. Matches ``raon.modules.voxtral_wrapper._extract_features_causal_streaming``.

  * ``StreamingFrameStack`` replaces ``StreamingDownsampler``: pure 4x reshape
    (frame-stacking) without applying the encoder's audio_language_projection_*
    layers — Raon uses its own ``input_adaptor`` afterwards.

  * Drops ``VoxtralStreamingSession`` and decoder-driven path: this file is for
    audio-encoder-only streaming used by the Raon duplex generator.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.core import eval as _force_compute  # alias to dodge non-MLX-aware lint hooks

from .encoder import AudioEncoder

WINDOW_SIZE = 400
HOP_LENGTH = 160
N_FFT = 400
NUM_MEL_BINS = 128


class StreamingMelCausal:
    """Incremental causal log-mel computation matching Raon's training-time path.

    Parity contract with ``_extract_features_causal_streaming``:
      Feed the same 16kHz samples through ``.append`` (any chunking) and
      concatenate the returned per-call outputs. The result equals the
      reference per-utterance causal log-mel, frame-for-frame, up to fp32
      rounding. Each adapter token consumes 8 mel frames at 10ms hop = 80ms.

    Args:
        mel_filters_mx: [freq_bins, n_mels] mel filter bank, precomputed.
        n_fft: STFT FFT size and window length.
        hop_length: STFT hop in samples.
    """

    def __init__(
        self,
        mel_filters_mx: mx.array,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
    ) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self._mel_filters_mx = mel_filters_mx  # [freq_bins, n_mels]

        # Periodic Hann window (matches torch.hann_window(n, periodic=True)).
        n = mx.arange(n_fft, dtype=mx.float32)
        self._window = 0.5 * (1.0 - mx.cos(2.0 * math.pi * n / n_fft))
        _force_compute(self._window)

        self._stft_cache_np: Optional[np.ndarray] = None
        self._is_first_chunk: bool = True
        self._running_max: float = float("-inf")

    def reset(self) -> None:
        self._stft_cache_np = None
        self._is_first_chunk = True
        self._running_max = float("-inf")

    def append(self, samples_np: np.ndarray) -> Optional[mx.array]:
        """Feed a 16kHz mono float32 chunk; return [mel_bins, n_new_frames] or None.

        Returns None when not enough samples have arrived to emit a frame.
        """
        if samples_np.size == 0:
            return None
        if samples_np.dtype != np.float32:
            samples_np = samples_np.astype(np.float32)
        samples_np = samples_np.reshape(-1)

        if self._is_first_chunk:
            waveform = np.concatenate(
                [np.zeros(self.n_fft // 2, dtype=np.float32), samples_np]
            )
            self._is_first_chunk = False
        else:
            assert self._stft_cache_np is not None
            waveform = np.concatenate([self._stft_cache_np, samples_np])

        total = waveform.shape[0]
        if total < self.n_fft:
            self._stft_cache_np = waveform
            return None

        num_frames = (total - self.n_fft) // self.hop_length + 1
        if num_frames <= 1:
            self._stft_cache_np = waveform
            return None

        emit_frames = num_frames - 1
        consumed = (emit_frames - 1) * self.hop_length + self.n_fft
        used = waveform[:consumed]

        # STFT (center=False, periodic Hann). Frame i covers [i*hop, i*hop+n_fft).
        frame_starts = np.arange(emit_frames) * self.hop_length
        idx = frame_starts[:, None] + np.arange(self.n_fft)[None, :]
        frames_np = used[idx]  # [emit_frames, n_fft]
        frames_mx = mx.array(frames_np, dtype=mx.float32) * self._window[None, :]
        spectrum = mx.fft.rfft(frames_mx, n=self.n_fft, axis=-1)
        magnitudes = mx.abs(spectrum) ** 2
        mel_spec = magnitudes @ self._mel_filters_mx  # [emit_frames, n_mels]
        log_spec = mx.log10(mx.maximum(mel_spec, 1e-10))  # [emit_frames, n_mels]

        # Running-max log normalization, in numpy (small, per-call).
        log_spec_np = np.array(log_spec)
        per_frame_max = log_spec_np.max(axis=1)  # [emit_frames]
        with_running = np.maximum(per_frame_max, self._running_max)
        running_vals = np.maximum.accumulate(with_running)  # cummax
        self._running_max = float(running_vals[-1])
        floor = (running_vals - 8.0)[:, None]  # [emit_frames, 1]
        log_spec_np = np.maximum(log_spec_np, floor)
        log_spec_np = (log_spec_np + 4.0) / 4.0

        # Stash leftover waveform (next frame starts at emit_frames * hop).
        self._stft_cache_np = waveform[emit_frames * self.hop_length :].copy()

        out = mx.array(log_spec_np.T, dtype=mx.float32)  # [mel_bins, n_new_frames]
        _force_compute(out)
        return out


class StreamingCausalConv1d:
    """Incremental causal Conv1d wrapper. Vendored verbatim from mlx-audio."""

    def __init__(self, causal_conv):
        self.conv = causal_conv  # CausalConv1d
        self.kernel_size = causal_conv.kernel_size
        self.stride = causal_conv.stride
        self.left_pad = causal_conv.padding
        self._keep = self.kernel_size - self.stride
        self._state: Optional[mx.array] = None
        self._initialized = False

    def reset(self) -> None:
        self._state = None
        self._initialized = False

    def step(self, x_new: mx.array) -> mx.array:
        if x_new.shape[0] == 0:
            return x_new[:0]
        if not self._initialized:
            if self._keep > 0:
                pad = mx.zeros((self._keep, x_new.shape[-1]), dtype=x_new.dtype)
                context = mx.concatenate([pad, x_new], axis=0)
            else:
                context = x_new
            self._initialized = True
        else:
            context = (
                mx.concatenate([self._state, x_new], axis=0)
                if self._state is not None
                else x_new
            )

        if context.shape[0] < self.kernel_size:
            self._state = context
            return mx.zeros((0, self.conv.conv.weight.shape[0]), dtype=x_new.dtype)

        out = self.conv.conv(context[None, :, :]).squeeze(0)
        n_out = out.shape[0]

        if self._keep > 0:
            leftover = context.shape[0] - n_out * self.stride
            if leftover <= 0:
                self._state = None
            elif leftover >= self._keep:
                self._state = context[-self._keep :]
            else:
                self._state = context[-leftover:]
        else:
            self._state = None
        return out


class StreamingConvStem:
    """Streaming version of AudioEncoder.conv_stem (two CausalConv1d + GELU)."""

    def __init__(self, encoder: AudioEncoder):
        self._c0 = StreamingCausalConv1d(encoder.conv_layers_0_conv)
        self._c1 = StreamingCausalConv1d(encoder.conv_layers_1_conv)

    def reset(self) -> None:
        self._c0.reset()
        self._c1.reset()

    def step(self, mel_chunk: mx.array) -> mx.array:
        target_dtype = self._c0.conv.conv.weight.dtype
        if mel_chunk.shape[1] == 0:
            return mx.zeros((0, self._c0.conv.conv.weight.shape[0]), dtype=target_dtype)
        x = mel_chunk.T.astype(target_dtype)  # [frames, num_mel_bins]
        x = self._c0.step(x)
        x = nn.gelu(x)
        x = self._c1.step(x)
        x = nn.gelu(x)
        return x


class StreamingEncoder:
    """Streaming wrapper over the encoder's transformer layers + final norm.

    Uses RotatingKVCache(max_size=sliding_window) per layer and tracks the
    global RoPE position. Parity with non-streaming `encode_full` for short
    inputs that fit in one window; equivalent to `encode_chunks` for longer.
    """

    def __init__(self, encoder: AudioEncoder):
        from mlx_lm.models.cache import RotatingKVCache

        self.encoder = encoder
        self._sw = encoder.config.sliding_window
        self._caches = [
            RotatingKVCache(max_size=self._sw, keep=0)
            for _ in range(len(encoder.transformer_layers))
        ]
        self._pos = 0

    def reset(self) -> None:
        from mlx_lm.models.cache import RotatingKVCache

        self._caches = [
            RotatingKVCache(max_size=self._sw, keep=0)
            for _ in range(len(self.encoder.transformer_layers))
        ]
        self._pos = 0

    def step(self, conv_chunk: mx.array) -> mx.array:
        chunk_len = conv_chunk.shape[0]
        if chunk_len == 0:
            return conv_chunk

        mask = self._caches[0].make_mask(chunk_len, window_size=self._sw, return_array=True)
        x = conv_chunk
        for i, layer in enumerate(self.encoder.transformer_layers):
            x = layer(x, self._pos, mask, cache=self._caches[i])
        out = self.encoder.transformer_norm(x)
        self._pos += chunk_len
        return out


class StreamingFrameStack:
    """4x downsample by frame-stacking: [n_frames, dim] -> [n_frames/4, dim*4].

    Replaces mlx-audio's StreamingDownsampler. Does NOT apply any projection;
    the caller (Raon's input_adaptor) projects the stacked output to 4096.
    """

    def __init__(self, downsample_factor: int = 4):
        self._ds = downsample_factor
        self._buf: Optional[mx.array] = None

    def reset(self) -> None:
        self._buf = None

    def step(self, encoded_chunk: mx.array) -> mx.array:
        """encoded_chunk: [n_frames, dim] -> [n_groups, dim*ds]."""
        if encoded_chunk.shape[0] == 0:
            dim = 0 if self._buf is None else self._buf.shape[-1] * self._ds
            return mx.zeros((0, dim), dtype=encoded_chunk.dtype)

        x = (
            mx.concatenate([self._buf, encoded_chunk], axis=0)
            if self._buf is not None and self._buf.shape[0] > 0
            else encoded_chunk
        )
        n = x.shape[0]
        usable = n - (n % self._ds)
        if usable == 0:
            self._buf = x
            dim = encoded_chunk.shape[-1] * self._ds
            return mx.zeros((0, dim), dtype=encoded_chunk.dtype)

        groups = usable // self._ds
        stacked = x[:usable].reshape(groups, x.shape[-1] * self._ds)
        self._buf = x[usable:] if usable < n else None
        return stacked
