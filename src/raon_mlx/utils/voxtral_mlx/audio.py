"""Mel filter bank computation for Voxtral Realtime.

Slaney-normalized 128-bin filter bank over 0..8kHz at 16kHz, computed once
at load time and stored as an mx.array. Same numerical convention as
WhisperFeatureExtractor / mlx_audio.dsp.mel_filters; vendored here so we
don't pull in mlx-audio (which requires transformers>=5).
"""

from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=None)
def mel_filters(
    sample_rate: int = 16000,
    n_fft: int = 400,
    n_mels: int = 128,
    f_min: float = 0.0,
    f_max: float | None = None,
    norm: str | None = "slaney",
    mel_scale: str = "slaney",
) -> mx.array:
    """Compute mel filter bank as [freq_bins, n_mels].

    Defaults match the Voxtral Realtime audio encoding config used by
    Raon-SpeechChat-9B (sr=16000, n_fft=400, n_mels=128, slaney norm/scale).
    """

    def hz_to_mel(freq: float, scale: str) -> float:
        if scale == "htk":
            return 2595.0 * math.log10(1.0 + freq / 700.0)
        f_sp = 200.0 / 3
        mels = freq / f_sp
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = math.log(6.4) / 27.0
        if freq >= min_log_hz:
            mels = min_log_mel + math.log(freq / min_log_hz) / logstep
        return mels

    def mel_to_hz_arr(mels: mx.array, scale: str) -> mx.array:
        if scale == "htk":
            return 700.0 * (mx.power(10.0, mels / 2595.0) - 1.0)
        f_sp = 200.0 / 3
        freqs = f_sp * mels
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = math.log(6.4) / 27.0
        return mx.where(
            mels >= min_log_mel,
            min_log_hz * mx.exp(logstep * (mels - min_log_mel)),
            freqs,
        )

    f_max = f_max if f_max is not None else sample_rate / 2.0

    n_freqs = n_fft // 2 + 1
    all_freqs = mx.linspace(0, sample_rate // 2, n_freqs)

    m_min = hz_to_mel(f_min, mel_scale)
    m_max = hz_to_mel(f_max, mel_scale)
    m_pts = mx.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz_arr(m_pts, mel_scale)

    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = mx.expand_dims(f_pts, 0) - mx.expand_dims(all_freqs, 1)

    down_slopes = (-slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    fb = mx.maximum(mx.zeros_like(down_slopes), mx.minimum(down_slopes, up_slopes))

    if norm == "slaney":
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        fb = fb * mx.expand_dims(enorm, 0)

    # Returned as [freq_bins, n_mels] so callers can do `magnitudes @ filters`.
    return fb
