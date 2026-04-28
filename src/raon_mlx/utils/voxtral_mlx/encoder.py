"""Pure MLX implementation of the Voxtral Realtime causal audio encoder.

Vendored from Blaizzy/mlx-audio (mlx_audio/stt/models/voxtral_realtime/encoder.py),
adapted for Raon-SpeechChat:
  * Drops audio_language_projection_* (Raon uses its own input_adaptor).
  * Builder accepts the HF audio_encoder config dict directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class EncoderConfig:
    dim: int = 1280
    n_layers: int = 32
    n_heads: int = 32
    head_dim: int = 64
    hidden_dim: int = 5120
    norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    sliding_window: int = 750
    downsample_factor: int = 4
    num_mel_bins: int = 128

    @classmethod
    def from_hf(cls, ae_cfg: dict) -> "EncoderConfig":
        """Build from the ``audio_encoder_config`` sub-dict of a Raon HF config.

        Recognized keys (HF Voxtral Realtime):
            hidden_size, intermediate_size, num_hidden_layers,
            num_attention_heads, head_dim, num_mel_bins, rms_norm_eps,
            rope_theta, sliding_window, downsample_factor.
        """
        rope_params = ae_cfg.get("rope_parameters") or {}
        rope_theta = rope_params.get(
            "rope_theta", ae_cfg.get("rope_theta", 1_000_000.0)
        )
        return cls(
            dim=int(ae_cfg.get("hidden_size", 1280)),
            n_layers=int(ae_cfg.get("num_hidden_layers", 32)),
            n_heads=int(ae_cfg.get("num_attention_heads", 32)),
            head_dim=int(ae_cfg.get("head_dim", 64)),
            hidden_dim=int(ae_cfg.get("intermediate_size", 5120)),
            norm_eps=float(ae_cfg.get("rms_norm_eps", 1e-5)),
            rope_theta=float(rope_theta),
            sliding_window=int(ae_cfg.get("sliding_window", 750)),
            downsample_factor=int(ae_cfg.get("downsample_factor", 4)),
            num_mel_bins=int(ae_cfg.get("num_mel_bins", 128)),
        )


class CausalConv1d(nn.Module):
    """Causal 1D convolution with left-only padding (NLC layout)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size - stride
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: [batch, seq, channels]
        if self.padding > 0:
            x = mx.pad(x, [(0, 0), (self.padding, 0), (0, 0)])
        return self.conv(x)


class EncoderAttention(nn.Module):
    """Multi-head attention with RoPE and selective biases (q,v,o yes; k no)."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window
        self.rope_theta = config.rope_theta
        attn_dim = config.n_heads * config.head_dim

        self.wq = nn.Linear(config.dim, attn_dim, bias=True)
        self.wk = nn.Linear(config.dim, attn_dim, bias=False)
        self.wv = nn.Linear(config.dim, attn_dim, bias=True)
        self.wo = nn.Linear(attn_dim, config.dim, bias=True)

    def __call__(self, x: mx.array, rope_offset: int, mask, cache=None) -> mx.array:
        seq_len = x.shape[0]
        q = self.wq(x).reshape(1, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(1, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(1, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=rope_offset)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=rope_offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(seq_len, self.n_heads * self.head_dim)
        return self.wo(attn_out)


class EncoderLayer(nn.Module):
    """One transformer block: pre-RMSNorm, attention, residual, pre-RMSNorm, SwiGLU FFN, residual."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.attention = EncoderAttention(config)
        self.ffn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        # SwiGLU FFN: w1=gate (no bias), w3=up (no bias), w2=down (bias).
        self.feed_forward_w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.feed_forward_w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.feed_forward_w2 = nn.Linear(config.hidden_dim, config.dim, bias=True)

    def __call__(self, x: mx.array, rope_offset: int, mask, cache=None) -> mx.array:
        h = self.attention_norm(x)
        h = self.attention(h, rope_offset, mask, cache=cache)
        x = x + h

        h = self.ffn_norm(x)
        gate = nn.silu(self.feed_forward_w1(h))
        up = self.feed_forward_w3(h)
        x = x + self.feed_forward_w2(gate * up)
        return x


class AudioEncoder(nn.Module):
    """Conv stem + N-layer causal transformer + final norm. No projector.

    The Raon checkpoint applies its own ``input_adaptor`` (5120 -> 4096) on
    the frame-stacked encoder output, so we do NOT instantiate
    ``audio_language_projection_*`` here.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        self.conv_layers_0_conv = CausalConv1d(
            config.num_mel_bins, config.dim, kernel_size=3, stride=1
        )
        self.conv_layers_1_conv = CausalConv1d(
            config.dim, config.dim, kernel_size=3, stride=2
        )

        self.transformer_layers = [EncoderLayer(config) for _ in range(config.n_layers)]
        self.transformer_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

    def conv_stem(self, mel: mx.array) -> mx.array:
        """Run conv stem on a [mel_bins, frames] log-mel.

        Returns ``[T_conv, dim]`` where ``T_conv = ceil(mel_frames / 2)``
        (conv2 has stride 2). Caller is responsible for truncating to a
        multiple of ``downsample_factor`` if frame-stacking afterwards —
        truncate the *trailing* rows to match the Raon reference path,
        which does ``usable_mel = (n // ds) * ds`` (drops the tail).
        """
        x = mel.T[None, :, :]
        x = nn.gelu(self.conv_layers_0_conv(x))
        x = nn.gelu(self.conv_layers_1_conv(x))
        x = x.squeeze(0)
        return x

    def encode_full(self, conv_out: mx.array) -> mx.array:
        """Non-streaming encode using SDPA causal mask. Caller handles downsample."""
        x = conv_out
        for layer in self.transformer_layers:
            x = layer(x, 0, "causal")
        return self.transformer_norm(x)
