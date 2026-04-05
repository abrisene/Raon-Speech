# Qwen3.5 hybrid SSM-transformer backbone for MLX.
# Adapted from mlx-lm's qwen3_5.py implementation.
# Original: Copyright © 2026 Apple Inc.
#
# This implements the "GatedDeltaNet" linear attention layers that replace
# standard attention in 75% of Qwen3.5's layers, making it a hybrid
# SSM-transformer architecture.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..modules.kv_cache import KVCache, create_additive_causal_mask


class _MLP(nn.Module):
    """SwiGLU MLP (same structure as Qwen3, standalone for clean init)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, xs: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(xs)) * self.up_proj(xs))


@dataclass
class Qwen3_5Config:
    hidden_size: int = 4096
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    num_hidden_layers: int = 32
    intermediate_size: int = 12288
    head_dim: int = 256
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    rope_theta: float = 100000.0
    max_position_embeddings: int = 262144
    full_attention_interval: int = 4
    # Linear attention params
    linear_num_value_heads: int = 32
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    # Partial rotary for full attention
    partial_rotary_factor: float = 0.25

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


class ArraysCache:
    """Cache for SSM state (conv state + recurrent state)."""

    def __init__(self, size: int = 2):
        self._cache: list[mx.array | None] = [None] * size

    def __getitem__(self, idx: int) -> mx.array | None:
        return self._cache[idx]

    def __setitem__(self, idx: int, val: mx.array | None):
        self._cache[idx] = val

    @property
    def offset(self) -> int:
        # Used by mask creation — SSM tracks state differently
        if self._cache[0] is not None:
            return 1  # Signal that we have state
        return 0

    def reset(self):
        self._cache = [None] * len(self._cache)


class RMSNormGated(nn.Module):
    """RMSNorm with gating (used in GatedDeltaNet output)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones(dim)
        self.eps = eps

    def __call__(self, x: mx.array, z: mx.array) -> mx.array:
        normed = mx.fast.rms_norm(x, self.weight, self.eps)
        return normed * nn.silu(z)


class GatedDeltaNet(nn.Module):
    """Linear attention layer using gated delta update (SSM-style).

    Replaces standard attention in 75% of Qwen3.5 layers.
    """

    def __init__(self, cfg: Qwen3_5Config):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_v_heads = cfg.linear_num_value_heads
        self.num_k_heads = cfg.linear_num_key_heads
        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = cfg.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = mx.ones(self.num_v_heads)
        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)

        self.norm = RMSNormGated(self.head_v_dim, eps=cfg.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array | None = None,
        cache: ArraysCache | None = None,
    ) -> mx.array:
        from mlx_lm.models.gated_delta import gated_delta_update

        B, S, _ = inputs.shape

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            cache[0] = conv_input[:, -(self.conv_kernel_size - 1):]
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, state = gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, state, mask, use_kernel=True)

        if cache is not None:
            cache[1] = state

        out = self.norm(out, z)
        out = self.out_proj(out.reshape(B, S, -1))
        return out


class Qwen3_5Attention(nn.Module):
    """Full attention layer for Qwen3.5 (used in every 4th layer).

    Similar to Qwen3 attention but with:
    - Larger head_dim (256 vs 128)
    - Fewer heads (16 vs 32)
    - Partial rotary embedding
    - Output gating
    """

    def __init__(self, cfg: Qwen3_5Config):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scale = cfg.head_dim ** (-0.5)

        # q_proj is 2x because of output gating: [queries, gate]
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

        # Partial rotary: only rotate partial_rotary_factor of the head_dim
        rotary_dim = int(self.head_dim * cfg.partial_rotary_factor)
        self.rope = nn.RoPE(rotary_dim, traditional=False, base=cfg.rope_theta)

    def __call__(
        self,
        xs: mx.array,
        cache: KVCache | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        B, T, _ = xs.shape

        # q_proj outputs [queries, gate] interleaved per head
        q_proj_out = self.q_proj(xs)
        queries, gate = mx.split(
            q_proj_out.reshape(B, T, self.n_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, T, -1)  # [B, T, n_heads * head_dim]

        k = self.k_proj(xs).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(xs).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        k = self.k_norm(k)

        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        k = self.rope(k, offset=offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        if self.n_kv_heads < self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        out = mx.fast.scaled_dot_product_attention(queries, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        # Output gating
        return self.o_proj(out * mx.sigmoid(gate))


class Qwen3_5DecoderLayer(nn.Module):
    """Hybrid decoder layer — routes between GatedDeltaNet and full attention."""

    def __init__(self, cfg: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.is_linear = (layer_idx + 1) % cfg.full_attention_interval != 0

        if self.is_linear:
            self.linear_attn = GatedDeltaNet(cfg)
        else:
            self.self_attn = Qwen3_5Attention(cfg)

        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = _MLP(cfg.hidden_size, cfg.intermediate_size)

    def __call__(self, xs: mx.array, cache=None, mask=None) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(xs), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(xs), cache=cache, mask=mask)
        h = xs + r
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen3_5Model(nn.Module):
    """Qwen3.5 hybrid SSM-transformer backbone.

    Drop-in replacement for Qwen3Model when used with Raon.
    Same interface: __call__ returns (normed_output, pre_norm_output).
    """

    def __init__(self, cfg: Qwen3_5Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [Qwen3_5DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        cache: list | None = None,
        mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Forward pass — same return signature as Qwen3Model.

        Returns (normed_output, pre_norm_output) for compatibility with Raon's
        talker projection which needs pre-norm hidden states.
        """
        if inputs_embeds is None:
            assert input_ids is not None
            xs = self.embed_tokens(input_ids)
        else:
            xs = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        # Build masks for attention and SSM layers
        if xs.shape[1] > 1:
            # Find first attention layer to get its cache offset
            fa_idx = self.cfg.full_attention_interval - 1
            fa_cache = cache[fa_idx] if fa_idx < len(cache) else None
            offset = fa_cache.offset if fa_cache is not None and hasattr(fa_cache, 'offset') else 0
            attn_mask = create_additive_causal_mask(xs.shape[1], offset)
            attn_mask = attn_mask.astype(xs.dtype)
        else:
            attn_mask = None

        # SSM mask: True for valid positions, None for single-step
        ssm_mask = mx.ones((xs.shape[0], xs.shape[1]), dtype=mx.bool_) if xs.shape[1] > 1 else None

        for i, layer in enumerate(self.layers):
            layer_mask = ssm_mask if layer.is_linear else attn_mask
            xs = layer(xs, cache=cache[i], mask=layer_mask)

        pre_norm = xs
        return self.norm(xs), pre_norm

    def make_cache(self) -> list:
        """Create cache for all layers — ArraysCache for SSM, KVCache for attention."""
        caches = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=2))
            else:
                caches.append(KVCache(head_dim=self.cfg.head_dim, n_kv_heads=self.cfg.num_key_value_heads))
        return caches

    def load_raon_weights(self, weights: dict[str, mx.array]):
        """Load text_model weights (same interface as Qwen3Model)."""
        self.load_weights(list(weights.items()), strict=False)
