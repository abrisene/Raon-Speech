# Qwen3 transformer backbone for MLX.
# Implements the "thinker" (36-layer Qwen3) used as Raon-Speech's language model backbone.

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ..modules.kv_cache import KVCache, create_additive_causal_mask


@dataclass
class Qwen3Config:
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    num_hidden_layers: int = 36
    intermediate_size: int = 12288
    head_dim: int = 128
    vocab_size: int = 153723
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    max_position_embeddings: int = 262144

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


class Qwen3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones(dim)
        self.eps = eps

    def __call__(self, xs: mx.array) -> mx.array:
        return mx.fast.rms_norm(xs, self.weight, self.eps)


class Qwen3RotaryEmbedding(nn.Module):
    """RoPE for Qwen3 with configurable theta."""

    def __init__(self, head_dim: int, theta: float = 5000000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

    def __call__(self, q: mx.array, k: mx.array, offset: int = 0) -> tuple[mx.array, mx.array]:
        seq_len = q.shape[2]
        positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)
        dim = self.head_dim
        freqs = positions[:, None] / mx.power(self.theta, mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        cos = mx.cos(freqs)  # [seq_len, dim//2]
        sin = mx.sin(freqs)  # [seq_len, dim//2]
        # Apply rotary embeddings
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        return q, k


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply rotary position embeddings. x shape: [B, heads, seq, dim]."""
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    # cos/sin shape: [seq, d] -> broadcast to [1, 1, seq, d]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scale = cfg.head_dim ** (-0.5)

        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)

        # QK normalization (Qwen3 feature)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

        self.rope = Qwen3RotaryEmbedding(self.head_dim, theta=cfg.rope_theta)

    def __call__(
        self,
        xs: mx.array,
        cache: KVCache | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        B, T, _ = xs.shape

        q = self.q_proj(xs).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(xs).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(xs).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        offset = cache.offset if cache is not None else 0
        q, k = self.rope(q, k, offset=offset)

        # KV cache
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # GQA: repeat KV heads to match Q heads
        if self.n_kv_heads < self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(out)


class Qwen3MLP(nn.Module):
    """SwiGLU MLP: gate_proj * silu(up_proj(x)) -> down_proj."""

    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, xs: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(xs)) * self.up_proj(xs))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.self_attn = Qwen3Attention(cfg)
        self.mlp = Qwen3MLP(cfg)
        self.input_layernorm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(
        self,
        xs: mx.array,
        cache: KVCache | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        residual = xs
        xs = self.input_layernorm(xs)
        xs = self.self_attn(xs, cache=cache, mask=mask)
        xs = residual + xs

        residual = xs
        xs = self.post_attention_layernorm(xs)
        xs = self.mlp(xs)
        xs = residual + xs
        return xs


class Qwen3Model(nn.Module):
    """Qwen3 transformer backbone (thinker).

    This is the core language model that processes text and audio token embeddings.
    It does NOT include the token embedding layer or lm_head — those are handled
    by the parent RaonModel which manages the shared vocabulary across text and
    audio tokens.
    """

    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [Qwen3DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        cache: list[KVCache] | None = None,
        mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Forward pass through the Qwen3 model.

        Returns:
            Tuple of (normed_output, pre_norm_output).
            normed_output: After final RMSNorm — used for lm_head/text logits.
            pre_norm_output: Last layer output before norm — used for talker projection.
        """
        if inputs_embeds is None:
            assert input_ids is not None
            xs = self.embed_tokens(input_ids)
        else:
            xs = inputs_embeds

        # Build causal mask if not provided and seq_len > 1
        if mask is None and xs.shape[1] > 1:
            offset = cache[0].offset if cache is not None else 0
            mask = create_additive_causal_mask(xs.shape[1], offset)
            mask = mask.astype(xs.dtype)

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            xs = layer(xs, cache=layer_cache, mask=mask)

        pre_norm = xs
        return self.norm(xs), pre_norm

    def make_cache(self) -> list[KVCache]:
        return [
            KVCache(head_dim=self.cfg.head_dim, n_kv_heads=self.cfg.num_key_value_heads)
            for _ in self.layers
        ]

    def load_raon_weights(self, weights: dict[str, mx.array]):
        """Load text_model weights from a Raon checkpoint.

        Args:
            weights: Dict with keys already stripped of the 'text_model.' prefix.
        """
        mapped = []
        for k, v in weights.items():
            # Keys map directly: HF text_model.* matches our Qwen3Model structure
            # e.g. layers.0.self_attn.q_proj.weight -> self.layers[0].self_attn.q_proj.weight
            mapped.append((k, v))

        self.load_weights(mapped, strict=False)
