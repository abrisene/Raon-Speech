# KV cache implementations for MLX inference.
# Adapted from mlx-examples and PersonaPlex/Kyutai Moshi MLX port.
# Original: Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any

import mlx.core as mx


class KVCache:
    def __init__(self, head_dim: int, n_kv_heads: int):
        self.n_kv_heads = n_kv_heads
        if isinstance(head_dim, int):
            self.k_head_dim = self.v_head_dim = head_dim
        elif isinstance(head_dim, tuple) and len(head_dim) == 2:
            self.k_head_dim, self.v_head_dim = head_dim
        else:
            raise ValueError("head_dim must be an int or a tuple of two ints")
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 256

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
        cache_position: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Update cache with new keys/values and return full cache.

        Args:
            keys: New key states [B, n_kv_heads, S, head_dim].
            values: New value states [B, n_kv_heads, S, head_dim].
            cache_position: If provided, explicit positions to write [S].
                This enables overwriting specific cache positions (for duplex).
                If None, appends sequentially at self.offset (standard behavior).
        """
        if cache_position is not None:
            # Explicit position mode: write K/V at specified positions
            pos = cache_position.tolist() if hasattr(cache_position, 'tolist') else list(cache_position)
            max_pos = max(pos) + 1

            # Ensure cache is large enough
            if self.keys is None or max_pos > self.keys.shape[2]:
                B = keys.shape[0]
                target_size = ((max_pos + self.step - 1) // self.step) * self.step
                k_shape = (B, self.n_kv_heads, target_size, self.k_head_dim)
                v_shape = (B, self.n_kv_heads, target_size, self.v_head_dim)
                new_k = mx.zeros(k_shape, keys.dtype)
                new_v = mx.zeros(v_shape, values.dtype)
                if self.keys is not None:
                    old_len = self.keys.shape[2]
                    new_k[..., :old_len, :] = self.keys[..., :old_len, :]
                    new_v[..., :old_len, :] = self.values[..., :old_len, :]
                self.keys = new_k
                self.values = new_v

            # Write at explicit positions
            for i, p in enumerate(pos):
                self.keys[..., p:p+1, :] = keys[..., i:i+1, :]
                self.values[..., p:p+1, :] = values[..., i:i+1, :]

            self.offset = max_pos
            return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]

        # Standard sequential append mode
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B = keys.shape[0]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, self.n_kv_heads, n_steps * self.step, self.k_head_dim)
            v_shape = (B, self.n_kv_heads, n_steps * self.step, self.v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                assert self.values is not None
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        assert self.values is not None
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def rewind(self, n: int) -> None:
        """Roll back the cache offset by n positions, allowing overwrite."""
        self.offset = max(0, self.offset - n)

    def reset(self):
        self.offset = 0
        self.keys = None
        self.values = None

    @property
    def state(self):
        return self.keys, self.values


class RotatingKVCache:
    def __init__(self, head_dim: int, n_kv_heads: int, max_size: int, keep: int = 0, step: int = 256):
        self.n_kv_heads = n_kv_heads
        if isinstance(head_dim, int):
            self.k_head_dim = self.v_head_dim = head_dim
        elif isinstance(head_dim, tuple) and len(head_dim) == 2:
            self.k_head_dim, self.v_head_dim = head_dim
        else:
            raise ValueError("head_dim must be an int or a tuple of two ints")
        self.keep = keep
        self.keys = None
        self.values = None
        self.offset = 0
        self.max_size = max_size
        self.step = step
        self._idx = 0

    def _trim(self, trim_size: int, v: mx.array, append: mx.array | None = None) -> mx.array:
        to_cat = []
        if trim_size > 0:
            to_cat = [v[..., : self.keep, :], v[..., trim_size + self.keep :, :]]
        else:
            to_cat = [v]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        prev = self.offset
        B, _, S = keys.shape[:3]

        if S > 1:
            if self.keys is None:
                self.keys = keys
                self.values = values
            else:
                trim_size = self.keys.shape[2] - self.max_size + 1
                self.keys = self._trim(trim_size, self.keys, keys)
                self.values = self._trim(trim_size, self.values, values)
            self.offset += S
            self._idx = self.keys.shape[2]
            return self.keys, self.values

        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, self.n_kv_heads, new_size, self.k_head_dim)
            v_shape = (B, self.n_kv_heads, new_size, self.v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                assert self.values is not None
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size

        if self._idx == self.max_size:
            self._idx = self.keep

        self.keys[..., self._idx : self._idx + 1, :] = keys
        assert self.values is not None
        self.values[..., self._idx : self._idx + 1, :] = values
        self.offset += 1
        self._idx += 1

        if self.offset < self.max_size:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        return self.keys, self.values

    def reset(self):
        self.offset = 0
        self._idx = 0
        self.keys = None
        self.values = None

    @property
    def state(self):
        return self.keys, self.values


def create_additive_causal_mask(N: int, offset: int = 0) -> mx.array:
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9


def create_attention_mask(h: mx.array, cache: Any | None = None) -> mx.array | None:
    T = h.shape[1]
    if T > 1:
        if cache is not None and cache[0] is not None:
            c = cache[0]
            if isinstance(c, RotatingKVCache):
                offset = min(c.max_size - 1, c.offset)
            else:
                offset = c.offset
        else:
            offset = 0
        mask = create_additive_causal_mask(T, offset)
        mask = mask.astype(h.dtype)
    else:
        mask = None
    return mask
