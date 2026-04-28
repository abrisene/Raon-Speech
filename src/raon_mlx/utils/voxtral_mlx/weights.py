"""HF safetensors -> MLX param mapping for the Raon Voxtral encoder + input adaptor.

The Raon-SpeechChat-9B checkpoint stores the audio encoder with HF Voxtral
naming (``audio_encoder.encoder.layers.{i}.self_attn.{q,k,v,o}_proj.*``,
``audio_encoder.encoder.embedder.{conv1,conv2}.*``,
``audio_encoder.encoder.norm.weight``) and the Raon-specific input adaptor
under ``input_adaptor.*``. This module:

  1. Streams the right tensors out of the safetensors shards (resolving local
     paths or HF Hub IDs as needed).
  2. Renames keys to the MLX module structure used by ``encoder.AudioEncoder``.
  3. Transposes Conv1d weights from PyTorch ``[out, in, k]`` to MLX ``[out, k, in]``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import mlx.core as mx


def _resolve_safetensors_files(model_path: str) -> tuple[Path, list[Path]]:
    """Return (model_dir, list of safetensors shard paths) for a local or HF repo."""
    p = Path(model_path)
    if p.is_dir():
        idx = p / "model.safetensors.index.json"
        if idx.exists():
            with open(idx) as f:
                index = json.load(f)
            shards = sorted({p / fn for fn in index["weight_map"].values()})
            return p, list(shards)
        single = p / "model.safetensors"
        if single.exists():
            return p, [single]
        raise FileNotFoundError(
            f"No model.safetensors[.index.json] found in {p}"
        )

    # HF Hub path: download index + shards on demand.
    from huggingface_hub import hf_hub_download

    try:
        idx_path = Path(
            hf_hub_download(repo_id=model_path, filename="model.safetensors.index.json")
        )
        with open(idx_path) as f:
            index = json.load(f)
        shard_names = sorted(set(index["weight_map"].values()))
        shards = [
            Path(hf_hub_download(repo_id=model_path, filename=name))
            for name in shard_names
        ]
        return idx_path.parent, shards
    except Exception:
        single = Path(hf_hub_download(repo_id=model_path, filename="model.safetensors"))
        return single.parent, [single]


def _stream_keys(
    shard_paths: Iterable[Path], wanted_prefixes: tuple[str, ...]
) -> dict[str, mx.array]:
    """Return a flat dict of tensors whose keys start with any of ``wanted_prefixes``."""
    out: dict[str, mx.array] = {}
    for shard in shard_paths:
        weights = mx.load(str(shard))
        for k, v in weights.items():
            if k.startswith(wanted_prefixes):
                out[k] = v
    return out


# ---- Mapper helpers ----------------------------------------------------------

_HF_ENC_PREFIX = "audio_encoder.encoder."
_HF_ADAPTOR_PREFIX = "input_adaptor."


def _map_encoder_key(hf_key: str) -> str | None:
    """Map an ``audio_encoder.encoder.*`` HF key to its MLX equivalent.

    Returns None if the key isn't recognized (caller should warn / skip).
    """
    rest = hf_key[len(_HF_ENC_PREFIX) :]

    # Conv stem: embedder.conv1.{w,b}, embedder.conv2.{w,b}
    if rest.startswith("embedder.conv1."):
        param = rest[len("embedder.conv1.") :]
        return f"conv_layers_0_conv.conv.{param}"
    if rest.startswith("embedder.conv2."):
        param = rest[len("embedder.conv2.") :]
        return f"conv_layers_1_conv.conv.{param}"

    # Final norm
    if rest == "norm.weight":
        return "transformer_norm.weight"

    # Layers: layers.{i}.{...}
    if rest.startswith("layers."):
        tail = rest[len("layers.") :]  # "{i}.<...>"
        idx_str, param_path = tail.split(".", 1)

        # Self-attention
        if param_path.startswith("self_attn."):
            sub = param_path[len("self_attn.") :]
            mapping = {
                "q_proj": "wq",
                "k_proj": "wk",
                "v_proj": "wv",
                "o_proj": "wo",
            }
            for hf_name, mlx_name in mapping.items():
                if sub.startswith(hf_name + "."):
                    suffix = sub[len(hf_name) + 1 :]  # "weight" or "bias"
                    return f"transformer_layers.{idx_str}.attention.{mlx_name}.{suffix}"
            return None

        # Norm before attention -> attention_norm
        if param_path == "self_attn_layer_norm.weight":
            return f"transformer_layers.{idx_str}.attention_norm.weight"

        # Norm before FFN -> ffn_norm
        if param_path == "final_layer_norm.weight":
            return f"transformer_layers.{idx_str}.ffn_norm.weight"

        # MLP: gate_proj -> w1, up_proj -> w3, down_proj -> w2
        if param_path.startswith("mlp."):
            sub = param_path[len("mlp.") :]
            mapping = {
                "gate_proj": "feed_forward_w1",
                "up_proj": "feed_forward_w3",
                "down_proj": "feed_forward_w2",
            }
            for hf_name, mlx_name in mapping.items():
                if sub.startswith(hf_name + "."):
                    suffix = sub[len(hf_name) + 1 :]
                    return f"transformer_layers.{idx_str}.{mlx_name}.{suffix}"
            return None

    return None


def _maybe_transpose_conv(mlx_key: str, tensor: mx.array) -> mx.array:
    """PyTorch Conv1d weights are [out, in, k]; MLX needs [out, k, in]."""
    if "conv_layers_" in mlx_key and mlx_key.endswith(".weight") and tensor.ndim == 3:
        return tensor.transpose(0, 2, 1)
    return tensor


# ---- Public API --------------------------------------------------------------


def load_encoder_weights(model_path: str, dtype: mx.Dtype = mx.float32) -> dict[str, mx.array]:
    """Load HF audio_encoder.encoder.* weights remapped to ``AudioEncoder`` keys.

    Returns a flat dict of MLX-named keys -> mx.array. Caller can then call
    ``model.load_weights(list(weights.items()))``.
    """
    _, shards = _resolve_safetensors_files(model_path)
    raw = _stream_keys(shards, (_HF_ENC_PREFIX,))
    out: dict[str, mx.array] = {}
    skipped: list[str] = []
    for hf_key, tensor in raw.items():
        mlx_key = _map_encoder_key(hf_key)
        if mlx_key is None:
            skipped.append(hf_key)
            continue
        tensor = _maybe_transpose_conv(mlx_key, tensor)
        if tensor.dtype != dtype:
            tensor = tensor.astype(dtype)
        out[mlx_key] = tensor
    if skipped:
        # Surface unmapped keys explicitly — silent skip is how we got here last time.
        import logging

        logging.getLogger(__name__).warning(
            "voxtral_mlx: %d unmapped HF encoder keys (sample: %s)",
            len(skipped),
            skipped[:5],
        )
    return out


def load_input_adaptor_weights(
    model_path: str, dtype: mx.Dtype = mx.float32
) -> dict[str, mx.array]:
    """Load Raon's ``input_adaptor.*`` weights under MLX-friendly keys.

    Returned dict has keys: ``proj_0.weight``, ``proj_2.weight``, ``post_norm.weight``.
    """
    _, shards = _resolve_safetensors_files(model_path)
    raw = _stream_keys(shards, (_HF_ADAPTOR_PREFIX,))
    name_map = {
        "input_adaptor.proj.0.weight": "proj_0.weight",
        "input_adaptor.proj.2.weight": "proj_2.weight",
        "input_adaptor.post_norm.weight": "post_norm.weight",
    }
    out: dict[str, mx.array] = {}
    for hf_key, tensor in raw.items():
        mlx_key = name_map.get(hf_key)
        if mlx_key is None:
            continue
        if tensor.dtype != dtype:
            tensor = tensor.astype(dtype)
        out[mlx_key] = tensor
    missing = set(name_map.values()) - set(out.keys())
    if missing:
        import logging

        logging.getLogger(__name__).warning(
            "voxtral_mlx: missing input_adaptor keys: %s", sorted(missing)
        )
    return out


def load_audio_encoder_config(model_path: str) -> dict:
    """Load the ``audio_encoder_config`` sub-dict of the Raon HF config."""
    p = Path(model_path)
    if p.is_dir():
        cfg_path = p / "config.json"
    else:
        from huggingface_hub import hf_hub_download

        cfg_path = Path(hf_hub_download(repo_id=model_path, filename="config.json"))
    with open(cfg_path) as f:
        cfg = json.load(f)
    return cfg["audio_encoder_config"]


__all__ = [
    "load_encoder_weights",
    "load_input_adaptor_weights",
    "load_audio_encoder_config",
]
