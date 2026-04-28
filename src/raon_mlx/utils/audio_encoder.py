# Audio encoder for STT — uses PyTorch for the Whisper-like AuT encoder,
# then converts output to MLX for the rest of the pipeline.
#
# The audio encoder runs once per utterance (not in the autoregressive loop)
# so keeping it in PyTorch has minimal performance impact.

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def encode_audio(
    audio_path: str,
    model_path: str,
    input_adaptor_proj_0: mx.array,
    input_adaptor_proj_2: mx.array,
    input_adaptor_post_norm_weight: mx.array,
    max_audio_chunk_length: int | None = 192000,
) -> tuple[mx.array, mx.array]:
    """Encode audio file to thinker-space embeddings.

    Uses PyTorch AuT encoder, then applies input adaptor in MLX.
    """
    import torch
    import soundfile as sf

    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio_t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    if sr != 24000:
        import torchaudio
        audio_t = torchaudio.functional.resample(audio_t.squeeze(0), orig_freq=sr, new_freq=24000).unsqueeze(0)

    audio_lengths = torch.tensor([audio_t.shape[-1]], dtype=torch.long)
    if max_audio_chunk_length is not None and audio_t.shape[-1] > max_audio_chunk_length:
        audio_t = audio_t[:, :, :max_audio_chunk_length]
        audio_lengths = torch.tensor([max_audio_chunk_length], dtype=torch.long)

    return encode_audio_tensor(audio_t, audio_lengths, model_path,
                                input_adaptor_proj_0, input_adaptor_proj_2, input_adaptor_post_norm_weight)


def encode_audio_tensor(
    audio_tensor,
    audio_lengths,
    model_path: str,
    input_adaptor_proj_0: mx.array,
    input_adaptor_proj_2: mx.array,
    input_adaptor_post_norm_weight: mx.array,
) -> tuple[mx.array, mx.array]:
    """Encode a PyTorch audio tensor to thinker-space embeddings.

    Args:
        audio_tensor: Raw audio [1, 1, samples] or [1, samples] PyTorch tensor.
        audio_lengths: Valid sample lengths [1] PyTorch tensor.
        model_path: Path to HF checkpoint (for audio encoder weights).
        input_adaptor_proj_0: Input adaptor weight [4096, 2048].
        input_adaptor_proj_2: Input adaptor weight [4096, 4096].
        input_adaptor_post_norm_weight: RMSNorm weight [4096].

    Returns:
        Tuple of (audio_embeds [1, total_frames, 4096], mask [1, total_frames]).
        If the processor chunked a long utterance into multiple batch rows, the
        per-row encoder outputs are concatenated back into a single temporal
        sequence for STT generation.
    """
    import torch

    if audio_tensor.ndim == 2:
        audio_tensor = audio_tensor.unsqueeze(1)  # [B, samples] -> [B, 1, samples]

    encoder = _get_audio_encoder(model_path)
    with torch.no_grad():
        output = encoder(audio=audio_tensor.float(), audio_lengths=audio_lengths)
        embeds = output.embeds  # [1, num_frames, 2048]
    embeds_np = embeds.cpu().float().numpy()

    # The processor may chunk a long utterance into multiple rows. Collapse the
    # batch back into one continuous sequence of frame embeddings.
    if embeds_np.ndim != 3:
        raise ValueError(f"Expected encoder embeds to be rank-3, got shape {embeds_np.shape}")
    if embeds_np.shape[0] == 1:
        merged_np = embeds_np
    else:
        merged_np = embeds_np.reshape(1, embeds_np.shape[0] * embeds_np.shape[1], embeds_np.shape[2])

    embeds_mx = mx.array(merged_np)
    mask_mx = mx.ones((1, embeds_mx.shape[1]), dtype=mx.bool_)

    # Input adaptor: 2-layer MLP (GELU) + RMSNorm
    x = nn.gelu(embeds_mx @ input_adaptor_proj_0.T)
    x = x @ input_adaptor_proj_2.T
    x = mx.fast.rms_norm(x, input_adaptor_post_norm_weight, 1e-6)

    return x, mask_mx


_audio_encoder_cache = None


def _get_audio_encoder(model_path: str):
    """Lazy-load the PyTorch audio encoder (cached)."""
    global _audio_encoder_cache
    if _audio_encoder_cache is not None:
        return _audio_encoder_cache

    import torch
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoderConfig,
    )
    from transformers.models.voxtral_realtime.configuration_voxtral_realtime import VoxtralRealtimeEncoderConfig

    from raon.modules.audio_encoder import AuTWrapper
    from raon.modules.voxtral_wrapper import VoxtralWrapper
    from raon.utils.misc import load_safetensors_by_prefix

    # Load config
    import json
    from pathlib import Path
    with open(Path(model_path) / "config.json") as f:
        cfg = json.load(f)

    ae_cfg = cfg["audio_encoder_config"]
    model_type = ae_cfg.get("model_type", "")

    if "voxtral" in model_type:
        vox_cfg = VoxtralRealtimeEncoderConfig(**ae_cfg)
        if not hasattr(vox_cfg, "rope_theta") or getattr(vox_cfg, "rope_theta", None) is None:
            vox_cfg.rope_theta = ae_cfg.get("rope_theta", 1000000.0)
        encoder = VoxtralWrapper.from_config(vox_cfg, dtype=torch.float32)

        state_dicts = load_safetensors_by_prefix(
            model_path,
            prefixes={"audio_encoder": "audio_encoder.encoder."},
            dtype=torch.float32,
        )
        enc_weights = state_dicts["audio_encoder"]
        result = encoder.encoder.load_state_dict(enc_weights, strict=False)
        if result.missing_keys:
            logger = __import__("logging").getLogger(__name__)
            logger.warning("Voxtral encoder missing keys: %s", result.missing_keys[:5])
    else:
        ae_config = Qwen3OmniMoeAudioEncoderConfig(**ae_cfg)
        encoder = AuTWrapper.from_config(ae_config, dtype=torch.float32)

        # Load weights from checkpoint.
        # The HF keys are audio_encoder.encoder.*, which after prefix stripping become
        # encoder.*. The AuTEncoder expects keys without the encoder. prefix.
        state_dicts = load_safetensors_by_prefix(
            model_path,
            prefixes={"audio_encoder": "audio_encoder.encoder."},
            dtype=torch.float32,
        )
        encoder.encoder.load_state_dict(state_dicts["audio_encoder"], strict=False)

    encoder.requires_grad_(False)

    _audio_encoder_cache = encoder
    return encoder
