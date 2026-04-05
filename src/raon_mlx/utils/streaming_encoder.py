# Streaming audio encoder bridge for duplex inference.
# Loads ONLY the audio encoder + input adaptor from the HF checkpoint (not the full 9B model).
# Runs on CPU to avoid competing with MLX for GPU memory.

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

SAMPLES_PER_FRAME = 1920  # 80ms at 24kHz


class StreamingAudioEncoder:
    """Lightweight PyTorch audio encoder for duplex streaming.

    Loads only the audio encoder + input adaptor weights (~500MB),
    NOT the full 9B model. Runs on CPU to preserve GPU memory for MLX.
    """

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._encoder = None  # VoxtralWrapper or AuTWrapper
        self._input_adaptor = None  # nn.Module
        self._streaming_state = None
        self._device = "cpu"
        self._loaded = False
        self._encoder_type = None  # "voxtral" or "aut"

    def load(self) -> None:
        if self._loaded:
            return

        import torch
        from raon.utils.misc import load_safetensors_by_prefix

        model_path = self._model_path
        logger.info("Loading audio encoder from %s...", model_path)

        # Read config to determine encoder type
        with open(Path(model_path) / "config.json") as f:
            cfg = json.load(f)
        ae_cfg = cfg["audio_encoder_config"]
        model_type = ae_cfg.get("model_type", "")

        if "voxtral" in model_type:
            self._load_voxtral(model_path, ae_cfg)
        else:
            self._load_aut(model_path, ae_cfg)

        # Load input_adaptor weights
        # Structure: proj.0 (encoder_dim→4096, GELU), proj.2 (4096→4096), post_norm (RMSNorm 4096)
        adaptor_weights = load_safetensors_by_prefix(
            model_path,
            prefixes={"adaptor": "input_adaptor."},
            dtype=torch.float32,
        )
        adaptor_state = adaptor_weights.get("adaptor", {})
        if adaptor_state:
            # Infer input dim from proj.0 weight shape [4096, encoder_dim]
            proj0_weight = adaptor_state.get("proj.0.weight")
            if proj0_weight is not None:
                encoder_dim = proj0_weight.shape[1]
                self._input_adaptor = _build_input_adaptor(encoder_dim, 4096, adaptor_state)
                self._input_adaptor.requires_grad_(False)
                logger.info("Input adaptor loaded (encoder_dim=%d → 4096)", encoder_dim)
            else:
                logger.warning("Input adaptor weights found but missing proj.0.weight")
        else:
            logger.warning("No input_adaptor weights found — output will be raw encoder embeddings")

        self._loaded = True
        logger.info("Audio encoder loaded on %s (type=%s)", self._device, self._encoder_type)

    def _load_voxtral(self, model_path: str, ae_cfg: dict) -> None:
        import torch
        from transformers.models.voxtral_realtime.configuration_voxtral_realtime import VoxtralRealtimeEncoderConfig
        from raon.modules.voxtral_wrapper import VoxtralWrapper
        from raon.utils.misc import load_safetensors_by_prefix

        vox_cfg = VoxtralRealtimeEncoderConfig(**ae_cfg)
        if not hasattr(vox_cfg, "rope_theta") or getattr(vox_cfg, "rope_theta", None) is None:
            vox_cfg.rope_theta = ae_cfg.get("rope_theta", 1000000.0)

        # Build wrapper with random weights
        wrapper = VoxtralWrapper.from_config(vox_cfg, dtype=torch.float32)

        # Load encoder weights with the correct prefix for this checkpoint.
        # Raon-SpeechChat-9B uses 'audio_encoder.encoder.' (not 'audio_tower.')
        enc_state = load_safetensors_by_prefix(
            model_path,
            prefixes={"enc": "audio_encoder.encoder."},
            dtype=torch.float32,
        )
        enc_weights = enc_state.get("enc", {})
        if not enc_weights:
            # Fallback: try audio_tower. prefix (standard Voxtral checkpoints)
            enc_state = load_safetensors_by_prefix(
                model_path,
                prefixes={"enc": "audio_tower."},
                dtype=torch.float32,
            )
            enc_weights = enc_state.get("enc", {})

        if enc_weights:
            result = wrapper.encoder.load_state_dict(enc_weights, strict=False)
            logger.info(
                "Voxtral encoder loaded: %d params, %d missing, %d unexpected",
                len(enc_weights), len(result.missing_keys), len(result.unexpected_keys),
            )
            if result.missing_keys:
                logger.warning("Missing encoder keys: %s", result.missing_keys[:5])
        else:
            logger.error("No encoder weights found! Audio encoding will be random.")

        wrapper.requires_grad_(False)
        self._encoder = wrapper
        self._encoder_type = "voxtral"

    def _load_aut(self, model_path: str, ae_cfg: dict) -> None:
        import torch
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
        from raon.modules.audio_encoder import AuTWrapper
        from raon.utils.misc import load_safetensors_by_prefix

        config = Qwen3OmniMoeAudioEncoderConfig(**ae_cfg)
        self._encoder = AuTWrapper.from_config(config, dtype=torch.float32)

        state_dicts = load_safetensors_by_prefix(
            model_path,
            prefixes={"audio_encoder": "audio_encoder.encoder."},
            dtype=torch.float32,
        )
        self._encoder.encoder.load_state_dict(state_dicts["audio_encoder"], strict=False)
        self._encoder.requires_grad_(False)
        self._encoder_type = "aut"

    def reset(self) -> None:
        if not self._loaded:
            self.load()
        self._streaming_state = self._encoder.init_streaming_state()
        logger.info("Streaming encoder state reset (type=%s).", self._encoder_type)

    def encode_frame(self, pcm: mx.array) -> mx.array:
        """Encode one audio frame to thinker-space embeddings.

        Args:
            pcm: Raw PCM audio [1, 1, 1920] as MLX array.

        Returns:
            Thinker-space embedding [1, num_frames, 4096] as MLX array.
        """
        import torch

        if self._streaming_state is None:
            self.reset()

        # Convert MLX -> numpy -> PyTorch (CPU)
        pcm_np = np.array(pcm, copy=False)
        if pcm_np.ndim == 3:
            audio_3d = torch.tensor(pcm_np, dtype=torch.float32)  # [1, 1, 1920]
        elif pcm_np.ndim == 2:
            audio_3d = torch.tensor(pcm_np, dtype=torch.float32).unsqueeze(1)  # [1, 1, 1920]
        else:
            audio_3d = torch.tensor(pcm_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.inference_mode():
            # Run encoder in streaming mode
            encoder_output = self._encoder(
                audio_3d,
                streaming_state=self._streaming_state,
            )

            embeds = encoder_output.embeds  # [1, num_frames, encoder_dim]
            self._streaming_state = encoder_output.streaming_state

            if embeds.shape[1] == 0:
                return mx.zeros((1, 0, 4096))

            # Apply input adaptor if available
            if self._input_adaptor is not None:
                embeds = self._input_adaptor(embeds)

        # Convert to MLX
        embeds_np = embeds.cpu().float().numpy()
        return mx.array(embeds_np)


def _build_input_adaptor(input_dim: int, output_dim: int, state_dict: dict) -> Any:
    """Build a minimal input adaptor module from weights.

    State dict keys: proj.0.weight [4096, input_dim], proj.2.weight [4096, 4096],
    post_norm.weight [4096].
    """
    import torch
    from torch import nn as torch_nn

    class InputAdaptor(torch_nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            # Use nn.Sequential with indices matching the state dict keys (proj.0, proj.2)
            self.proj = torch_nn.Sequential()
            self.proj.add_module("0", torch_nn.Linear(in_dim, out_dim, bias=False))
            self.proj.add_module("1", torch_nn.GELU())
            self.proj.add_module("2", torch_nn.Linear(out_dim, out_dim, bias=False))
            self.post_norm = torch_nn.RMSNorm(out_dim, eps=1e-6)

        def forward(self, x):
            x = self.proj(x)
            return self.post_norm(x)

    adaptor = InputAdaptor(input_dim, output_dim)
    adaptor.load_state_dict(state_dict, strict=False)
    return adaptor


# Singleton
_encoder_lock = threading.Lock()
_encoder_cache: dict[str, StreamingAudioEncoder] = {}


def get_streaming_encoder(model_path: str) -> StreamingAudioEncoder:
    """Get or create a cached streaming encoder for the given model."""
    with _encoder_lock:
        if model_path not in _encoder_cache:
            encoder = StreamingAudioEncoder(model_path)
            encoder.load()
            _encoder_cache[model_path] = encoder
        return _encoder_cache[model_path]
