# Pure MLX port of the Voxtral Realtime audio encoder used by Raon-SpeechChat.
# Vendored and adapted from Blaizzy/mlx-audio (MIT licensed):
#   mlx_audio/stt/models/voxtral_realtime/{encoder.py, streaming.py, audio.py}
#
# Adaptations vs upstream:
#   - causal STFT path matching Raon's _extract_features_causal_streaming
#     (running-max log normalization, n_fft//2 left-only pad on first chunk,
#      drop-last-frame, leftover-waveform stft cache)
#   - encoder strips audio_language_projection_* (Raon uses its own input_adaptor)
#   - HF-named weight mapping (audio_encoder.encoder.layers.{i}.self_attn.{q,k,v,o}_proj
#     -> encoder.transformer_layers.{i}.attention.w{q,k,v,o})

from .encoder import AudioEncoder, EncoderConfig
from .streaming import (
    StreamingCausalConv1d,
    StreamingConvStem,
    StreamingEncoder,
    StreamingFrameStack,
    StreamingMelCausal,
)
from .weights import load_encoder_weights, load_input_adaptor_weights

__all__ = [
    "AudioEncoder",
    "EncoderConfig",
    "StreamingCausalConv1d",
    "StreamingConvStem",
    "StreamingEncoder",
    "StreamingFrameStack",
    "StreamingMelCausal",
    "load_encoder_weights",
    "load_input_adaptor_weights",
]
