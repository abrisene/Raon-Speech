# Speaker embedding extraction for voice conditioning.
# Uses SpeechBrain ECAPA-TDNN (PyTorch) for embedding extraction,
# then projects to thinker embedding space via the MLX projection weight.

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np

SPEAKER_EMBEDDING_PLACEHOLDER_ID = 151671


def extract_speaker_embedding(
    audio_path: str,
    projection_weight: mx.array,
    source_sample_rate: int = 24000,
    max_seconds: float = 10.0,
) -> mx.array:
    """Extract speaker embedding from a reference audio file.

    Uses SpeechBrain ECAPA-TDNN (PyTorch, CPU) to get a 192-dim speaker vector,
    then projects it to thinker embedding space (4096-dim) using the model's
    speaker_encoder.projection weight.

    Args:
        audio_path: Path to reference audio file.
        projection_weight: The speaker_encoder.projection.weight from the model.
            Shape: [4096, 192].
        source_sample_rate: Expected sample rate of the model (24000 for Raon).
        max_seconds: Maximum audio duration to use.

    Returns:
        Speaker embedding. Shape: [1, 1, 4096].
    """
    import torch
    import torchaudio
    import soundfile as sf

    # Load audio
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono

    audio_t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, samples]

    # Resample to model's expected rate if needed
    if sr != source_sample_rate:
        audio_t = torchaudio.functional.resample(audio_t, orig_freq=sr, new_freq=source_sample_rate)
        sr = source_sample_rate

    # Resample to 16kHz for ECAPA-TDNN
    audio_16k = torchaudio.functional.resample(audio_t, orig_freq=sr, new_freq=16000)

    # Truncate to max_seconds
    max_samples = int(max_seconds * 16000)
    if audio_16k.shape[1] > max_samples:
        audio_16k = audio_16k[:, :max_samples]

    # Extract ECAPA-TDNN embedding
    ecapa = _get_ecapa_model()
    with torch.no_grad():
        lengths = torch.tensor([audio_16k.shape[1]], dtype=torch.long)
        wav_lens = lengths.float() / float(audio_16k.shape[1])
        embedding = ecapa.encode_batch(audio_16k, wav_lens)  # [1, 1, 192]
        embedding = embedding.squeeze(1)  # [1, 192]

    # Convert to MLX and project
    embedding_mx = mx.array(embedding.numpy())  # [1, 192]
    # projection_weight shape: [4096, 192] — standard nn.Linear weight
    projected = embedding_mx @ projection_weight.T  # [1, 4096]
    return projected[:, None, :]  # [1, 1, 4096]


_ecapa_model = None


def _get_ecapa_model():
    """Lazy-load the ECAPA-TDNN model (cached)."""
    global _ecapa_model
    if _ecapa_model is not None:
        return _ecapa_model

    import torch

    # Patch for speechbrain compat
    if not hasattr(__builtins__, "__SPEECHBRAIN_PATCHED__"):
        import huggingface_hub as _hfhub
        _orig = _hfhub.hf_hub_download
        def _patched(*args, **kwargs):
            kwargs.pop("use_auth_token", None)
            return _orig(*args, **kwargs)
        _hfhub.hf_hub_download = _patched

        import torchaudio
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: []

    from speechbrain.inference.speaker import EncoderClassifier

    cache_dir = os.environ.get(
        "SPEECHBRAIN_ECAPA_SAVEDIR",
        os.path.expanduser("~/.cache/raon/speechbrain"),
    )
    model_id = "speechbrain/spkrec-ecapa-voxceleb"
    local_dir = os.path.join(cache_dir, model_id.replace("/", "_"))
    os.makedirs(cache_dir, exist_ok=True)

    from huggingface_hub import snapshot_download
    snapshot_download(model_id, local_dir=local_dir)

    _ecapa_model = EncoderClassifier.from_hparams(
        source=local_dir,
        savedir=local_dir,
        run_opts={"device": "cpu"},
    )
    for param in _ecapa_model.parameters():
        param.requires_grad = False

    return _ecapa_model
