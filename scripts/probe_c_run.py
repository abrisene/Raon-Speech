"""Probe C harness: run ~30 duplex frames on a user.wav and emit per-frame
diagnostics about audio_input_embeds + placeholder slot variation.

Run:
    RAON_PROBE_C=1 uv run python scripts/probe_c_run.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

os.environ.setdefault("RAON_PROBE_C", "1")

import mlx.core as mx  # noqa: E402

from raon_mlx.pipeline import RaonMLXPipeline  # noqa: E402
from raon_mlx.models.duplex_generate import (  # noqa: E402
    SAMPLES_PER_FRAME, init_duplex_state, duplex_step,
)


MODEL_PATH = os.environ.get("RAON_MODEL_PATH", "models/Raon-SpeechChat-9B-mlx-8bit")
HF_PATH = os.environ.get("RAON_HF_PATH", "models/Raon-SpeechChat-9B")
QUANT = os.environ.get("RAON_QUANT", "hybrid")
USER_WAV = os.environ.get(
    "RAON_USER_WAV", "output/duplex_smoke_after_encoder_fix/user.wav"
)
N_FRAMES = int(os.environ.get("RAON_PROBE_FRAMES", "40"))


def main() -> int:
    print(f"[probe_c] loading pipeline from {MODEL_PATH} quant={QUANT}", flush=True)
    pipe = RaonMLXPipeline(MODEL_PATH, hf_model_path=HF_PATH, quant=QUANT)

    audio, sr = sf.read(USER_WAV, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == 24000, f"expected 24kHz, got {sr}"
    print(f"[probe_c] user wav: {len(audio)} samples, {len(audio)/sr:.2f}s", flush=True)

    state = init_duplex_state(
        pipe.model,
        pipe.processor.tokenizer,
        hf_model_path=pipe.hf_model_path,
        system_prompt="You are engaging in real-time conversation.",
        speak_first=False,
    )

    n = min(N_FRAMES, (len(audio) - SAMPLES_PER_FRAME + 1) // SAMPLES_PER_FRAME)
    print(f"[probe_c] running {n} frames", flush=True)
    for i in range(n):
        s = i * SAMPLES_PER_FRAME
        frame_pcm = audio[s:s + SAMPLES_PER_FRAME]
        audio_input = mx.array(frame_pcm[None, None, :])
        state, _out, _txt = duplex_step(pipe.model, state, audio_input)

    print("[probe_c] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
