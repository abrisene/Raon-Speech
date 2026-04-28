"""Run PT's offline duplex on the same user.wav as MLX, for comparison."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from raon.pipeline import RaonPipeline


HF_PATH = os.environ.get("RAON_HF_PATH", "models/Raon-SpeechChat-9B")
USER_WAV = os.environ.get("RAON_USER_WAV", "output/duplex_smoke_after_encoder_fix/user.wav")
OUT_DIR = os.environ.get("RAON_OUT", "output/duplex_pt_reference")


def main() -> int:
    device = os.environ.get("RAON_DEVICE", "cpu")
    print(f"[run] PT pipeline {HF_PATH} on {device}", flush=True)
    dtype = os.environ.get("RAON_DTYPE", "float32")
    pipe = RaonPipeline(HF_PATH, device=device, dtype=dtype, attn_implementation="eager")

    audio, sr = sf.read(USER_WAV, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == 24000, f"expected 24kHz, got {sr}"
    audio_t = torch.from_numpy(audio[None, :]).to(device)

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    summary = pipe.duplex(audio_input=audio_t, output_dir=OUT_DIR)
    print(f"[run] summary: {summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
