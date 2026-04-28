"""Probe C2: isolate whether the encoder OUTPUT saturation is caused by
streaming-state accumulation, or by the input itself.

Compares:
  (a) input PCM frame-to-frame cosine (raw audio variation)
  (b) streaming encoder output cosine (current path)
  (c) fresh encoder per frame: reset() before every encode_frame call
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

import mlx.core as mx  # noqa: E402

from raon_mlx.utils.streaming_encoder import get_streaming_encoder, StreamingAudioEncoder

HF_PATH = os.environ.get("RAON_HF_PATH", "models/Raon-SpeechChat-9B")
USER_WAV = os.environ.get(
    "RAON_USER_WAV", "output/duplex_smoke_after_encoder_fix/user.wav"
)
SAMPLES_PER_FRAME = 1920
N_FRAMES = int(os.environ.get("RAON_PROBE_FRAMES", "40"))


def cos(a: mx.array, b: mx.array) -> float:
    af = a.reshape(-1).astype(mx.float32)
    bf = b.reshape(-1).astype(mx.float32)
    denom = mx.sqrt(mx.sum(af * af) * mx.sum(bf * bf)) + 1e-12
    return float((mx.sum(af * bf) / denom).item())


def cos_np(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1).astype(np.float32)
    bf = b.reshape(-1).astype(np.float32)
    denom = float(np.sqrt((af * af).sum() * (bf * bf).sum())) + 1e-12
    return float((af * bf).sum() / denom)


def main() -> int:
    print(f"[probe_c2] loading streaming encoder for {HF_PATH}", flush=True)
    enc_streaming = get_streaming_encoder(HF_PATH)
    enc_streaming.reset()

    print(f"[probe_c2] loading fresh encoder", flush=True)
    enc_fresh = StreamingAudioEncoder(HF_PATH)
    enc_fresh.load()

    audio, sr = sf.read(USER_WAV, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == 24000

    n = min(N_FRAMES, (len(audio) - SAMPLES_PER_FRAME + 1) // SAMPLES_PER_FRAME)
    print(f"[probe_c2] running {n} frames", flush=True)

    prev_pcm = None
    prev_stream = None
    prev_fresh = None

    for i in range(n):
        s = i * SAMPLES_PER_FRAME
        frame_pcm_np = audio[s:s + SAMPLES_PER_FRAME]
        frame_pcm_mx = mx.array(frame_pcm_np[None, None, :])

        # (a) raw input cosine
        pcm_cos = cos_np(frame_pcm_np, prev_pcm) if prev_pcm is not None else float("nan")
        pcm_rms = float(np.sqrt((frame_pcm_np ** 2).mean()))

        # (b) streaming encoder
        out_stream = enc_streaming.encode_frame(frame_pcm_mx)
        if out_stream.shape[1] == 0:
            stream_cos = float("nan")
            stream_norm = float("nan")
        else:
            slot = out_stream[:, -1:, :]
            stream_norm = float(mx.sqrt(mx.sum(slot.astype(mx.float32) ** 2)).item())
            stream_cos = cos(slot, prev_stream) if prev_stream is not None else float("nan")
            prev_stream = slot

        # (c) fresh encoder: reset, feed N most-recent frames? For a fair comparison,
        # we feed only the current frame, which means there is no STFT context for it.
        # Instead, replay all frames up to i with a fresh encoder.
        enc_fresh.reset()
        out_fresh = None
        for j in range(i + 1):
            s_j = j * SAMPLES_PER_FRAME
            f_j = audio[s_j:s_j + SAMPLES_PER_FRAME]
            out_j = enc_fresh.encode_frame(mx.array(f_j[None, None, :]))
            if out_j.shape[1] > 0:
                out_fresh = out_j
        if out_fresh is None or out_fresh.shape[1] == 0:
            fresh_cos = float("nan")
            fresh_norm = float("nan")
        else:
            slot_f = out_fresh[:, -1:, :]
            fresh_norm = float(mx.sqrt(mx.sum(slot_f.astype(mx.float32) ** 2)).item())
            fresh_cos = cos(slot_f, prev_fresh) if prev_fresh is not None else float("nan")
            prev_fresh = slot_f

        print(
            f"[C2] f={i+1:3d} pcm_rms={pcm_rms:.4f} pcm_cos={pcm_cos:+.3f}  "
            f"stream_norm={stream_norm:.3f} stream_cos={stream_cos:+.3f}  "
            f"fresh_norm={fresh_norm:.3f} fresh_cos={fresh_cos:+.3f}",
            flush=True,
        )
        prev_pcm = frame_pcm_np

    print("[probe_c2] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
