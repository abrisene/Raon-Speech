"""Probe: feed silence codes through MLX Mimi decode (batch) and decode_step (streaming),
compare to raw silence."""

from __future__ import annotations

import os
import sys
import numpy as np
import mlx.core as mx

from raon_mlx.pipeline import RaonMLXPipeline


MODEL = os.environ.get("RAON_MODEL_PATH", "models/Raon-SpeechChat-9B-mlx-8bit")
HF = os.environ.get("RAON_HF_PATH", "models/Raon-SpeechChat-9B")


def main() -> int:
    print("[mimi] loading", flush=True)
    pipe = RaonMLXPipeline(MODEL, hf_model_path=HF, quant="hybrid")
    mimi = pipe.model.mimi

    # Get silence codes (encode 0 PCM)
    silence_pcm = mx.zeros((1, 1, 1920))
    sil_codes = mimi.encode(silence_pcm)  # [1, codebooks, 1]
    sil_codes_16 = sil_codes[:, :16, :]   # [1, 16, 1]
    print(f"[mimi] silence codes shape: {sil_codes_16.shape}", flush=True)
    print(f"[mimi] silence code values (first 5 codebooks): {sil_codes_16[0, :5, 0].tolist()}", flush=True)

    # Batch decode 4 frames worth of silence codes
    n_frames = 8
    sil_repeat = mx.broadcast_to(sil_codes_16, (1, 16, n_frames))
    out_batch = mimi.decode(sil_repeat)  # [1, 1, n_frames * 1920]
    out_batch_np = np.array(out_batch[0, 0]).astype(np.float32)
    print(f"[mimi] batch decode {n_frames} frames: rms={float(np.sqrt(np.mean(out_batch_np**2))):.6f} max_abs={float(np.max(np.abs(out_batch_np))):.6f}", flush=True)

    # Streaming decode same frames one by one
    mimi.reset_all()
    streaming_chunks = []
    for i in range(n_frames):
        chunk = mimi.decode_step(sil_codes_16)  # [1, 1, 1920]
        chunk_np = np.array(chunk[0, 0]).astype(np.float32)
        streaming_chunks.append(chunk_np)
        rms = float(np.sqrt(np.mean(chunk_np**2)))
        max_abs = float(np.max(np.abs(chunk_np)))
        print(f"[mimi] streaming frame {i}: rms={rms:.6f} max_abs={max_abs:.6f}", flush=True)
    out_stream = np.concatenate(streaming_chunks)
    print(f"[mimi] streaming full {n_frames} frames: rms={float(np.sqrt(np.mean(out_stream**2))):.6f} max_abs={float(np.max(np.abs(out_stream))):.6f}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
