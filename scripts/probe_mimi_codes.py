"""Probe: take a known set of audio codes and decode via streaming vs batch
Mimi to see if they diverge."""

from __future__ import annotations

import os
import sys
import numpy as np
import mlx.core as mx

from raon_mlx.pipeline import RaonMLXPipeline


def main() -> int:
    pipe = RaonMLXPipeline(
        os.environ.get("RAON_MODEL_PATH", "models/Raon-SpeechChat-9B-mlx-8bit"),
        hf_model_path=os.environ.get("RAON_HF_PATH", "models/Raon-SpeechChat-9B"),
        quant="hybrid",
    )
    mimi = pipe.model.mimi

    # First, get silence codes for reference
    silence_pcm = mx.zeros((1, 1, 1920))
    sil_codes = mimi.encode(silence_pcm)[:, :16, :]  # [1, 16, 1]
    print(f"silence codes: {sil_codes[0, :, 0].tolist()}", flush=True)

    # Use the codes from the duplex run frame 0 (from probe output)
    # gen_codes=[784, 1418, 1669, 1557, 1800, ...] (only first 5 shown)
    # Use a synthetic non-silence code set: random codes
    rng = np.random.default_rng(42)
    fake_codes = rng.integers(0, 2048, size=(16,))
    fake_codes_mx = mx.array(fake_codes[None, :, None])

    print(f"\n--- batch decode ---", flush=True)
    mimi.reset_all()
    out_batch_sil = mimi.decode(sil_codes)
    out_batch_sil_np = np.array(out_batch_sil[0, 0]).astype(np.float32)
    print(f"silence codes batch: rms={float(np.sqrt(np.mean(out_batch_sil_np**2))):.6f} max_abs={float(np.max(np.abs(out_batch_sil_np))):.6f}", flush=True)

    mimi.reset_all()
    out_batch_fake = mimi.decode(fake_codes_mx)
    out_batch_fake_np = np.array(out_batch_fake[0, 0]).astype(np.float32)
    print(f"fake codes batch: rms={float(np.sqrt(np.mean(out_batch_fake_np**2))):.6f} max_abs={float(np.max(np.abs(out_batch_fake_np))):.6f}", flush=True)

    print(f"\n--- streaming decode (after init silence) ---", flush=True)
    mimi.reset_all()
    # Init: feed 1 silence to prime
    _ = mimi.decode_step(sil_codes)
    # Now decode the test code
    out_stream_sil = mimi.decode_step(sil_codes)
    out_stream_sil_np = np.array(out_stream_sil[0, 0]).astype(np.float32)
    print(f"silence codes streaming: rms={float(np.sqrt(np.mean(out_stream_sil_np**2))):.6f} max_abs={float(np.max(np.abs(out_stream_sil_np))):.6f}", flush=True)

    mimi.reset_all()
    _ = mimi.decode_step(sil_codes)
    out_stream_fake = mimi.decode_step(fake_codes_mx)
    out_stream_fake_np = np.array(out_stream_fake[0, 0]).astype(np.float32)
    print(f"fake codes streaming: rms={float(np.sqrt(np.mean(out_stream_fake_np**2))):.6f} max_abs={float(np.max(np.abs(out_stream_fake_np))):.6f}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
