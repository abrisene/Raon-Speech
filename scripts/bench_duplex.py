"""Deterministic micro-benchmark for duplex_step performance.

Runs the same audio through duplex_step N times with a fixed seed before each
run, so RTF is comparable across runs (otherwise the sampled assistant response
length dominates the RTF metric).

Usage:
    PYTHONPATH=src .venv/bin/python scripts/bench_duplex.py [--runs N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

from raon_mlx.pipeline import RaonMLXPipeline
from raon_mlx.models.duplex_generate import (
    SAMPLES_PER_FRAME, init_duplex_state, duplex_step,
)


def run_once(pipe: RaonMLXPipeline, audio: np.ndarray, seed: int = 42) -> tuple[float, float]:
    """One run with fixed seed. Returns (avg_frame_ms, rtf)."""
    mx.random.seed(seed)
    np.random.seed(seed)

    state = init_duplex_state(
        pipe.model, pipe.processor.tokenizer,
        hf_model_path=pipe.hf_model_path,
        system_prompt="You are engaging in real-time conversation.",
    )

    num_frames = (len(audio) - SAMPLES_PER_FRAME + 1) // SAMPLES_PER_FRAME
    frame_times: list[float] = []

    # Warmup 3 frames (cache is cold; first frame is huge)
    warmup = min(3, num_frames)
    for i in range(warmup):
        s = i * SAMPLES_PER_FRAME
        frame_pcm = audio[s:s + SAMPLES_PER_FRAME]
        state, _, _ = duplex_step(pipe.model, state, mx.array(frame_pcm[None, None, :]))

    # Timed run
    timed_frames = num_frames - warmup
    for i in range(warmup, num_frames):
        s = i * SAMPLES_PER_FRAME
        frame_pcm = audio[s:s + SAMPLES_PER_FRAME]
        ai = mx.array(frame_pcm[None, None, :])
        t0 = time.perf_counter()
        state, _, _ = duplex_step(pipe.model, state, ai)
        t1 = time.perf_counter()
        frame_times.append((t1 - t0) * 1000)

    avg_ms = float(np.mean(frame_times))
    user_secs = timed_frames * SAMPLES_PER_FRAME / 24000
    decode_secs = sum(frame_times) / 1000
    rtf = decode_secs / user_secs
    return avg_ms, rtf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--audio", default="/tmp/user_3s.wav")
    parser.add_argument("--model", default="models/Raon-SpeechChat-9B-mlx-8bit")
    parser.add_argument("--hf", default="models/Raon-SpeechChat-9B")
    args = parser.parse_args()

    pipe = RaonMLXPipeline(args.model, hf_model_path=args.hf, quant="8bit")

    audio, sr = sf.read(args.audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 24000:
        ratio = 24000 / sr
        new_len = int(len(audio) * ratio)
        idx = np.arange(new_len) / ratio
        lo = np.floor(idx).astype(int)
        hi = np.minimum(lo + 1, len(audio) - 1)
        frac = idx - lo
        audio = audio[lo] * (1 - frac) + audio[hi] * frac

    results = []
    for run_i in range(args.runs):
        # Same seed each run so generated audio (and thus phase distribution) matches
        avg_ms, rtf = run_once(pipe, audio, seed=42)
        results.append((avg_ms, rtf))
        print(f"  run {run_i + 1}: avg_frame={avg_ms:6.2f} ms, RTF={rtf:.4f}")

    avg_ms_all = np.mean([r[0] for r in results])
    rtf_all = np.mean([r[1] for r in results])
    print(f"\nMEAN over {args.runs} runs: avg_frame={avg_ms_all:6.2f} ms, RTF={rtf_all:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
