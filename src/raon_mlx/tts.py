#!/usr/bin/env python3
"""Quick TTS CLI for Raon-Speech MLX.

Usage:
    python -m raon_mlx.tts "Hello world" --output output.wav
    python -m raon_mlx.tts "Hello world" --model models/Raon-Speech-9B --quant hybrid
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import soundfile as sf

from .models.raon import RaonMLX
from .models.generate import tts_generate


def main():
    parser = argparse.ArgumentParser(description="Raon-Speech MLX TTS")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--model", default="models/Raon-Speech-9B", help="Model path")
    parser.add_argument("--output", "-o", default="output/mlx_tts.wav", help="Output wav path")
    parser.add_argument("--quant", choices=["none", "4bit", "8bit", "hybrid"], default="hybrid",
                        help="Quantization: none, 4bit (all), 8bit (all), hybrid (4bit thinker + 8bit rest)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=1.2, help="Audio sampling temperature")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling")
    args = parser.parse_args()

    # Tokenize using PyTorch processor (shares tokenizer with MLX model)
    from raon.utils.processor import RaonProcessor, get_default_tts_prompt

    print(f"Loading tokenizer from {args.model}...")
    processor = RaonProcessor.from_pretrained(args.model)
    prompt = get_default_tts_prompt()
    messages = [{"role": "user", "content": f"{prompt}:\n{args.text}"}]
    inputs = processor(messages, add_generation_prompt=True, force_audio_output=True, device="cpu")
    input_ids = mx.array(inputs["input_ids"].numpy())
    print(f"Input: {input_ids.shape[1]} tokens")

    # Load model
    print(f"Loading MLX model from {args.model}...")
    t0 = time.time()
    model = RaonMLX()
    model.load_weights_from_raon(args.model)
    t1 = time.time()
    print(f"Model loaded in {t1 - t0:.1f}s")

    # Quantize
    if args.quant == "4bit":
        nn.quantize(model.thinker, bits=4, group_size=64)
        nn.quantize(model.talker, bits=4, group_size=64)
        nn.quantize(model.code_predictor.model, bits=4, group_size=64)
        print("Quantized: all 4-bit")
    elif args.quant == "8bit":
        nn.quantize(model.thinker, bits=8, group_size=64)
        nn.quantize(model.talker, bits=8, group_size=64)
        nn.quantize(model.code_predictor.model, bits=8, group_size=64)
        print("Quantized: all 8-bit")
    elif args.quant == "hybrid":
        nn.quantize(model.thinker, bits=4, group_size=64)
        nn.quantize(model.talker, bits=8, group_size=64)
        nn.quantize(model.code_predictor.model, bits=8, group_size=64)
        print("Quantized: hybrid (thinker=4bit, talker+cp=8bit)")

    # Generate
    print(f'\nGenerating: "{args.text}"')
    t0 = time.time()
    pcm, sr = tts_generate(
        model, input_ids,
        max_new_tokens=args.max_tokens,
        audio_temperature=args.temperature,
        top_k=args.top_k,
    )
    # Force materialization
    if pcm.shape[1] > 0:
        _ = pcm[0, 0].item()
    t1 = time.time()

    if pcm.shape[1] == 0:
        print("No audio generated.")
        return

    duration = pcm.shape[1] / sr
    rtf = (t1 - t0) / duration
    print(f"Generated {duration:.1f}s audio in {t1 - t0:.1f}s (RTF {rtf:.2f}, {1/rtf:.1f}x real-time)")

    sf.write(args.output, np.array(pcm[0]), sr)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
