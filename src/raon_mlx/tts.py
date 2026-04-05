#!/usr/bin/env python3
"""Quick TTS CLI for Raon-Speech MLX.

Usage:
    python -m raon_mlx.tts "Hello world" --output output.wav
    python -m raon_mlx.tts "Hello world" --model models/Raon-Speech-9B --quant hybrid
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import warnings

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import soundfile as sf

from .models.raon import RaonMLX
from .models.generate import tts_generate


def _is_mlx_model(path: str) -> bool:
    """Check if path contains a pre-converted MLX model."""
    return os.path.exists(os.path.join(path, "model.safetensors")) and os.path.exists(os.path.join(path, "config.json")) and not os.path.exists(os.path.join(path, "model.safetensors.index.json"))


def _find_tokenizer_path(model_path: str) -> str:
    """Find the HF tokenizer — either in the model dir or the source model."""
    if os.path.exists(os.path.join(model_path, "tokenizer.json")):
        return model_path
    # Check config for source model
    import json
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        source = cfg.get("source_model")
        if source and os.path.exists(os.path.join(source, "tokenizer.json")):
            return source
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Raon-Speech MLX TTS")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--model", default="models/Raon-Speech-9B", help="Model path (HF or pre-converted MLX)")
    parser.add_argument("--output", "-o", default="output/mlx_tts.wav", help="Output wav path")
    parser.add_argument("--quant", choices=["none", "4bit", "8bit", "hybrid"], default="hybrid",
                        help="Quantization (ignored for pre-converted MLX models)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=1.2, help="Audio sampling temperature")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling")
    args = parser.parse_args()

    # Suppress noisy tokenizer warnings at all levels
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    # Tokenize (suppress Rust tokenizer warnings via OS-level fd redirect)
    tokenizer_path = _find_tokenizer_path(args.model)
    import sys
    from raon.utils.processor import RaonProcessor, get_default_tts_prompt

    processor = RaonProcessor.from_pretrained(tokenizer_path)
    prompt = get_default_tts_prompt()
    messages = [{"role": "user", "content": f"{prompt}:\n{args.text}"}]
    inputs = processor(messages, add_generation_prompt=True, force_audio_output=True, device="cpu")
    input_ids = mx.array(inputs["input_ids"].numpy())
    logging.disable(logging.NOTSET)
    print(f"Input: {input_ids.shape[1]} tokens")

    # Load model
    is_mlx = _is_mlx_model(args.model)
    t0 = time.time()
    model = RaonMLX()
    if is_mlx:
        print(f"Loading pre-converted MLX model from {args.model}...")
        model.load_mlx_weights(args.model)
        print("Loaded (pre-quantized)")
    else:
        print(f"Loading from HF checkpoint {args.model}...")
        model.load_weights_from_raon(args.model)
        # Quantize on the fly
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
    t1 = time.time()
    print(f"Model ready in {t1 - t0:.1f}s")

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
