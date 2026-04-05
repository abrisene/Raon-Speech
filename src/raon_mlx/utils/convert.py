#!/usr/bin/env python3
"""Convert Raon-Speech HF checkpoint to quantized MLX format.

Usage:
    python -m raon_mlx.utils.convert models/Raon-Speech-9B --output models/Raon-Speech-9B-mlx-4bit
    python -m raon_mlx.utils.convert models/Raon-Speech-9B --output models/Raon-Speech-9B-mlx-hybrid --quant hybrid
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from ..models.raon import RaonMLX


def convert(
    model_path: str,
    output_path: str,
    quant: str = "hybrid",
) -> None:
    """Convert and quantize a Raon-Speech model to MLX format.

    Args:
        model_path: Path to HF checkpoint directory.
        output_path: Path to save the quantized MLX model.
        quant: Quantization mode: "none", "4bit", "8bit", "hybrid".
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}...")
    t0 = time.time()
    model = RaonMLX()
    model.load_weights_from_raon(model_path)
    t1 = time.time()
    print(f"Loaded in {t1 - t0:.1f}s")

    # Quantize
    if quant == "4bit":
        nn.quantize(model.thinker, bits=4, group_size=64)
        nn.quantize(model.talker, bits=4, group_size=64)
        nn.quantize(model.code_predictor.model, bits=4, group_size=64)
        print("Quantized: all 4-bit")
    elif quant == "8bit":
        nn.quantize(model.thinker, bits=8, group_size=64)
        nn.quantize(model.talker, bits=8, group_size=64)
        nn.quantize(model.code_predictor.model, bits=8, group_size=64)
        print("Quantized: all 8-bit")
    elif quant == "hybrid":
        nn.quantize(model.thinker, bits=4, group_size=64)
        nn.quantize(model.talker, bits=8, group_size=64)
        nn.quantize(model.code_predictor.model, bits=8, group_size=64)
        print("Quantized: hybrid (thinker=4bit, talker+cp=8bit)")
    elif quant == "none":
        print("No quantization")
    else:
        raise ValueError(f"Unknown quant mode: {quant}")

    # Save all weights
    print(f"Saving to {output_dir}...")
    t0 = time.time()
    flat_weights = dict(nn.utils.tree_flatten(model.parameters()))
    mx.save_safetensors(str(output_dir / "model.safetensors"), flat_weights)
    t1 = time.time()
    print(f"Saved {len(flat_weights)} parameters in {t1 - t0:.1f}s")

    # Save config
    config = {
        "quant": quant,
        "source_model": model_path,
        "thinker_bits": 4 if quant in ("4bit", "hybrid") else (8 if quant == "8bit" else 16),
        "talker_bits": 4 if quant == "4bit" else (8 if quant in ("8bit", "hybrid") else 16),
        "cp_bits": 4 if quant == "4bit" else (8 if quant in ("8bit", "hybrid") else 16),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Calculate size
    total_bytes = sum(v.nbytes for v in flat_weights.values())
    print(f"Total size: {total_bytes / 1e9:.2f} GB")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Convert Raon-Speech to quantized MLX format")
    parser.add_argument("model_path", help="Path to HF checkpoint directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory for MLX model")
    parser.add_argument("--quant", choices=["none", "4bit", "8bit", "hybrid"], default="hybrid",
                        help="Quantization mode (default: hybrid)")
    args = parser.parse_args()
    convert(args.model_path, args.output, args.quant)


if __name__ == "__main__":
    main()
