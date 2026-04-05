#!/usr/bin/env python3
"""STT CLI for Raon-Speech MLX.

Usage:
    python -m raon_mlx.stt input.wav --model models/Raon-Speech-9B-mlx-hybrid
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import warnings

import mlx.core as mx
import mlx.nn as nn


def main():
    parser = argparse.ArgumentParser(description="Raon-Speech MLX STT")
    parser.add_argument("audio", help="Input audio file")
    parser.add_argument("--model", default="models/Raon-Speech-9B-mlx-hybrid", help="MLX or HF model path")
    parser.add_argument("--hf-model", default="models/Raon-Speech-9B",
                        help="HF checkpoint path (for audio encoder weights and tokenizer)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max output tokens")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    # Load tokenizer (need the processor for STT prompt formatting)
    _old = os.dup(2)
    _null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_null, 2)

    from raon.utils.processor import RaonProcessor, get_default_stt_prompt

    # Find tokenizer
    hf_path = args.hf_model
    if not os.path.exists(os.path.join(hf_path, "tokenizer.json")):
        import json
        config_path = os.path.join(args.model, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            src = cfg.get("source_model", "")
            if src and os.path.exists(os.path.join(src, "tokenizer.json")):
                hf_path = src

    processor = RaonProcessor.from_pretrained(hf_path)

    # Format STT messages: audio placeholder + transcription prompt
    stt_prompt = get_default_stt_prompt()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": args.audio},
                {"type": "text", "text": stt_prompt},
            ],
        }
    ]

    inputs = processor(
        messages,
        add_generation_prompt=True,
        force_audio_output=False,
        device="cpu",
        max_audio_chunk_length=192000,
    )

    input_ids = mx.array(inputs["input_ids"].numpy())

    import time as _t
    _t.sleep(0.05)
    os.dup2(_old, 2)
    os.close(_null)
    os.close(_old)
    logging.disable(logging.NOTSET)

    print(f"Input: {input_ids.shape[1]} tokens")
    print(f"Audio: {args.audio}")

    # Check if there are audio input embeddings from the processor
    audio_input = inputs.get("audio_input")
    audio_input_lengths = inputs.get("audio_input_lengths")

    # Load MLX model
    from .models.raon import RaonMLX

    t0 = time.time()
    model = RaonMLX()
    # Always load from HF for STT (pre-converted models may have embedding quantization issues)
    model.load_weights_from_raon(hf_path)
    nn.quantize(model.thinker, bits=4, group_size=64)
    nn.quantize(model.talker, bits=8, group_size=64)
    nn.quantize(model.code_predictor.model, bits=8, group_size=64)
    print("Loaded + quantized (hybrid)")

    # Encode audio using PyTorch audio encoder with the processor's pre-loaded audio
    print("Encoding audio...")
    t_enc_start = time.time()

    from .utils.audio_encoder import encode_audio_tensor

    audio_embeds, audio_mask = encode_audio_tensor(
        audio_tensor=audio_input,
        audio_lengths=audio_input_lengths,
        model_path=hf_path,
        input_adaptor_proj_0=model.input_adaptor.proj_0.weight,
        input_adaptor_proj_2=model.input_adaptor.proj_2.weight,
        input_adaptor_post_norm_weight=model.input_adaptor.post_norm.weight,
    )
    t_enc_end = time.time()
    print(f"Audio encoded: {audio_embeds.shape[1]} frames in {t_enc_end - t_enc_start:.1f}s")

    # Generate transcription
    from .models.generate import stt_generate

    print("Transcribing...")
    t0 = time.time()
    token_ids = stt_generate(
        model,
        input_ids=input_ids,
        audio_embeds=audio_embeds,
        audio_embeds_mask=audio_mask,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    t1 = time.time()

    # Decode tokens
    text = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"\nTranscription ({t1 - t0:.1f}s):")
    print(f"  {text}")


if __name__ == "__main__":
    main()
