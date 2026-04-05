# Generation loop for Raon-Speech MLX inference.
# Implements TTS: text tokens → thinker → talker → audio codes → Mimi decode → PCM.

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .raon import RaonMLX
from ..modules.kv_cache import KVCache

# Special token IDs (from raon/utils/special_tokens.py)
IM_END = 151645
AUDIO_START = 151669
AUDIO_END = 151670
AUDIO_OUTPUT_PLACEHOLDER = 151675
AUDIO_OUTPUT_PAD = 151677


def sample_token(logits: mx.array, temperature: float = 1.0, top_k: int = 20, top_p: float = 0.8) -> mx.array:
    """Sample a token from logits with temperature, top-k, and top-p filtering.

    Args:
        logits: Shape [batch, vocab_size].
        temperature: Sampling temperature. 0 = greedy.
        top_k: Keep only top-k logits. 0 = disabled.
        top_p: Nucleus sampling threshold. 1.0 = disabled.

    Returns:
        Sampled token IDs. Shape [batch, 1].
    """
    if temperature == 0:
        return logits.argmax(axis=-1, keepdims=True)

    logits = logits / temperature

    if top_k > 0:
        # Zero out everything below the top-k threshold
        top_k_vals = mx.sort(logits, axis=-1)[..., -top_k:]
        threshold = top_k_vals[..., :1]
        logits = mx.where(logits < threshold, mx.array(float("-inf")), logits)

    if top_p < 1.0:
        sorted_logits = mx.sort(logits, axis=-1)
        sorted_probs = mx.softmax(sorted_logits, axis=-1)
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
        # Remove tokens with cumulative probability above top_p
        sorted_mask = cumulative_probs < (1.0 - top_p)
        # Shift mask right to keep at least one token
        threshold_logit = mx.where(sorted_mask, mx.array(float("-inf")), sorted_logits).min(axis=-1, keepdims=True)
        logits = mx.where(logits < threshold_logit, mx.array(float("-inf")), logits)

    probs = mx.softmax(logits, axis=-1)
    return mx.random.categorical(mx.log(probs + 1e-10))[:, None]


def generate_audio_codes(
    model: RaonMLX,
    talker_hidden: mx.array,
    temperature: float = 1.2,
    top_k: int = 20,
) -> mx.array:
    """Generate 16 audio codebook codes from the talker's last hidden state.

    Flow:
    1. audio_lm_head(talker_hidden) → first codebook logits (2049 = 2048 codes + 1 EOS)
    2. Sample first code
    3. proj_code(talker_hidden) → hidden_embed [B, 1, 1024]
    4. codec_embedding(first_code) → code_embed [B, 1, 1024]
    5. Concatenate [hidden_embed, code_embed] → [B, 2, 1024]
    6. Code predictor autoregressively predicts codes 2-16

    Args:
        model: The full RaonMLX model.
        talker_hidden: Last talker hidden state. Shape: [B, 1, 2048].
        temperature: Sampling temperature for audio codes.
        top_k: Top-k filtering for audio codes.

    Returns:
        Audio codes. Shape: [B, 16]. Contains all 16 codebook codes.
        If first code == 2048 (EOS), remaining codes are 0.
    """
    B = talker_hidden.shape[0]
    codebook_size = 2048
    num_code_groups = 16

    # First codebook: from audio_lm_head
    first_logits = model.audio_lm_head(talker_hidden[:, -1])  # [B, 2049]
    if temperature > 0:
        first_code = sample_token(first_logits, temperature=temperature, top_k=top_k, top_p=1.0)
    else:
        first_code = first_logits.argmax(axis=-1, keepdims=True)  # [B, 1]

    # Check for audio end
    audio_end_mask = first_code[:, 0] == codebook_size
    safe_first_code = mx.minimum(first_code, codebook_size - 1)

    # Prepare code predictor input
    hidden_embed = model.proj_code(talker_hidden[:, -1:])  # [B, 1, 1024]
    code_embed = model.code_predictor.codec_embedding(safe_first_code)  # [B, 1, 1024]
    inputs_embeds = mx.concatenate([hidden_embed, code_embed], axis=1)  # [B, 2, 1024]

    # Autoregressive code prediction for remaining 15 codebooks
    cp_cache = model.code_predictor.make_cache()

    # Prefill with the 2-token input
    cp_out = model.code_predictor.model(inputs_embeds=inputs_embeds, cache=cp_cache)  # [B, 2, 1024]

    codes = [first_code[:, 0]]  # list of [B] arrays

    # First predicted code (codebook 2): use last hidden from prefill
    logits = cp_out[:, -1:] @ model.code_predictor.fused_lm_head[0].T  # [B, 1, 2048]
    logits = logits[:, 0]  # [B, 2048]
    if temperature > 0:
        next_code = sample_token(logits, temperature=temperature, top_k=top_k, top_p=1.0)[:, 0]
    else:
        next_code = logits.argmax(axis=-1)
    codes.append(next_code)

    # Remaining codebooks 3-16
    for i in range(1, num_code_groups - 1):
        # Embed the last predicted code with group offset
        code_input = next_code[:, None] + (i * codebook_size)
        code_emb = model.code_predictor.codec_embedding(code_input)  # [B, 1, 1024]

        # Run through code predictor (single step with cache)
        cp_out = model.code_predictor.model(inputs_embeds=code_emb, cache=cp_cache)  # [B, 1, 1024]

        # Get logits for next codebook
        logits = cp_out[:, -1:] @ model.code_predictor.fused_lm_head[i].T  # [B, 1, 2048]
        logits = logits[:, 0]  # [B, 2048]
        if temperature > 0:
            next_code = sample_token(logits, temperature=temperature, top_k=top_k, top_p=1.0)[:, 0]
        else:
            next_code = logits.argmax(axis=-1)
        codes.append(next_code)

    all_codes = mx.stack(codes, axis=1)  # [B, 16]

    # Zero out codes after audio end
    if audio_end_mask.any():
        all_codes = mx.where(audio_end_mask[:, None], mx.zeros_like(all_codes), all_codes)
        all_codes = mx.where(
            mx.broadcast_to(audio_end_mask[:, None], all_codes.shape),
            mx.concatenate([mx.full((B, 1), codebook_size, dtype=all_codes.dtype), mx.zeros((B, 15), dtype=all_codes.dtype)], axis=1),
            all_codes,
        )

    return all_codes


def tts_generate(
    model: RaonMLX,
    input_ids: mx.array,
    max_new_tokens: int = 512,
    temperature: float = 1.2,
    audio_temperature: float = 1.2,
    top_k: int = 20,
    top_p: float = 0.8,
) -> tuple[mx.array, int]:
    """Generate speech audio from text token IDs.

    Args:
        model: Full RaonMLX model with all weights loaded.
        input_ids: Tokenized text prompt. Shape: [1, seq_len].
        max_new_tokens: Maximum generation steps.
        temperature: Text sampling temperature.
        audio_temperature: Audio code sampling temperature.
        top_k: Top-k for sampling.
        top_p: Top-p for text sampling.

    Returns:
        Tuple of (audio_waveform, sample_rate).
        audio_waveform: Shape [1, num_samples].
    """
    B = input_ids.shape[0]
    assert B == 1, "Batch size must be 1 for now"

    codebook_size = 2048
    num_code_groups = 16

    # Create KV caches
    thinker_cache = model.thinker.make_cache()
    talker_cache = model.talker.make_cache()

    # Prefill: run input_ids through thinker
    thinker_out = model.thinker(input_ids=input_ids, cache=thinker_cache)
    text_logits = model.lm_head(thinker_out)

    # Project to talker space
    talker_input = model.thinker_to_talker_proj(thinker_out)
    talker_out = model.talker(talker_input, cache=talker_cache)

    # For TTS, force audio output: emit AUDIO_OUTPUT_PAD to start audio generation
    # Sample first text token to check, but we force audio mode
    sequences = input_ids
    audio_codes_list: list[mx.array] = []
    is_generating_audio = True

    # Generate first audio frame from prefill output
    first_codes = generate_audio_codes(model, talker_out[:, -1:], temperature=audio_temperature, top_k=top_k)
    audio_end = (first_codes[:, 0] == codebook_size).item()

    if not audio_end:
        audio_codes_list.append(first_codes)

    # Append AUDIO_OUTPUT_PLACEHOLDER token for next step
    next_token = mx.array([[AUDIO_OUTPUT_PLACEHOLDER]])
    sequences = mx.concatenate([sequences, next_token], axis=1)

    # Autoregressive loop
    for step in range(max_new_tokens - 1):
        if audio_end:
            break

        # Run single token through thinker
        thinker_out = model.thinker(input_ids=next_token, cache=thinker_cache)
        text_logits = model.lm_head(thinker_out)

        # Project to talker
        talker_input = model.thinker_to_talker_proj(thinker_out)
        talker_out = model.talker(talker_input, cache=talker_cache)

        if is_generating_audio:
            # Generate audio codes
            codes = generate_audio_codes(model, talker_out, temperature=audio_temperature, top_k=top_k)
            audio_end = (codes[:, 0] == codebook_size).item()

            if not audio_end:
                audio_codes_list.append(codes)
                next_token = mx.array([[AUDIO_OUTPUT_PLACEHOLDER]])
            else:
                next_token = mx.array([[AUDIO_END]])
        else:
            # Sample text token
            next_logits = text_logits[:, -1]
            next_logits_masked = next_logits.at[..., AUDIO_OUTPUT_PAD].add(float("-inf"))
            next_token = sample_token(next_logits_masked, temperature=temperature, top_k=top_k, top_p=top_p)

            if next_token.item() == IM_END:
                break
            if next_token.item() == AUDIO_START:
                is_generating_audio = True
                next_token = mx.array([[AUDIO_OUTPUT_PLACEHOLDER]])

        sequences = mx.concatenate([sequences, next_token], axis=1)

    if not audio_codes_list:
        # No audio generated
        return mx.zeros((1, 0)), 24000

    # Stack audio codes: [num_frames, 16] -> [1, 16, num_frames] for Mimi
    all_codes = mx.stack(audio_codes_list, axis=1)  # [1, num_frames, 16]
    all_codes = all_codes.transpose(0, 2, 1)  # [1, 16, num_frames] — Mimi expects [B, codebooks, frames]

    # Pad to 32 codebooks (Mimi uses 32, we only predict 16)
    # Remaining 16 codebooks are zeros (acoustic refinement codes)
    padding = mx.zeros((1, 16, all_codes.shape[2]), dtype=all_codes.dtype)
    all_codes_32 = mx.concatenate([all_codes, padding], axis=1)  # [1, 32, num_frames]

    # Decode through Mimi
    pcm = model.mimi.decode(all_codes_32)  # [1, 1, num_samples]
    pcm = pcm[:, 0]  # [1, num_samples]

    return pcm, 24000
