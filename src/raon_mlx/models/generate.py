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
    suppress_eos: bool = False,
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
    if suppress_eos and first_logits.shape[-1] > codebook_size:
        # Suppress AUDIO_END token (index 2048)
        first_logits = first_logits.at[..., codebook_size].add(-1e9)
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


def stt_generate(
    model: RaonMLX,
    input_ids: mx.array,
    audio_embeds: mx.array,
    audio_embeds_mask: mx.array,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_k: int = 20,
    top_p: float = 0.8,
) -> list[int]:
    """Generate text transcription from audio input (STT).

    The audio embeddings are inserted at the AUDIO_INPUT_PLACEHOLDER positions
    in the input sequence.

    Args:
        model: Full RaonMLX model with all weights loaded.
        input_ids: Tokenized STT prompt with audio placeholders. Shape: [1, seq_len].
        audio_embeds: Encoded audio embeddings from the audio encoder + input adaptor.
            Shape: [1, num_frames, 4096].
        audio_embeds_mask: Valid frame mask. Shape: [1, num_frames].
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (lower = more deterministic for STT).
        top_k: Top-k filtering.
        top_p: Top-p filtering.

    Returns:
        List of generated token IDs (decode with tokenizer to get text).
    """
    AUDIO_INPUT_PLACEHOLDER = 151676  # <|audio_input_placeholder|>

    # Build input embeddings, replacing audio placeholders with audio embeds
    AUDIO_INPUT_PLACEHOLDER = 151676

    inputs_embeds = model.thinker.embed_tokens(input_ids)  # [1, seq_len, 4096]

    # Find placeholder range (they're consecutive)
    input_ids_list = input_ids[0].tolist()
    first_ph = None
    last_ph = None
    for i, tid in enumerate(input_ids_list):
        if tid == AUDIO_INPUT_PLACEHOLDER:
            if first_ph is None:
                first_ph = i
            last_ph = i

    if first_ph is not None and audio_embeds.shape[1] > 0:
        n_placeholders = last_ph - first_ph + 1
        n_audio = audio_embeds.shape[1]
        n_replace = min(n_placeholders, n_audio)

        # Build new embeddings: [before_placeholders | audio_embeds | after_placeholders]
        audio_embeds_cast = audio_embeds[:, :n_replace].astype(inputs_embeds.dtype)
        before = inputs_embeds[:, :first_ph, :]
        after = inputs_embeds[:, first_ph + n_replace:, :]
        inputs_embeds = mx.concatenate([before, audio_embeds_cast, after], axis=1)

    # Create KV caches
    thinker_cache = model.thinker.make_cache()

    # Prefill
    thinker_normed, _ = model.thinker(inputs_embeds=inputs_embeds, cache=thinker_cache)
    text_logits = model.lm_head(thinker_normed)

    # Autoregressive text generation
    generated_ids = []
    for step in range(max_new_tokens):
        next_logits = text_logits[:, -1]

        # Suppress audio tokens during STT
        next_logits = next_logits.at[:, AUDIO_OUTPUT_PAD].add(-1e9)

        if temperature > 0:
            next_token = sample_token(next_logits, temperature=temperature, top_k=top_k, top_p=top_p)
        else:
            next_token = next_logits.argmax(axis=-1, keepdims=True)

        token_id = next_token[0, 0].item()

        if token_id == IM_END:
            break

        generated_ids.append(token_id)

        # Feed back through thinker
        thinker_normed, _ = model.thinker(input_ids=next_token, cache=thinker_cache)
        text_logits = model.lm_head(thinker_normed)

    return generated_ids


def _get_audio_output_embed(model: RaonMLX, codes: mx.array) -> mx.array:
    """Convert generated audio codes to thinker input embeddings.

    This is the critical feedback loop: the thinker needs to know what audio was
    generated so it can condition the next frame. The codes are decoded through
    Mimi's VQ (not full waveform decode) to get latent features, then projected
    to thinker embedding space via the output_adaptor.

    Args:
        model: The full RaonMLX model.
        codes: Audio codes for one frame. Shape: [B, 16].

    Returns:
        Thinker-space embedding. Shape: [B, 1, 4096].
    """
    # VQ decode with only 16 codebooks (don't pad to 32)
    B = codes.shape[0]
    codes_16 = codes[:, :, None]  # [B, 16, 1] — single frame

    # VQ decode: codes -> latent features [B, 512, 1]
    latent = model.mimi.quantizer.decode(codes_16)  # [B, 512, 1]
    latent = latent.transpose(0, 2, 1)  # [B, 1, 512]

    # Project to thinker embedding space
    return model.output_adaptor(latent)  # [B, 1, 4096]


def tts_generate(
    model: RaonMLX,
    input_ids: mx.array,
    max_new_tokens: int = 512,
    temperature: float = 1.2,
    audio_temperature: float = 1.2,
    top_k: int = 20,
    top_p: float = 0.8,
    speaker_embedding: mx.array | None = None,
    on_audio_frame: callable | None = None,
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
        speaker_embedding: Optional speaker conditioning. Shape: [1, 1, 4096].
            If provided, replaces the SPEAKER_EMBEDDING_PLACEHOLDER token embedding.
        on_audio_frame: Optional callback for streaming. Called with (pcm_chunk, sample_rate)
            after each decoded audio frame.

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

    # Build input embeddings, optionally replacing speaker placeholder
    inputs_embeds = model.thinker.embed_tokens(input_ids)  # [1, seq_len, 4096]

    if speaker_embedding is not None:
        # Find SPEAKER_EMBEDDING_PLACEHOLDER positions and replace
        speaker_token_id = 151671  # SPEAKER_EMBEDDING_PLACEHOLDER
        speaker_mask = (input_ids == speaker_token_id)  # [1, seq_len]
        if speaker_mask.any():
            # Replace the placeholder embedding with the speaker embedding
            spk = speaker_embedding[:, 0, :]  # [1, 4096]
            for pos in range(input_ids.shape[1]):
                if input_ids[0, pos].item() == speaker_token_id:
                    inputs_embeds = inputs_embeds.at[:, pos, :].add(spk - inputs_embeds[:, pos, :])

    # Prefill: run through thinker
    thinker_normed, thinker_pre_norm = model.thinker(inputs_embeds=inputs_embeds, cache=thinker_cache)

    # Project PRE-NORM output to talker space (critical: talker sees unnormed hidden states)
    talker_input = model.thinker_to_talker_proj(thinker_pre_norm)
    talker_out = model.talker(talker_input, cache=talker_cache)

    audio_codes_list: list[mx.array] = []
    is_generating_audio = True

    # Generate first audio frame from prefill output
    # Suppress AUDIO_END on first frame (model needs to produce at least some audio)
    first_codes = generate_audio_codes(model, talker_out[:, -1:], temperature=audio_temperature, top_k=top_k,
                                       suppress_eos=True)
    audio_codes_list.append(first_codes)

    # Autoregressive loop
    min_audio_frames = 5  # Don't allow AUDIO_END before generating at least this many frames
    for step in range(max_new_tokens - 1):
        # Feed audio code embeddings back to thinker (the critical feedback loop)
        last_codes = audio_codes_list[-1]
        audio_embed = _get_audio_output_embed(model, last_codes)  # [B, 1, 4096]
        thinker_normed, thinker_pre_norm = model.thinker(inputs_embeds=audio_embed, cache=thinker_cache)

        # Project PRE-NORM to talker
        talker_input = model.thinker_to_talker_proj(thinker_pre_norm)
        talker_out = model.talker(talker_input, cache=talker_cache)

        # Generate audio codes
        suppress = len(audio_codes_list) < min_audio_frames
        codes = generate_audio_codes(model, talker_out, temperature=audio_temperature, top_k=top_k,
                                     suppress_eos=suppress)
        audio_end = (codes[:, 0] == codebook_size).item()

        if audio_end:
            break
        audio_codes_list.append(codes)

        # Streaming: decode and emit each frame as it's generated
        if on_audio_frame is not None:
            frame_codes = codes[:, :, None]  # [1, 16, 1]
            frame_pcm = model.mimi.decode_step(frame_codes)  # [1, 1, samples_per_frame]
            mx.synchronize()
            on_audio_frame(frame_pcm[:, 0], 24000)

    if not audio_codes_list:
        return mx.zeros((1, 0)), 24000

    # Stack audio codes: [num_frames, 16] -> [1, 16, num_frames] for Mimi
    all_codes = mx.stack(audio_codes_list, axis=1)  # [1, num_frames, 16]
    all_codes = all_codes.transpose(0, 2, 1)  # [1, 16, num_frames]

    # Decode through Mimi with only the 16 generated codebooks (NOT padded to 32)
    # The RVQ decode sums contributions from each codebook — padding with zeros
    # would subtract from the reconstruction since the zero code != silence.
    pcm = model.mimi.decode(all_codes)  # [1, 1, num_samples]
    pcm = pcm[:, 0]  # [1, num_samples]

    return pcm, 24000
