# Duplex (full-duplex realtime) generation engine for Raon-Speech MLX.
# Frame-by-frame bidirectional inference: encode user audio + generate assistant audio.

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .raon import RaonMLX
from .generate import sample_token, generate_audio_codes, _get_audio_output_embed
from ..modules.kv_cache import KVCache, create_additive_causal_mask
from ..utils.streaming_encoder import StreamingAudioEncoder
from ..utils.special_tokens import (
    AUDIO_INPUT_PLACEHOLDER,
    AUDIO_OUTPUT_END_PAD,
    AUDIO_OUTPUT_PAD,
    AUDIO_OUTPUT_PLACEHOLDER,
    AUDIO_START,
    DUPLEX_SIL,
    IM_START,
    SPEAKER_EMBEDDING_PLACEHOLDER,
)
from ..utils.state_machine import (
    DuplexMachineState,
    DuplexPhase,
    DuplexStateConfig,
    DuplexStateManager,
)

logger = logging.getLogger(__name__)

# Constants
SAMPLES_PER_FRAME = 1920  # 80ms at 24kHz
CODEBOOK_SIZE = 2048
NUM_CODE_GROUPS = 16
TEXT_VOCAB_SIZE = 151936  # Qwen3 text vocab (tokens below this are text)


@dataclass
class DuplexDecodingState:
    """Mutable state for one active duplex decoding session."""

    sequences: mx.array  # [1, seq_len] int32
    thinker_cache: list[KVCache]
    talker_cache: list[KVCache]
    audio_codes: mx.array  # [1, num_frames, 16]
    audio_codes_mask: mx.array  # [1, num_frames] bool
    machine_state: DuplexMachineState
    state_manager: DuplexStateManager
    semantic_buffer: mx.array | None
    temperature: float = 0.9
    top_k: int = 66
    top_p: float = 0.99
    eos_penalty: float = 0.0
    sil_penalty: float = 0.0
    bc_penalty: float = 0.0
    speaker_embeds: mx.array | None = None
    forced_sil_remaining: int = 0
    last_sequence_len: int = 0
    prev_audio_feedback: mx.array | None = None  # [1, 1, 4096]
    streaming_encoder: StreamingAudioEncoder | None = field(default=None, repr=False)
    _silence_codes: mx.array | None = field(default=None, repr=False)


def _get_silence_codes(model: RaonMLX) -> mx.array:
    """Get silence codebook values by encoding a zero-PCM frame."""
    silence_pcm = mx.zeros((1, 1, SAMPLES_PER_FRAME))
    codes = model.mimi.encode(silence_pcm)  # [1, codebooks, 1]
    return codes[0, :NUM_CODE_GROUPS, 0]  # [16]


def _encode_user_audio(
    encoder: StreamingAudioEncoder,
    pcm: mx.array,
) -> mx.array:
    """Encode one frame of user audio to thinker embedding space.

    Uses the PyTorch streaming encoder (Voxtral/AuT) with input adaptor.

    Args:
        encoder: Streaming audio encoder (PyTorch on MPS).
        pcm: Raw PCM audio [1, 1, 1920].

    Returns:
        Thinker-space embedding [1, num_frames, 4096].
        num_frames is typically 1 but may be 0 if buffering.
    """
    return encoder.encode_frame(pcm)


def init_duplex_state(
    model: RaonMLX,
    tokenizer,
    *,
    hf_model_path: str,
    system_prompt: str = "You are engaging in real-time conversation.",
    speak_first: bool = False,
    temperature: float = 0.9,
    top_k: int = 66,
    top_p: float = 0.99,
    eos_penalty: float = 0.0,
    sil_penalty: float = 0.0,
    bc_penalty: float = 0.0,
    speaker_embeds: mx.array | None = None,
) -> DuplexDecodingState:
    """Initialize duplex decoding state and run the first frame.

    Returns:
        Initialized DuplexDecodingState ready for duplex_step().
    """
    state_config = DuplexStateConfig(
        use_duplex_end_pad=True,
        use_sil_token=True,
        no_audio_in_sil=False,
        sequence_mode="uta",
        use_backchannel_token=False,
    )
    state_manager = DuplexStateManager(state_config)

    # Initialize MLX streaming audio encoder (pure MLX; replaces prior PyTorch wrapper)
    from ..utils.streaming_encoder import get_streaming_encoder
    streaming_encoder = get_streaming_encoder(hf_model_path)
    streaming_encoder.reset()

    # Tokenize system prompt
    messages = [{"role": "system", "content": system_prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    prompt_tokens = tokenizer.encode(prompt_text)

    if speaker_embeds is not None:
        prompt_tokens.append(SPEAKER_EMBEDDING_PLACEHOLDER.id)

    prompt_tokens.extend([IM_START.id, AUDIO_START.id])
    sequences = mx.array([prompt_tokens], dtype=mx.int32)

    # Build input embeddings
    inputs_embeds = model.thinker.embed_tokens(sequences)

    if speaker_embeds is not None:
        input_list = sequences[0].tolist()
        for i, tid in enumerate(input_list):
            if tid == SPEAKER_EMBEDDING_PLACEHOLDER.id:
                spk = speaker_embeds[:, 0, :].astype(inputs_embeds.dtype)
                before = inputs_embeds[:, :i, :]
                after = inputs_embeds[:, i + 1:, :]
                inputs_embeds = mx.concatenate([before, spk[:, None, :], after], axis=1)
                break

    thinker_cache = model.thinker.make_cache()
    talker_cache = model.talker.make_cache()

    thinker_normed, thinker_pre_norm = model.thinker(
        inputs_embeds=inputs_embeds, cache=thinker_cache,
    )
    text_logits = model.lm_head(thinker_normed)

    initial_machine_state = state_manager.initial_state(speak_first=speak_first)

    # Force initial prediction
    forced_id = state_manager.initial_forced_prediction_id(speak_first)
    if forced_id is not None:
        forced_logits = mx.full(text_logits.shape, -1e9)
        forced_logits = forced_logits.at[:, -2, forced_id].add(1e9 + 0.0)
        text_logits = forced_logits

    audio_codes = mx.zeros((1, 0, NUM_CODE_GROUPS), dtype=mx.int32)
    audio_codes_mask = mx.zeros((1, 0), dtype=mx.bool_)

    # Run talker on init prefill output (keeps talker cache aligned with thinker)
    init_talker_input = model.thinker_to_talker_proj(thinker_pre_norm)
    init_talker_out = model.talker(init_talker_input, cache=talker_cache)

    (
        new_sequences, new_audio_codes, new_audio_codes_mask,
        new_machine_state, first_frame_codes,
    ) = _update_duplex_sequences_and_generate_audio_codes(
        model=model,
        text_logits=text_logits,
        talker_out=init_talker_out,
        sequences=sequences,
        audio_codes=audio_codes,
        audio_codes_mask=audio_codes_mask,
        machine_state=initial_machine_state,
        state_manager=state_manager,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_penalty=eos_penalty,
        sil_penalty=sil_penalty,
        bc_penalty=bc_penalty,
    )

    # Run the newly appended frame tokens through the thinker to update KV cache.
    # The prefill covered the system prompt + [IM_START, AUDIO_START].
    # _update_duplex_sequences appended frame tokens (e.g. [AIP, SIL/text, AOP])
    # to sequences but didn't run them through the thinker. We need the cache
    # to include these so the next duplex_step's attention is correctly aligned.
    # Run the newly appended frame tokens through thinker + talker with explicit
    # cache_position so they're written at the correct positions.
    num_new_tokens = new_sequences.shape[1] - sequences.shape[1]
    if num_new_tokens > 0:
        new_token_ids = new_sequences[:, -num_new_tokens:]
        new_embeds = model.thinker.embed_tokens(new_token_ids)
        # If audio output was generated, replace AOP position with feedback embedding
        if new_machine_state.emitted_audio and new_audio_codes.shape[1] > 0:
            last_codes = new_audio_codes[:, -1, :]
            feedback_embed = _get_audio_output_embed(model, last_codes)
            new_embeds = mx.concatenate([
                new_embeds[:, :-1, :],
                feedback_embed,
            ], axis=1)

        # Explicit positions: these tokens are at the END of the sequence
        init_cache_pos = mx.arange(
            new_sequences.shape[1] - num_new_tokens,
            new_sequences.shape[1],
        )
        init_pos_ids = init_cache_pos[None, :]

        _, thinker_pre_norm_frame = model.thinker(
            inputs_embeds=new_embeds, cache=thinker_cache,
            position_ids=init_pos_ids, cache_position=init_cache_pos,
        )
        talker_frame_input = model.thinker_to_talker_proj(thinker_pre_norm_frame)
        model.talker(
            talker_frame_input, cache=talker_cache,
            position_ids=init_pos_ids, cache_position=init_cache_pos,
        )

    # Reset Mimi streaming state
    model.mimi.reset_all()
    silence_codes = _get_silence_codes(model)
    model.mimi.reset_all()

    emitted_audio = new_machine_state.emitted_audio
    prev_audio_feedback = None

    if not emitted_audio:
        silence_frame = silence_codes[None, :, None]
        model.mimi.decode_step(silence_frame)
    else:
        if new_audio_codes.shape[1] > 0:
            last_codes = new_audio_codes[:, -1, :]
            decode_codes = last_codes[:, :, None]
            model.mimi.decode_step(decode_codes)
            prev_audio_feedback = _get_audio_output_embed(model, last_codes)

    forced_sil_remaining = 1 if (not speak_first and state_config.use_sil_token) else 0

    # Force computation before returning
    mx.synchronize()

    return DuplexDecodingState(
        sequences=new_sequences,
        thinker_cache=thinker_cache,
        talker_cache=talker_cache,
        audio_codes=new_audio_codes,
        audio_codes_mask=new_audio_codes_mask,
        machine_state=new_machine_state,
        state_manager=state_manager,
        semantic_buffer=None,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_penalty=eos_penalty,
        sil_penalty=sil_penalty,
        bc_penalty=bc_penalty,
        speaker_embeds=speaker_embeds,
        forced_sil_remaining=forced_sil_remaining,
        last_sequence_len=new_sequences.shape[1],
        prev_audio_feedback=prev_audio_feedback,
        streaming_encoder=streaming_encoder,
        _silence_codes=silence_codes,
    )


def duplex_step(
    model: RaonMLX,
    state: DuplexDecodingState,
    audio_input: mx.array,
) -> tuple[DuplexDecodingState, mx.array, list[int]]:
    """Run one duplex decoding step.

    Args:
        model: RaonMLX model.
        state: Current duplex decoding state.
        audio_input: One frame of user audio [1, 1, 1920] float32.

    Returns:
        (updated_state, output_audio, new_text_token_ids)
        output_audio: [1, 1, 1920] float32.
        new_text_token_ids: List of new text token IDs (empty if none).
    """
    # 1. Encode user audio via PyTorch streaming encoder
    assert state.streaming_encoder is not None, "streaming_encoder required for duplex"
    audio_input_embeds = _encode_user_audio(state.streaming_encoder, audio_input)

    if audio_input_embeds.shape[1] == 0:
        audio_input_embeds = mx.zeros((1, 1, 4096))
    if audio_input_embeds.shape[1] > 1:
        audio_input_embeds = audio_input_embeds[:, -1:, :]

    # 2. Build thinker input from last frame tokens.
    # The last frame tokens are ALREADY in the KV cache from the previous step.
    # We OVERWRITE those cache positions with new K/V that include the real audio
    # embeddings, using explicit cache_position (matching PyTorch behavior).
    num_input_tokens = state.machine_state.num_input_tokens
    seq_len = state.sequences.shape[1]

    # Compute explicit position_ids and cache_position for the last N tokens
    # (same as PyTorch: cache_position = arange(seq_len - N, seq_len))
    cache_pos = mx.arange(seq_len - num_input_tokens, seq_len)
    position_ids = cache_pos[None, :]  # [1, N]

    last_tokens = state.sequences[:, -num_input_tokens:]
    frame_embeds = model.thinker.embed_tokens(last_tokens)
    last_tokens_list = last_tokens[0].tolist()

    # Replace AUDIO_INPUT_PLACEHOLDER with encoded user audio
    for i, tid in enumerate(last_tokens_list):
        if tid == AUDIO_INPUT_PLACEHOLDER.id:
            before = frame_embeds[:, :i, :]
            after = frame_embeds[:, i + 1:, :]
            frame_embeds = mx.concatenate([before, audio_input_embeds, after], axis=1)
            break

    # Replace AUDIO_OUTPUT_PLACEHOLDER with feedback from previous frame
    if state.prev_audio_feedback is not None:
        for i, tid in enumerate(last_tokens_list):
            if tid == AUDIO_OUTPUT_PLACEHOLDER.id:
                before = frame_embeds[:, :i, :]
                after = frame_embeds[:, i + 1:, :]
                frame_embeds = mx.concatenate(
                    [before, state.prev_audio_feedback, after], axis=1,
                )
                break

    # 3. Thinker forward with explicit positions — overwrites cache at those positions
    thinker_normed, thinker_pre_norm = model.thinker(
        inputs_embeds=frame_embeds, cache=state.thinker_cache,
        position_ids=position_ids, cache_position=cache_pos,
    )
    text_logits = model.lm_head(thinker_normed)

    # 3b. Talker forward — must run on EVERY frame to keep cache aligned.
    # Uses same position_ids/cache_position as thinker.
    talker_input = model.thinker_to_talker_proj(thinker_pre_norm)
    talker_out = model.talker(talker_input, cache=state.talker_cache,
                               position_ids=position_ids, cache_position=cache_pos)

    # 4. Force SIL if in warmup
    if state.forced_sil_remaining > 0 and state.state_manager.config.use_sil_token:
        forced_logits = mx.full(text_logits.shape, -1e9)
        forced_logits = forced_logits.at[:, -2, DUPLEX_SIL.id].add(1e9 + 0.0)
        text_logits = forced_logits

    # 5. Update sequences and generate audio codes
    prev_audio_codes_len = state.audio_codes.shape[1]
    (
        new_sequences, new_audio_codes, new_audio_codes_mask,
        new_machine_state, frame_codes,
    ) = _update_duplex_sequences_and_generate_audio_codes(
        model=model,
        text_logits=text_logits,
        talker_out=talker_out,
        sequences=state.sequences,
        audio_codes=state.audio_codes,
        audio_codes_mask=state.audio_codes_mask,
        machine_state=state.machine_state,
        state_manager=state.state_manager,
        temperature=state.temperature,
        top_k=state.top_k,
        top_p=state.top_p,
        eos_penalty=state.eos_penalty,
        sil_penalty=state.sil_penalty,
        bc_penalty=state.bc_penalty,
    )

    # 6. Decode audio output
    is_sil_frame = not new_machine_state.emitted_audio
    new_semantic_buffer = state.semantic_buffer
    prev_audio_feedback = state.prev_audio_feedback

    if is_sil_frame:
        new_semantic_buffer = None
        silence = state._silence_codes
        if silence is None:
            silence = _get_silence_codes(model)
        silence_frame = silence[None, :, None]
        decoded_audio = model.mimi.decode_step(silence_frame)
    else:
        if new_audio_codes.shape[1] > prev_audio_codes_len:
            current_codes = new_audio_codes[0, -1]  # [16]
            output_codes = current_codes[None, :, None]  # [1, 16, 1]
            decoded_audio = model.mimi.decode_step(output_codes)
            prev_audio_feedback = _get_audio_output_embed(model, current_codes[None, :])
        else:
            silence = state._silence_codes
            if silence is None:
                silence = _get_silence_codes(model)
            silence_frame = silence[None, :, None]
            decoded_audio = model.mimi.decode_step(silence_frame)

    # 7. Extract new text token IDs
    new_token_ids = new_sequences[0, state.last_sequence_len:].tolist()
    ignored = {
        AUDIO_INPUT_PLACEHOLDER.id, AUDIO_OUTPUT_PLACEHOLDER.id,
        AUDIO_OUTPUT_PAD.id, AUDIO_OUTPUT_END_PAD.id,
        AUDIO_START.id, IM_START.id, DUPLEX_SIL.id,
    }
    text_token_ids = [
        tid for tid in new_token_ids
        if tid < TEXT_VOCAB_SIZE and tid not in ignored
    ]

    # Force evaluation
    mx.synchronize()

    # 8. Build updated state
    updated_state = DuplexDecodingState(
        sequences=new_sequences,
        thinker_cache=state.thinker_cache,
        talker_cache=state.talker_cache,
        audio_codes=new_audio_codes,
        audio_codes_mask=new_audio_codes_mask,
        machine_state=new_machine_state,
        state_manager=state.state_manager,
        semantic_buffer=new_semantic_buffer,
        temperature=state.temperature,
        top_k=state.top_k,
        top_p=state.top_p,
        eos_penalty=state.eos_penalty,
        sil_penalty=state.sil_penalty,
        bc_penalty=state.bc_penalty,
        speaker_embeds=state.speaker_embeds,
        forced_sil_remaining=max(0, state.forced_sil_remaining - 1),
        last_sequence_len=new_sequences.shape[1],
        prev_audio_feedback=prev_audio_feedback,
        streaming_encoder=state.streaming_encoder,
        _silence_codes=state._silence_codes,
    )

    return updated_state, decoded_audio, text_token_ids


def _update_duplex_sequences_and_generate_audio_codes(
    *,
    model: RaonMLX,
    text_logits: mx.array,
    talker_out: mx.array,
    sequences: mx.array,
    audio_codes: mx.array,
    audio_codes_mask: mx.array,
    machine_state: DuplexMachineState,
    state_manager: DuplexStateManager,
    temperature: float,
    top_k: int,
    top_p: float,
    eos_penalty: float,
    sil_penalty: float,
    bc_penalty: float,
) -> tuple[mx.array, mx.array, mx.array, DuplexMachineState, mx.array | None]:
    """Sample text prediction, generate audio codes, update sequences.

    The talker has already been run by the caller (duplex_step runs it on every
    frame to keep its KV cache aligned). We use the pre-computed talker_out
    for audio code generation.

    Returns:
        (new_sequences, new_audio_codes, new_audio_codes_mask, new_machine_state, frame_codes)
    """
    cfg = state_manager.config
    vocab_size = TEXT_VOCAB_SIZE

    # Text prediction at position -2 (before [A])
    user_logits = text_logits[:, -2:-1, :vocab_size]

    # Apply penalties
    if eos_penalty > 0:
        user_logits = user_logits.at[..., cfg.duplex_pad_token_id].add(-eos_penalty)
    if sil_penalty > 0 and cfg.use_sil_token:
        user_logits = user_logits.at[..., cfg.duplex_sil_token_id].add(-sil_penalty)
    if (
        bc_penalty != 0
        and cfg.use_backchannel_token
        and machine_state.phase == DuplexPhase.SIL
    ):
        user_logits = user_logits.at[..., cfg.duplex_bc_token_id].add(-bc_penalty)

    # State machine logit mask
    if cfg.use_duplex_end_pad:
        user_logits = state_manager.apply_logit_mask(
            user_logits, machine_state, vocab_size,
        )

    # Sample text prediction
    if temperature > 0:
        scaled_logits = user_logits[:, 0] / temperature
        probs = mx.softmax(scaled_logits, axis=-1)
        probs = mx.clip(probs, 1e-10, None)
        predicted_token = mx.random.categorical(mx.log(probs))[:, None]
    else:
        predicted_token = user_logits[:, 0].argmax(axis=-1, keepdims=True)

    predicted_id = predicted_token[0, 0].item()

    # Generate audio codes using pre-computed talker output
    is_in_speech = machine_state.phase == DuplexPhase.SPEECH
    new_audio_codes_frame = None

    if is_in_speech:
        new_audio_codes_frame = generate_audio_codes(
            model, talker_out[:, -1:], temperature=1.2, top_k=top_k, suppress_eos=True,
        )

    # State machine transition
    new_machine_state, frame_tokens, emitted_audio = state_manager.transition(
        machine_state, predicted_id,
    )

    # Onset frame (SIL -> SPEECH): generate codes from talker output
    if emitted_audio and new_audio_codes_frame is None:
        new_audio_codes_frame = generate_audio_codes(
            model, talker_out[:, -1:], temperature=1.2, top_k=top_k, suppress_eos=True,
        )

    # Append audio codes
    if emitted_audio and new_audio_codes_frame is not None:
        # Clamp audio-end sentinel
        first_code = new_audio_codes_frame[:, 0]
        is_eos = first_code >= CODEBOOK_SIZE
        if is_eos.any().item():
            new_audio_codes_frame = mx.concatenate([
                mx.zeros((1, 1), dtype=new_audio_codes_frame.dtype),
                new_audio_codes_frame[:, 1:],
            ], axis=1)

        audio_codes = mx.concatenate(
            [audio_codes, new_audio_codes_frame[:, None, :]], axis=1,
        )
        audio_codes_mask = mx.concatenate(
            [audio_codes_mask, mx.array([[True]])], axis=1,
        )

    # Append frame tokens to sequences
    frame_token_ids = mx.array([frame_tokens], dtype=mx.int32)
    sequences = mx.concatenate([sequences, frame_token_ids], axis=1)

    return sequences, audio_codes, audio_codes_mask, new_machine_state, new_audio_codes_frame


def run_duplex_offline(
    model: RaonMLX,
    tokenizer,
    audio_path: str,
    output_dir: str,
    *,
    hf_model_path: str,
    system_prompt: str = "You are engaging in real-time conversation.",
    speak_first: bool = False,
    temperature: float = 0.9,
    top_k: int = 66,
    top_p: float = 0.99,
    eos_penalty: float = 0.0,
    sil_penalty: float = 0.0,
    bc_penalty: float = 0.0,
    speaker_embeds: mx.array | None = None,
) -> dict:
    """Run full-duplex inference on a WAV file (offline test harness).

    Returns:
        Summary dict with durations and sample counts.
    """
    import soundfile as sf

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load audio
    audio_data, sr = sf.read(audio_path, dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    if sr != 24000:
        ratio = 24000 / sr
        new_len = int(len(audio_data) * ratio)
        indices = np.arange(new_len) / ratio
        lo = np.floor(indices).astype(int)
        hi = np.minimum(lo + 1, len(audio_data) - 1)
        frac = indices - lo
        audio_data = audio_data[lo] * (1 - frac) + audio_data[hi] * frac
    sr = 24000

    # Init state
    logger.info("Initializing duplex state...")
    state = init_duplex_state(
        model, tokenizer,
        hf_model_path=hf_model_path,
        system_prompt=system_prompt,
        speak_first=speak_first,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_penalty=eos_penalty,
        sil_penalty=sil_penalty,
        bc_penalty=bc_penalty,
        speaker_embeds=speaker_embeds,
    )

    # Process frames
    num_samples = len(audio_data)
    num_frames = (num_samples - SAMPLES_PER_FRAME + 1) // SAMPLES_PER_FRAME
    if num_frames <= 0:
        raise ValueError(
            f"Audio too short: {num_samples} samples, need >= {SAMPLES_PER_FRAME}"
        )

    output_frames: list[np.ndarray] = []
    all_text_ids: list[int] = []
    frame_times: list[float] = []

    logger.info(f"Running duplex: {num_frames} frames ({num_samples / sr:.2f}s)")
    for i in range(num_frames):
        start_idx = i * SAMPLES_PER_FRAME
        frame_pcm = audio_data[start_idx:start_idx + SAMPLES_PER_FRAME]
        audio_input = mx.array(frame_pcm[None, None, :])

        t0 = time.perf_counter()
        state, output_audio, text_ids = duplex_step(model, state, audio_input)
        t1 = time.perf_counter()
        frame_times.append(t1 - t0)

        out_np = np.array(output_audio[0, 0], copy=False).astype(np.float32)
        output_frames.append(out_np)
        all_text_ids.extend(text_ids)

        phase = state.machine_state.phase.value
        if (i + 1) % 25 == 0 or i == 0:
            avg_ms = np.mean(frame_times[-25:]) * 1000
            logger.info(f"  Frame {i + 1}/{num_frames} [{phase}] avg={avg_ms:.1f}ms")

    # Save outputs
    assistant_audio = np.concatenate(output_frames)
    sf.write(str(output_path / "assistant.wav"), assistant_audio, sr)

    user_audio = audio_data[:num_frames * SAMPLES_PER_FRAME]
    sf.write(str(output_path / "user.wav"), user_audio, sr)

    max_len = max(len(user_audio), len(assistant_audio))
    user_padded = np.pad(user_audio, (0, max_len - len(user_audio)))
    asst_padded = np.pad(assistant_audio, (0, max_len - len(assistant_audio)))
    stereo = np.stack([user_padded, asst_padded], axis=-1)
    sf.write(str(output_path / "conversation.wav"), stereo, sr)

    transcript = tokenizer.decode(all_text_ids, skip_special_tokens=False) if all_text_ids else ""
    (output_path / "transcript.txt").write_text(transcript, encoding="utf-8")

    total_decode = sum(frame_times)
    user_seconds = num_frames * SAMPLES_PER_FRAME / sr
    summary = {
        "user_duration_sec": user_seconds,
        "assistant_samples": len(assistant_audio),
        "assistant_duration_sec": len(assistant_audio) / sr,
        "num_frames": num_frames,
        "total_decode_sec": total_decode,
        "avg_frame_ms": float(np.mean(frame_times) * 1000),
        "max_frame_ms": float(np.max(frame_times) * 1000),
        "rtf": total_decode / user_seconds if user_seconds > 0 else 0,
        "transcript": transcript,
    }

    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    logger.info(f"Done. RTF={summary['rtf']:.3f}, avg={summary['avg_frame_ms']:.1f}ms/frame")
    logger.info(f"Transcript: {transcript[:200]}")

    return summary
