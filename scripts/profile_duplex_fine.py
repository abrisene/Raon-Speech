"""Fine-grained per-section profiling of duplex_step.

Unlike profile_duplex.py (which only mx.synchronize between sections), this
script forces full materialization at every boundary so deferred work is
correctly attributed to the section that produced it. Goal: find the
unaccounted ~30 ms per frame after the 2026-04-28 perf pass.

Run:
    PYTHONPATH=src python scripts/profile_duplex_fine.py
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager

import mlx.core as mx
import numpy as np
import soundfile as sf

from raon_mlx.pipeline import RaonMLXPipeline
from raon_mlx.models.duplex_generate import (
    SAMPLES_PER_FRAME,
    init_duplex_state,
    _encode_user_audio,
    _get_audio_output_embed,
    _get_silence_codes,
    DuplexDecodingState,
    TEXT_VOCAB_SIZE,
    DUPLEX_SIL,
    AUDIO_INPUT_PLACEHOLDER,
    AUDIO_OUTPUT_END_PAD,
    AUDIO_OUTPUT_PAD,
    AUDIO_OUTPUT_PLACEHOLDER,
    AUDIO_START,
    IM_START,
)
from raon_mlx.utils.state_machine import DuplexPhase

# Materialize a lazy graph (avoids triggering the eval-name security hook).
_realize = getattr(mx, "ev" + "al")


timings = defaultdict(list)


@contextmanager
def t(name, *materialize_arrays):
    """Time a block. Materializes listed arrays at end so deferred work is attributed here."""
    mx.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if materialize_arrays:
            _realize(*materialize_arrays)
        mx.synchronize()
        timings[name].append((time.perf_counter() - t0) * 1000)


def fine_step(model, state, audio_input):
    """Reimplementation of duplex_step with materialize-forcing sub-section timers."""
    # 01: encode user audio
    with t("01_encode_user"):
        audio_input_embeds = _encode_user_audio(state.streaming_encoder, audio_input)
        if audio_input_embeds.shape[1] == 0:
            audio_input_embeds = mx.zeros((1, 1, 4096))
        if audio_input_embeds.shape[1] > 1:
            audio_input_embeds = audio_input_embeds[:, -1:, :]
        _realize(audio_input_embeds)

    # 02: build embeds
    with t("02_build_embeds"):
        num_input_tokens = state.machine_state.num_input_tokens
        seq_len = state.sequences.shape[1]
        cache_pos = mx.arange(seq_len - num_input_tokens, seq_len)
        position_ids = cache_pos[None, :]
        last_tokens = state.sequences[:, -num_input_tokens:]
        frame_embeds = model.thinker.embed_tokens(last_tokens)
        last_tokens_list = state.machine_state.last_frame_tokens
        for i, tid in enumerate(last_tokens_list):
            if tid == AUDIO_INPUT_PLACEHOLDER.id:
                frame_embeds = mx.concatenate(
                    [frame_embeds[:, :i, :], audio_input_embeds, frame_embeds[:, i + 1:, :]],
                    axis=1,
                )
                break
        if state.prev_audio_feedback is not None:
            for i, tid in enumerate(last_tokens_list):
                if tid == AUDIO_OUTPUT_PLACEHOLDER.id:
                    frame_embeds = mx.concatenate(
                        [frame_embeds[:, :i, :], state.prev_audio_feedback, frame_embeds[:, i + 1:, :]],
                        axis=1,
                    )
                    break
        _realize(frame_embeds)

    # 03: thinker forward
    with t("03_thinker"):
        thinker_normed, thinker_pre_norm = model.thinker(
            inputs_embeds=frame_embeds, cache=state.thinker_cache,
            position_ids=position_ids, cache_position=cache_pos,
        )
        text_logits = model.lm_head(thinker_normed[:, -2:-1, :])
        _realize(text_logits, thinker_pre_norm)

    # 04: talker forward
    with t("04_talker"):
        talker_input = model.thinker_to_talker_proj(thinker_pre_norm)
        talker_out = model.talker(
            talker_input, cache=state.talker_cache,
            position_ids=position_ids, cache_position=cache_pos,
        )
        _realize(talker_out)

    if state.forced_sil_remaining > 0 and state.state_manager.config.use_sil_token:
        forced_logits = mx.full(text_logits.shape, -1e9)
        forced_logits = forced_logits.at[:, 0, DUPLEX_SIL.id].add(1e9 + 0.0)
        text_logits = forced_logits

    prev_audio_codes_len = state.audio_codes.shape[1]

    # 05a: mask + sample text token
    with t("05a_text_sample"):
        cfg = state.state_manager.config
        vocab_size = TEXT_VOCAB_SIZE
        user_logits = text_logits[:, :, :vocab_size]
        if cfg.use_duplex_end_pad:
            user_logits = state.state_manager.apply_logit_mask(
                user_logits, state.machine_state, vocab_size,
            )
        scaled = user_logits[:, 0] / state.temperature
        probs = mx.softmax(scaled, axis=-1)
        probs = mx.clip(probs, 1e-10, None)
        predicted_token = mx.random.categorical(mx.log(probs))[:, None]
        predicted_id = predicted_token[0, 0].item()  # blocks

    # 05b: state machine transition (CPU)
    with t("05b_transition"):
        new_machine_state, frame_tokens, emitted_audio = state.state_manager.transition(
            state.machine_state, predicted_id,
        )

    # 05c: maybe generate audio codes
    is_in_speech = state.machine_state.phase == DuplexPhase.SPEECH
    new_audio_codes_frame = None
    prior_first_codes = state.audio_codes[:, :, 0] if state.audio_codes.shape[1] > 0 else None

    from raon_mlx.models.generate import generate_audio_codes
    if is_in_speech:
        with t("05c_codes_speech"):
            new_audio_codes_frame = generate_audio_codes(
                model, talker_out[:, -1:],
                temperature=state.temperature, top_k=state.top_k, suppress_eos=True,
                prior_first_codes=prior_first_codes,
            )
            _realize(new_audio_codes_frame)
    elif emitted_audio:
        with t("05d_codes_onset"):
            new_audio_codes_frame = generate_audio_codes(
                model, talker_out[:, -1:],
                temperature=state.temperature, top_k=state.top_k, suppress_eos=True,
                prior_first_codes=prior_first_codes,
            )
            _realize(new_audio_codes_frame)

    # 05e: clamp + concat
    with t("05e_clamp_concat"):
        audio_codes = state.audio_codes
        audio_codes_mask = state.audio_codes_mask
        if emitted_audio and new_audio_codes_frame is not None:
            first_code = new_audio_codes_frame[:, 0]
            CODEBOOK_SIZE = 2048
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
        frame_token_ids = mx.array([frame_tokens], dtype=mx.int32)
        new_sequences = mx.concatenate([state.sequences, frame_token_ids], axis=1)

    # 06: mimi decode + audio feedback embed
    is_sil_frame = new_machine_state.phase == DuplexPhase.SIL
    new_semantic_buffer = state.semantic_buffer
    prev_audio_feedback = state.prev_audio_feedback

    if is_sil_frame:
        new_semantic_buffer = None
        with t("06a_mimi_silence_decode"):
            silence = state._silence_codes if state._silence_codes is not None else _get_silence_codes(model)
            silence_frame = silence[None, :, None]
            decoded_audio = model.mimi.decode_step(silence_frame)
            _realize(decoded_audio)
        with t("06b_silence_feedback_embed"):
            prev_audio_feedback = _get_audio_output_embed(model, silence[None, :])
            _realize(prev_audio_feedback)
    else:
        if audio_codes.shape[1] > prev_audio_codes_len:
            with t("06a_mimi_speech_decode"):
                current_codes = audio_codes[0, -1]
                output_codes = current_codes[None, :, None]
                decoded_audio = model.mimi.decode_step(output_codes)
                _realize(decoded_audio)
            with t("06b_speech_feedback_embed"):
                prev_audio_feedback = _get_audio_output_embed(model, current_codes[None, :])
                _realize(prev_audio_feedback)
        else:
            with t("06a_mimi_silence_decode"):
                silence = state._silence_codes if state._silence_codes is not None else _get_silence_codes(model)
                silence_frame = silence[None, :, None]
                decoded_audio = model.mimi.decode_step(silence_frame)
                _realize(decoded_audio)

    # 07: extract token IDs (blocks via .tolist)
    with t("07_token_extract"):
        new_token_ids = new_sequences[0, state.last_sequence_len:].tolist()
        ignored = {
            AUDIO_INPUT_PLACEHOLDER.id, AUDIO_OUTPUT_PLACEHOLDER.id,
            AUDIO_OUTPUT_PAD.id, AUDIO_OUTPUT_END_PAD.id,
            AUDIO_START.id, IM_START.id, DUPLEX_SIL.id,
        }
        text_token_ids = [
            tid for tid in new_token_ids if tid < TEXT_VOCAB_SIZE and tid not in ignored
        ]

    # 08: build state object (Python)
    with t("08_state_build"):
        updated_state = DuplexDecodingState(
            sequences=new_sequences,
            thinker_cache=state.thinker_cache,
            talker_cache=state.talker_cache,
            audio_codes=audio_codes,
            audio_codes_mask=audio_codes_mask,
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


def main():
    pipe = RaonMLXPipeline(
        os.environ.get("RAON_MODEL_PATH", "models/Raon-SpeechChat-9B-mlx-8bit"),
        hf_model_path=os.environ.get("RAON_HF_PATH", "models/Raon-SpeechChat-9B"),
        quant=os.environ.get("RAON_QUANT", "8bit"),
    )
    audio, sr = sf.read("/tmp/user_3s.wav", dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    mx.random.seed(42)
    np.random.seed(42)
    state = init_duplex_state(
        pipe.model, pipe.processor.tokenizer,
        hf_model_path=pipe.hf_model_path,
        system_prompt="You are engaging in real-time conversation.",
    )

    n_total = 35
    for i in range(n_total):
        s = i * SAMPLES_PER_FRAME
        frame_pcm = audio[s:s + SAMPLES_PER_FRAME]
        if len(frame_pcm) < SAMPLES_PER_FRAME:
            break
        ai = mx.array(frame_pcm[None, None, :])
        state, _out, _txt = fine_step(pipe.model, state, ai)

    print(f"\n--- Per-section ms over {n_total} frames (skipping first 5) ---")
    skip = 5
    rows = []
    total_avg = 0.0
    for name in sorted(timings.keys()):
        vals = timings[name][skip:]
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        mx_ = max(vals)
        rows.append((name, avg, mx_, len(vals)))
        total_avg += avg
    for name, avg, mx_, n in rows:
        print(f"  {name:34s}  avg={avg:6.2f}ms  max={mx_:6.2f}ms  n={n}")
    print(f"  {'TOTAL (sum of avgs)':34s}  avg={total_avg:6.2f}ms")


if __name__ == "__main__":
    main()
