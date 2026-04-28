"""Profile per-section timing in the duplex_step function for one offline run.

Patches duplex_step at import time to insert mx.synchronize + perf_counter
around each major phase. Reports averages over the run.
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import mlx.core as mx

from raon_mlx.pipeline import RaonMLXPipeline
from raon_mlx.models.duplex_generate import (
    SAMPLES_PER_FRAME, init_duplex_state,
    _encode_user_audio, _get_audio_output_embed, _get_silence_codes,
    DuplexDecodingState,
)
from raon_mlx.utils.special_tokens import (
    AUDIO_INPUT_PLACEHOLDER, AUDIO_OUTPUT_PLACEHOLDER,
)
from raon_mlx.utils.state_machine import DuplexPhase

import raon_mlx.models.duplex_generate as dg


timings = defaultdict(list)


def t_section(name):
    class T:
        def __enter__(self):
            mx.synchronize()
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, *a):
            mx.synchronize()
            timings[name].append((time.perf_counter() - self.t0) * 1000)
    return T()


def patched_duplex_step(model, state, audio_input):
    with t_section("01_encode_user"):
        audio_input_embeds = _encode_user_audio(state.streaming_encoder, audio_input)
        if audio_input_embeds.shape[1] == 0:
            audio_input_embeds = mx.zeros((1, 1, 4096))
        if audio_input_embeds.shape[1] > 1:
            audio_input_embeds = audio_input_embeds[:, -1:, :]

    with t_section("02_build_embeds"):
        num_input_tokens = state.machine_state.num_input_tokens
        seq_len = state.sequences.shape[1]
        cache_pos = mx.arange(seq_len - num_input_tokens, seq_len)
        position_ids = cache_pos[None, :]
        last_tokens = state.sequences[:, -num_input_tokens:]
        frame_embeds = model.thinker.embed_tokens(last_tokens)
        last_tokens_list = last_tokens[0].tolist()
        for i, tid in enumerate(last_tokens_list):
            if tid == AUDIO_INPUT_PLACEHOLDER.id:
                before = frame_embeds[:, :i, :]
                after = frame_embeds[:, i + 1:, :]
                frame_embeds = mx.concatenate([before, audio_input_embeds, after], axis=1)
                break
        if state.prev_audio_feedback is not None:
            for i, tid in enumerate(last_tokens_list):
                if tid == AUDIO_OUTPUT_PLACEHOLDER.id:
                    before = frame_embeds[:, :i, :]
                    after = frame_embeds[:, i + 1:, :]
                    frame_embeds = mx.concatenate(
                        [before, state.prev_audio_feedback, after], axis=1,
                    )
                    break

    with t_section("03_thinker"):
        thinker_normed, thinker_pre_norm = model.thinker(
            inputs_embeds=frame_embeds, cache=state.thinker_cache,
            position_ids=position_ids, cache_position=cache_pos,
        )
        text_logits = model.lm_head(thinker_normed)

    with t_section("04_talker"):
        talker_input = model.thinker_to_talker_proj(thinker_pre_norm)
        talker_out = model.talker(talker_input, cache=state.talker_cache,
                                   position_ids=position_ids, cache_position=cache_pos)

    if state.forced_sil_remaining > 0 and state.state_manager.config.use_sil_token:
        from raon_mlx.utils.special_tokens import DUPLEX_SIL
        forced_logits = mx.full(text_logits.shape, -1e9)
        forced_logits = forced_logits.at[:, -2, DUPLEX_SIL.id].add(1e9 + 0.0)
        text_logits = forced_logits

    with t_section("05_update_seq_and_audio_codes"):
        prev_audio_codes_len = state.audio_codes.shape[1]
        (
            new_sequences, new_audio_codes, new_audio_codes_mask,
            new_machine_state, frame_codes,
        ) = dg._update_duplex_sequences_and_generate_audio_codes(
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

    with t_section("06_mimi_decode"):
        is_sil_frame = new_machine_state.phase == DuplexPhase.SIL
        new_semantic_buffer = state.semantic_buffer
        prev_audio_feedback = state.prev_audio_feedback
        if is_sil_frame:
            new_semantic_buffer = None
            silence = state._silence_codes if state._silence_codes is not None else _get_silence_codes(model)
            silence_frame = silence[None, :, None]
            decoded_audio = model.mimi.decode_step(silence_frame)
            prev_audio_feedback = _get_audio_output_embed(model, silence[None, :])
        else:
            if new_audio_codes.shape[1] > prev_audio_codes_len:
                current_codes = new_audio_codes[0, -1]
                output_codes = current_codes[None, :, None]
                decoded_audio = model.mimi.decode_step(output_codes)
                prev_audio_feedback = _get_audio_output_embed(model, current_codes[None, :])
            else:
                silence = state._silence_codes if state._silence_codes is not None else _get_silence_codes(model)
                silence_frame = silence[None, :, None]
                decoded_audio = model.mimi.decode_step(silence_frame)

    text_token_ids = []
    new_token_ids = new_sequences[0, state.last_sequence_len:].tolist()
    from raon_mlx.utils.special_tokens import (
        AUDIO_OUTPUT_PAD, AUDIO_OUTPUT_END_PAD, AUDIO_START, IM_START, DUPLEX_SIL,
    )
    ignored = {
        AUDIO_INPUT_PLACEHOLDER.id, AUDIO_OUTPUT_PLACEHOLDER.id,
        AUDIO_OUTPUT_PAD.id, AUDIO_OUTPUT_END_PAD.id,
        AUDIO_START.id, IM_START.id, DUPLEX_SIL.id,
    }
    text_token_ids = [tid for tid in new_token_ids if tid < dg.TEXT_VOCAB_SIZE and tid not in ignored]

    mx.synchronize()

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


def main():
    pipe = RaonMLXPipeline(
        os.environ.get("RAON_MODEL_PATH", "models/Raon-SpeechChat-9B-mlx-8bit"),
        hf_model_path=os.environ.get("RAON_HF_PATH", "models/Raon-SpeechChat-9B"),
        quant=os.environ.get("RAON_QUANT", "8bit"),
    )

    audio, sr = sf.read("/tmp/user_3s.wav", dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    state = init_duplex_state(
        pipe.model, pipe.processor.tokenizer,
        hf_model_path=pipe.hf_model_path,
        system_prompt="You are engaging in real-time conversation.",
    )

    n = 30
    for i in range(n):
        s = i * SAMPLES_PER_FRAME
        frame_pcm = audio[s:s + SAMPLES_PER_FRAME]
        audio_input = mx.array(frame_pcm[None, None, :])
        state, _out, _txt = patched_duplex_step(pipe.model, state, audio_input)

    print(f"\n--- Per-section ms over {n} frames (skipping first 5 for warmup) ---")
    total = 0.0
    for name in sorted(timings.keys()):
        vals = timings[name][5:]
        avg = sum(vals) / len(vals)
        mx_ = max(vals)
        total += avg
        print(f"  {name:30s}  avg={avg:6.1f}ms  max={mx_:6.1f}ms")
    print(f"  {'TOTAL (sum of avgs)':30s}  avg={total:6.1f}ms")


if __name__ == "__main__":
    main()
