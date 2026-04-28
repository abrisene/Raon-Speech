"""Run PT duplex on a wav file and dump per-frame audio_codes to compare with MLX."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from raon.pipeline import RaonPipeline


HF_PATH = os.environ.get("RAON_HF_PATH", "models/Raon-SpeechChat-9B")
USER_WAV = os.environ.get("RAON_USER_WAV", "/tmp/user_3s.wav")
OUT_DIR = Path(os.environ.get("RAON_OUT", "output/duplex_pt_codes"))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cpu"
    print(f"[run] PT pipeline {HF_PATH} on {device} fp32", flush=True)
    pipe = RaonPipeline(HF_PATH, device=device, dtype="float32", attn_implementation="eager")
    model = pipe.model

    audio, sr = sf.read(USER_WAV, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == 24000

    samples_per_frame = 1920
    n_frames = (len(audio) - samples_per_frame + 1) // samples_per_frame
    print(f"[run] {n_frames} frames", flush=True)

    # Build system prompt
    tokenizer = pipe.processor.tokenizer
    messages = [{"role": "system", "content": "You are engaging in real-time conversation."}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    system_tokens = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=device)

    state = model.init_duplex_decoding_state(
        sequences=system_tokens,
        attention_mask=torch.ones_like(system_tokens),
        do_sample=True,
        temperature=0.9,
        top_k=66,
        top_p=0.99,
        speak_first=False,
    )

    audio_t = torch.from_numpy(audio).to(device)
    log_path = OUT_DIR / "frame_codes.txt"
    log = open(str(log_path), "w", buffering=1)

    output_frames = []
    prev_seq_len = state.sequences.shape[1]
    for i in range(n_frames):
        s = i * samples_per_frame
        frame = audio_t[None, s:s + samples_per_frame]
        with torch.no_grad():
            state, decoded = model.duplex_decoding_step(state=state, audio_input=frame)
        output_frames.append(decoded.cpu().numpy().reshape(-1))

        cur_seq_len = state.sequences.shape[1]
        new_token_ids = state.sequences[0, prev_seq_len:cur_seq_len].tolist()
        prev_seq_len = cur_seq_len

        last_codes = state.audio_codes[0, -1].tolist() if state.audio_codes.shape[1] > 0 else []
        phase = state.machine_state.phase.name
        in_rms = float(torch.sqrt((frame ** 2).mean()).item())
        dec_np = decoded.cpu().numpy().reshape(-1)
        out_rms = float(np.sqrt((dec_np ** 2).mean()))
        log.write(
            f"[{phase}] f={i} out_rms={out_rms:.4f} in_rms={in_rms:.4f} "
            f"new_tok_ids={new_token_ids} last_codes={last_codes}\n"
        )

    log.close()

    # Save audio output
    asst = np.concatenate(output_frames).astype(np.float32)
    sf.write(str(OUT_DIR / "assistant.wav"), asst, sr)
    print(f"[run] wrote {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
