"""Run offline duplex on a user.wav to produce assistant.wav and conversation.wav."""

from __future__ import annotations

import os
import sys

from raon_mlx.pipeline import RaonMLXPipeline


def main() -> int:
    import secrets
    model_path = os.environ.get("RAON_MODEL_PATH", "models/Raon-SpeechChat-9B")
    hf_path = os.environ.get("RAON_HF_PATH", "models/Raon-SpeechChat-9B")
    user_wav = os.environ.get("RAON_USER_WAV", "output/duplex_smoke_after_encoder_fix/user.wav")
    base_out = os.environ.get("RAON_OUT", "output/duplex_run")
    suffix = os.environ.get("RAON_OUT_SUFFIX") or secrets.token_hex(2)
    out_dir = f"{base_out}_{suffix}"
    quant = os.environ.get("RAON_QUANT", "8bit")

    print(f"[run] model={model_path} quant={quant} user={user_wav} out={out_dir}", flush=True)
    pipe = RaonMLXPipeline(model_path, hf_model_path=hf_path, quant=quant)
    summary = pipe.duplex(audio_input=user_wav, output_dir=out_dir)
    print(f"[run] summary: {summary}", flush=True)
    print(f"[run] wrote {out_dir}/assistant.wav, {out_dir}/conversation.wav, {out_dir}/transcript.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
