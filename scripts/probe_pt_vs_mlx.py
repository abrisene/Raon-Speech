"""Probe: compare PT vs MLX thinker hidden state at L35 last position
on the same simple prefix.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HF_PATH = os.environ.get("RAON_HF_PATH", "models/Raon-SpeechChat-9B")
PROMPT = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello world.<|im_end|>\n<|im_start|>assistant\n"


def cos(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1).astype(np.float32)
    bf = b.reshape(-1).astype(np.float32)
    denom = float(np.sqrt((af * af).sum() * (bf * bf))) + 1e-12
    return float((af * bf).sum() / denom)


def run_mlx(token_ids):
    import mlx.core as mx
    from raon_mlx.models.raon import RaonMLX

    print("[mlx] loading model fp16...", flush=True)
    model = RaonMLX()
    model.load_weights_from_raon(HF_PATH)

    ids = mx.array([token_ids], dtype=mx.int32)
    embeds = model.thinker.embed_tokens(ids)
    cache = model.thinker.make_cache()
    _normed, pre_norm = model.thinker(inputs_embeds=embeds, cache=cache)
    last = pre_norm[:, -1, :]
    arr = np.array(last, copy=True).astype(np.float32)
    print(f"[mlx] L35 last-pos norm = {np.linalg.norm(arr):.4f}", flush=True)
    return arr.reshape(-1)


def run_pt(token_ids):
    import torch
    from transformers import AutoConfig
    from raon.models.raon import RaonModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[pt] loading model fp16 on {device}...", flush=True)
    _cfg = AutoConfig.from_pretrained(HF_PATH, trust_remote_code=True)
    model = RaonModel.from_pretrained(HF_PATH, torch_dtype=torch.float16, trust_remote_code=True)
    model.to(device)
    model.train(False)

    captured = {}

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["L35"] = h.detach()

    text_model = model.text_model
    text_model.layers[-1].register_forward_hook(hook)

    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        text_model(input_ids=ids, use_cache=False, output_hidden_states=False)

    last = captured["L35"][:, -1, :].float().cpu().numpy()
    print(f"[pt] L35 last-pos norm = {np.linalg.norm(last):.4f}", flush=True)
    return last.reshape(-1)


def main() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_PATH, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=False)
    print(f"[probe] prompt tokens: {len(ids)}", flush=True)

    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    m = p = None
    if which in ("mlx", "both"):
        m = run_mlx(ids)
        np.save("/tmp/probe_mlx_L35.npy", m)
    if which in ("pt", "both"):
        p = run_pt(ids)
        np.save("/tmp/probe_pt_L35.npy", p)
    if which == "both" and m is not None and p is not None:
        print(f"[compare] cos(mlx, pt) = {cos(m, p):.6f}", flush=True)
        print(f"[compare] norm mlx={np.linalg.norm(m):.4f} pt={np.linalg.norm(p):.4f}", flush=True)
        diff = m - p
        print(f"[compare] ||mlx - pt||_2 = {np.linalg.norm(diff):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
