"""Probe: feed the saved L35 hidden state through both MLX and PT talkers
(via thinker_to_talker_proj) and compare outputs.

Run after probe_pt_vs_mlx.py both has been run (uses /tmp/probe_pt_L35.npy).
"""

from __future__ import annotations

import os
import sys

import numpy as np

HF_PATH = os.environ.get("RAON_HF_PATH", "models/Raon-SpeechChat-9B")


def cos(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1).astype(np.float32)
    bf = b.reshape(-1).astype(np.float32)
    denom = float(np.sqrt((af * af).sum() * (bf * bf))) + 1e-12
    return float((af * bf).sum() / denom)


def run_mlx(hidden_4096):
    import mlx.core as mx
    from raon_mlx.models.raon import RaonMLX

    print("[mlx] loading model fp16...", flush=True)
    model = RaonMLX()
    model.load_weights_from_raon(HF_PATH)

    h = mx.array(hidden_4096[None, None, :].astype(np.float32))
    talker_in = model.thinker_to_talker_proj(h)
    cache = model.talker.make_cache()
    out = model.talker(talker_in, cache=cache)
    arr = np.array(out[:, 0, :], copy=True).astype(np.float32).reshape(-1)
    print(f"[mlx] talker out norm = {np.linalg.norm(arr):.4f}", flush=True)
    return arr


def run_pt(hidden_4096):
    import torch
    from transformers import AutoConfig
    from raon.models.raon import RaonModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[pt] loading model fp16 on {device}...", flush=True)
    _ = AutoConfig.from_pretrained(HF_PATH, trust_remote_code=True)
    model = RaonModel.from_pretrained(HF_PATH, torch_dtype=torch.float16, trust_remote_code=True)
    model.to(device)
    model.train(False)

    h = torch.from_numpy(hidden_4096[None, None, :].astype(np.float32)).to(device).to(torch.float16)
    with torch.no_grad():
        proj = model.thinker_to_talker_proj(h)
        talker_out = model.talker(inputs_embeds=proj, use_cache=False)
    last = talker_out.last_hidden_state[:, 0, :].float().cpu().numpy().reshape(-1)
    print(f"[pt] talker out norm = {np.linalg.norm(last):.4f}", flush=True)
    return last


def main() -> int:
    h_path = "/tmp/probe_pt_L35.npy"
    if not os.path.exists(h_path):
        print(f"[error] missing {h_path} — run probe_pt_vs_mlx.py both first")
        return 1
    h = np.load(h_path)
    print(f"[probe] hidden norm = {np.linalg.norm(h):.4f}", flush=True)

    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    m = p = None
    if which in ("mlx", "both"):
        m = run_mlx(h)
        np.save("/tmp/probe_mlx_talker.npy", m)
    if which in ("pt", "both"):
        p = run_pt(h)
        np.save("/tmp/probe_pt_talker.npy", p)
    if which == "both" and m is not None and p is not None:
        print(f"[compare] cos(mlx, pt) = {cos(m, p):.6f}", flush=True)
        print(f"[compare] norm mlx={np.linalg.norm(m):.4f} pt={np.linalg.norm(p):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
