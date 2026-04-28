"""Probe: feed the same L35 hidden state through full chain
(thinker_to_talker_proj -> talker -> audio_lm_head -> first code)
and compare audio codes between MLX and PT.

Uses /tmp/probe_pt_L35.npy from probe_pt_vs_mlx.py
"""

from __future__ import annotations

import os
import sys
import numpy as np

HF_PATH = os.environ.get("RAON_HF_PATH", "models/Raon-SpeechChat-9B")


def cos(a, b):
    af = a.reshape(-1).astype(np.float32); bf = b.reshape(-1).astype(np.float32)
    return float((af*bf).sum() / (np.sqrt((af*af).sum()*(bf*bf).sum()) + 1e-12))


def run_mlx(hidden):
    import mlx.core as mx
    from raon_mlx.models.raon import RaonMLX

    print("[mlx] loading...", flush=True)
    m = RaonMLX()
    m.load_weights_from_raon(HF_PATH)

    h = mx.array(hidden[None, None, :].astype(np.float32))
    talker_in = m.thinker_to_talker_proj(h)
    cache = m.talker.make_cache()
    talker_out = m.talker(talker_in, cache=cache)  # [1, 1, 2048]
    first_logits = m.audio_lm_head(talker_out[:, -1])  # [1, 2049]
    first_code = first_logits.argmax(axis=-1)  # [1]
    print(f"[mlx] first_logits norm={float(mx.linalg.norm(first_logits).item()):.3f}", flush=True)
    print(f"[mlx] argmax code = {int(first_code[0].item())}", flush=True)
    print(f"[mlx] top-5 logits/codes:", flush=True)
    fl = np.array(first_logits[0])
    top5 = np.argsort(-fl)[:5]
    for c in top5:
        print(f"  code={c} logit={fl[c]:.3f}")
    return np.array(first_logits[0]).astype(np.float32)


def run_pt(hidden):
    import torch
    from raon.models.raon import RaonModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[pt] loading on {device}...", flush=True)
    model = RaonModel.from_pretrained(HF_PATH, torch_dtype=torch.float16, trust_remote_code=True)
    model.to(device); model.train(False)

    h = torch.from_numpy(hidden[None, None, :].astype(np.float32)).to(device).to(torch.float16)
    with torch.no_grad():
        proj = model.thinker_to_talker_proj(h)
        talker_out = model.talker(inputs_embeds=proj, use_cache=False).last_hidden_state
        first_logits = model.audio_lm_head(talker_out[:, -1])  # [1, 2049]
    fl = first_logits[0].float().cpu().numpy().astype(np.float32)
    print(f"[pt] first_logits norm={np.linalg.norm(fl):.3f}", flush=True)
    print(f"[pt] argmax code = {int(np.argmax(fl))}", flush=True)
    print(f"[pt] top-5 logits/codes:", flush=True)
    top5 = np.argsort(-fl)[:5]
    for c in top5:
        print(f"  code={c} logit={fl[c]:.3f}")
    return fl


def main():
    h = np.load("/tmp/probe_pt_L35.npy")
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    m = p = None
    if which in ("mlx", "both"):
        m = run_mlx(h); np.save("/tmp/probe_logits_mlx.npy", m)
    if which in ("pt", "both"):
        p = run_pt(h); np.save("/tmp/probe_logits_pt.npy", p)
    if which == "both" and m is not None and p is not None:
        print(f"[compare] cos(mlx,pt)={cos(m,p):.6f}")
        print(f"[compare] argmax mlx={int(np.argmax(m))} pt={int(np.argmax(p))}")
        print(f"[compare] ||mlx-pt||={np.linalg.norm(m-p):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
