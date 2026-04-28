# Raon-Speech MLX Documentation

## Quick Reference

```bash
# Setup (one time)
uv venv --python 3.13 .venv && source .venv/bin/activate
uv pip install -r requirements.txt -e .
uv pip install 'huggingface_hub[cli]' mlx mlx-lm gradio
hf download KRAFTON/Raon-Speech-9B --local-dir models/Raon-Speech-9B

# TTS (2.6x real-time)
python -m raon_mlx.tts "Hello world" --model models/Raon-Speech-9B

# TTS with voice cloning
python -m raon_mlx.tts "Hello world" --model models/Raon-Speech-9B --speaker ref.wav

# TTS with voice seed (reproducible voice)
python -m raon_mlx.tts "Hello world" --model models/Raon-Speech-9B --quant hybrid --top-k 20

# STT (6.5x real-time)
python -m raon_mlx.stt audio.wav --hf-model models/Raon-Speech-9B

# Web demo (TTS, STT, SpeechChat, VoiceChat, TextQA)
python demo/gradio_mlx_demo.py --model models/Raon-Speech-9B --port 7880

# Optional: pre-convert for faster TTS loading (18.1 GB -> 6.46 GB)
python -m raon_mlx.utils.convert models/Raon-Speech-9B -o models/Raon-Speech-9B-mlx-hybrid
python -m raon_mlx.tts "Hello world" --model models/Raon-Speech-9B-mlx-hybrid
```

## Python API

```python
from raon_mlx.pipeline import RaonMLXPipeline

pipe = RaonMLXPipeline("models/Raon-Speech-9B", quant="hybrid")

# TTS
audio, sr = pipe.tts("Hello world!", speaker_audio="ref.wav", seed=42)
pipe.save_audio((audio, sr), "output.wav")

# STT
text = pipe.stt("input.wav")

# SpeechChat (audio in -> text out)
response = pipe.speech_chat("question.wav")

# VoiceChat (audio in -> text + audio out)
text, audio, sr = pipe.voice_chat("question.wav", speaker_audio="ref.wav")

# TextQA (text + optional audio -> text)
answer = pipe.textqa("What did the speaker say?", audio="input.wav")
```

## Documentation Index

| Document | Description |
|----------|-------------|
| [macOS Setup](macos-setup.md) | Full setup guide, dependencies, all CLI commands, benchmarks |
| [MLX Port Roadmap](mlx-port-roadmap.md) | Architecture details, phase completion status (incl. Phase 7 / Full-Duplex), per-frame profiling |
| [Test Checklist](mlx-test-checklist.md) | Comprehensive test plan with exact commands for every feature, incl. Duplex acceptance criteria |
| [MLX Duplex Debug Log](mlx-duplex-debug-log.md) | Diagnostic trail for the duplex audio bug: per-module parity proofs, sampling and quantization findings |
| [Duplex MLX Design](duplex-mlx-design.md) | Original design doc for the realtime duplex path |
| [Duplex MLX Plan](duplex-mlx-plan.md) | Original implementation plan for the realtime duplex path |
| [Qwen Backbone Upgrade](qwen-backbone-upgrade.md) | Analysis of swapping to Qwen3.5 or larger Qwen models |
| [Qwen3.5 Upgrade Plan](qwen3.5-upgrade-plan.md) | Training plan, hardware requirements, cost estimates, data needs |

## Architecture

```
Input text → Tokenizer → Thinker (Qwen3 36L or Qwen3.5 32L)
                              ↓
                    Thinker→Talker projection (MLP 4096→2048)
                              ↓
                         Talker (4 Qwen3 layers)
                              ↓
                    ┌─── audio_lm_head → 1st codebook code
                    │
                    └─── proj_code → Code predictor (5L, 15 sequential steps)
                                          ↓
                                   16 codebook codes per frame
                                          ↓
                                   Mimi decode → PCM audio
                                          ↓
                              Audio feedback: VQ decode → output_adaptor → thinker
```

For STT, audio flows the other direction:
```
Input audio → AuT encoder (24L Whisper-like) → Input adaptor (MLP 2048→4096) → Thinker
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `mlx` | ≥0.31 | Apple Silicon ML framework |
| `mlx-lm` | ≥0.31 | Qwen3.5 SSM kernel (`gated_delta_update`) |
| `torch` | ≥2.11 | Audio encoder (AuT), speaker encoder (ECAPA-TDNN) |
| `torchaudio` | ≥2.11 | Audio resampling |
| `transformers` | ≥4.57 | Tokenizer, model configs |
| `soundfile` | ≥0.13 | Audio I/O |
| `speechbrain` | ≥1.0 | Speaker encoder (ECAPA-TDNN) |
| `gradio` | ≥4.0 | Web demo |
| `safetensors` | * | Weight loading |

## Benchmarks (M5 Max, 128GB)

| Task | Speed | Details |
|------|-------|---------|
| TTS (hybrid quant) | **2.6x real-time** (RTF 0.38) | 29ms/frame, 34 fps |
| STT | **6.5x real-time** | 1.2s for 7.8s audio |
| VoiceChat | **Faster than real-time** | STT + response + TTS chained |
| Model load (pre-converted) | **Instant** | MLX lazy mmap |
| Model load (HF checkpoint) | ~2s | + on-the-fly quantization |

## Branches

| Branch | Description |
|--------|-------------|
| `main` | Upstream KRAFTON + docs + gitignore |
| `feat/mlx-port` | Production MLX port (19 commits) |
| `research/qwen-upgrade` | + Qwen3.5 backbone support (configurable) |

## Full-Duplex (Raon-SpeechChat-9B)

End-to-end realtime duplex on MLX, **RTF ~0.78 on M-series** (8-bit, 80 ms
frames, ~16 ms headroom — 2026-04-28 perf pass).

```bash
# Pre-convert at uniform 8-bit (10.32 GB; 4-bit thinker is too lossy for duplex):
python -m raon_mlx.utils.convert models/Raon-SpeechChat-9B \
    --output models/Raon-SpeechChat-9B-mlx-8bit --quant 8bit

# Realtime FastAPI + Gradio demo (open http://127.0.0.1:7862):
python demo/gradio_mlx_duplex_demo.py \
    --model-path models/Raon-SpeechChat-9B-mlx-8bit \
    --hf-model-path models/Raon-SpeechChat-9B \
    --quantize 8bit

# Offline duplex run on a 24kHz mono wav:
python scripts/run_offline_test.py

# Deterministic per-frame bench (8 runs, fixed seed):
PYTHONPATH=src python scripts/bench_duplex.py --runs 8
```

See [Phase 7 in the roadmap](mlx-port-roadmap.md), the
[Duplex Performance Profile](mlx-port-roadmap.md#duplex-performance-profile-8-bit-m-series)
section for the per-section breakdown and applied optimizations, and
[`docs/mlx-duplex-debug-log.md`](mlx-duplex-debug-log.md) for the duplex bug
hunt and per-module parity proofs.

Known soft-edge: the realtime demo can hit a Metal command-buffer race on
rapid Stop→Start session cycles (`SIGSEGV` exit 139). Wait ~1 s between
sessions or refresh the page; single sustained sessions are unaffected.

## Known Limitations

- Pre-converted MLX models have issues with STT (use `--hf-model` flag instead)
- Full-duplex SpeechChat needs `quant="8bit"` — hybrid (4-bit thinker) compounds
  error across the multi-frame loop and produces gibberish audio. Per-utterance
  TTS is unaffected.
- Qwen3.5 backbone requires adaptor retraining for speech tasks
- Tokenizer prints `OrderedVocab` warnings to stderr (cosmetic, from Rust tokenizer)
- Other LLMs sharing the GPU will reduce throughput significantly
