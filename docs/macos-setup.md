# Raon-Speech macOS Setup

## Environment

- **Machine**: Apple M5 Max, 128GB RAM
- **Python**: 3.13 via uv
- **PyTorch**: 2.11.0 (MPS backend)
- **MLX**: 0.31.1 (recommended — 2.6x real-time TTS)
- **Models downloaded**: `models/Raon-Speech-9B/`, `models/Raon-SpeechChat-9B/`

## Setup Steps

```bash
# Create venv
uv venv --python 3.13 .venv
source .venv/bin/activate

# Install deps
uv pip install -r requirements.txt -e .
uv pip install 'huggingface_hub[cli]' mlx mlx-lm gradio

# Download models (requires HF login: `hf auth login`)
hf download KRAFTON/Raon-Speech-9B --local-dir models/Raon-Speech-9B
hf download KRAFTON/Raon-SpeechChat-9B --local-dir models/Raon-SpeechChat-9B
```

## Running on MLX (Recommended)

MLX with hybrid quantization (4-bit backbone, 8-bit audio components) gives
2.6x real-time TTS on Apple Silicon.

### Quick Start

```bash
# TTS (2.6x real-time)
python -m raon_mlx.tts "Hello world!" --model models/Raon-Speech-9B --quant hybrid

# TTS with voice cloning
python -m raon_mlx.tts "Hello world!" --model models/Raon-Speech-9B --speaker ref.wav

# STT
python -m raon_mlx.stt audio.wav --hf-model models/Raon-Speech-9B

# Gradio web demo (TTS, STT, SpeechChat, TextQA)
python demo/gradio_mlx_demo.py --model models/Raon-Speech-9B --port 7880
```

### Optional: Pre-convert model for faster loading

```bash
# Convert once (18.1 GB -> 6.46 GB):
python -m raon_mlx.utils.convert models/Raon-Speech-9B -o models/Raon-Speech-9B-mlx-hybrid

# Use pre-converted model (TTS only, STT needs --hf-model for now):
python -m raon_mlx.tts "Hello world!" --model models/Raon-Speech-9B-mlx-hybrid
```

### Python API

```python
from raon_mlx.pipeline import RaonMLXPipeline

pipe = RaonMLXPipeline("models/Raon-Speech-9B", quant="hybrid")

# TTS
audio, sr = pipe.tts("Hello world!", speaker_audio="ref.wav")
pipe.save_audio((audio, sr), "output.wav")

# STT
text = pipe.stt("input.wav")

# SpeechChat
response = pipe.speech_chat("input.wav")
```

## Running on MPS (PyTorch, slower)

The original PyTorch pipeline works on MPS but is slower (0.4x real-time):

```python
from raon import RaonPipeline

pipe = RaonPipeline("models/Raon-Speech-9B", device="mps", dtype="float16", attn_implementation="sdpa")
audio, sr = pipe.tts("Hello world!")
pipe.save_audio((audio, sr), "output/test.wav")
```

## Benchmarks (M5 Max)

| Setup | TTS RTF | TTS Speed | STT (7.8s audio) |
|-------|---------|-----------|-------------------|
| **MLX hybrid quant** | **0.38** | **2.6x real-time** | **1.2s** |
| PyTorch MPS float16 | 2.52 | 0.4x real-time | ~3s |
| KRAFTON RTX 6000 Pro | 0.27 | 3.7x real-time | — |

MLX model size: 6.46 GB (hybrid quant) vs 18.1 GB (float16).

Per-frame breakdown (MLX hybrid, 29ms/frame avg):
- Thinker (36 Qwen3 layers, 4-bit): 14.5ms (32%)
- Code predictor (15 steps, 8-bit): 27.8ms (61%)
- Talker + projection + feedback: 3.3ms (7%)

## Known Issues

- `OrderedVocab` tokenizer warnings printed to stderr are cosmetic (from Rust tokenizer, cannot be suppressed)
- Pre-converted MLX models have embedding quantization issues for STT — use `--hf-model` for STT tasks
- `kernels` package in pyproject.toml is a dead dependency (not imported in source)
- Flash Attention not available on macOS; SDPA works fine for both MPS and MLX
- Other LLMs running on the same machine will compete for memory bandwidth and slow MLX inference significantly
- Voice is random each TTS generation without speaker conditioning (expected — model samples from voice space)

## Compatibility

Verified working:
- MLX 0.31+ on Apple Silicon (M-series)
- PyTorch 2.11 MPS backend (SDPA, float16, bfloat16)
- Python 3.11-3.13
- All CUDA guards in codebase are conditional (`torch.cuda.is_available()`)
- No hard CUDA dependencies in the import chain
