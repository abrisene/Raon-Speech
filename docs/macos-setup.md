# Raon-Speech macOS Setup

## Environment

- **Machine**: Apple M5 Max, 128GB RAM
- **Python**: 3.13 via uv
- **PyTorch**: 2.11.0 (MPS backend)
- **Models downloaded**: `models/Raon-Speech-9B/`, `models/Raon-SpeechChat-9B/`

## Setup Steps

```bash
# Create venv
uv venv --python 3.13 .venv
source .venv/bin/activate

# Install deps
uv pip install -r requirements.txt -e .
uv pip install 'huggingface_hub[cli]'

# Download models (requires HF login: `hf auth login`)
hf download KRAFTON/Raon-Speech-9B --local-dir models/Raon-Speech-9B
hf download KRAFTON/Raon-SpeechChat-9B --local-dir models/Raon-SpeechChat-9B
```

## Running on MPS

The pipeline accepts `device="mps"` directly:

```python
from raon import RaonPipeline

pipe = RaonPipeline("models/Raon-Speech-9B", device="mps", dtype="float16", attn_implementation="sdpa")

# TTS
audio, sr = pipe.tts("Hello world!")
pipe.save_audio((audio, sr), "output/test.wav")

# STT
text = pipe.stt("input.wav")
```

## Benchmark (M5 Max, float16, MPS)

- **Model load time**: ~18s (4 safetensors shards)
- **TTS**: 16.2s of audio generated in 40.9s → **RTF 2.52** (0.4x real-time)
- **Steady-state throughput**: ~7 tokens/sec after warmup
- **Memory**: ~18GB for float16 model

For comparison, KRAFTON reports RTF 0.27 (3.7x real-time) on RTX 6000 Pro.

## Known Issues

- Tokenizer warnings about regex pattern and special token mapping are cosmetic
- `OrderedVocab` holes warning is expected (audio special tokens)
- `torch_dtype` deprecation warnings from HF internals
- `kernels` package in pyproject.toml is a dead dependency (not imported in source)
- Flash Attention not available on MPS; SDPA works fine
- Mimi codec's sliding window attention falls back to SDPA (may affect audio >20s)

## MPS Compatibility

Verified working:
- SDPA attention on MPS
- float16 on MPS
- bfloat16 on MPS (PyTorch 2.11+)
- All CUDA guards in codebase are conditional (`torch.cuda.is_available()`)
- No hard CUDA dependencies in the import chain
