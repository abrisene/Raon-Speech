# Raon-Speech

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/Raon-Speech-Gradient-White.png">
    <img src="assets/Raon-Speech-Gradient-Black.png" alt="Raon-Speech Logo" width="360">
  </picture>
</div>
<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/Raon-Speechchat-Gradient-White.png">
    <img src="assets/Raon-Speechchat-Gradient-Black.png" alt="Raon-SpeechChat Logo" width="360">
  </picture>
</div>

<p align="center">
  <a href="https://www.krafton.ai/ko/"><img src="https://img.shields.io/badge/Homepage-KRAFTON%20AI-blue?style=flat&logo=google-chrome&logoColor=white" alt="Homepage"></a>
  <a href="https://github.com/krafton-ai/Raon-Speech"><img src="https://img.shields.io/badge/GitHub-Raon%20Speech-white?style=flat&logo=github&logoColor=black" alt="GitHub"></a>
  <a href="https://huggingface.co/KRAFTON"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-KRAFTON-yellow?style=flat" alt="Hugging Face"></a>
  <a href="https://x.com/Krafton_AI"><img src="https://img.shields.io/badge/X-KRAFTON%20AI-white?style=flat&logo=x&logoColor=black" alt="X"></a>
</p>

**Links**
- GitHub: https://github.com/krafton-ai/Raon-Speech
- Official Demo: https://raon.krafton.ai
- Hugging Face Org: https://huggingface.co/KRAFTON
- Speech model card: https://huggingface.co/KRAFTON/Raon-Speech-9B
- SpeechChat (Full-Duplex) model card: https://huggingface.co/KRAFTON/Raon-SpeechChat-9B
- Technical Report: https://huggingface.co/KRAFTON/Raon-Speech-9B/resolve/main/Technical_Report_Raon_Speech.pdf

Raon is a speech model built on HuggingFace Ecosystem.  
This repo contains two tracks:

- Raon-Speech (Offline SpeechLM): `TTS`, `STT`, `SpeechChat`, `TextQA`
- Raon-SpeechChat (Offline/Realtime Full-Duplex)

Both tracks share the same core model family and processor stack under `src/raon/`.

## Key Features

- **SpeechLLM Model:** Raon-Speech is a 9B bilingual (English/Korean) SpeechLM for speech understanding, answering, and generation.
- **Full-Duplex Model:** Raon-SpeechChat extends Raon-Speech to natural real-time full-duplex conversation and shows strong interaction quality, especially on turn-taking, backchanneling, and interruption handling.
- **Training Scale:** Raon-Speech is trained on 1M+ hours of curated English-Korean speech-text data and achieves state-of-the-art average performance across 42 speech and text benchmarks against similarly sized baselines. Raon-SpeechChat is continually trained on 116K hours of time-aligned dialogue data.
- **System Design:** The system is built with a staged LLM-to-SpeechLM training recipe and a full-duplex design based on causal streaming, interleaved speech-text modeling, explicit interaction-state modeling, and text lookahead.
- **Task Coverage:** Unified multi-task support for `STT`, `TTS`, `TextQA`, and `SpeechChat`, with optional speaker-conditioned TTS and TTS continuation from reference audio.
- **Transformers Integration:** Hugging Face Transformers integration via `AutoModel.from_pretrained(..., trust_remote_code=True)`.
- **Open Release:** We open-source model checkpoints, the training and inference pipeline, an interactive demo, and three Korean speech benchmarks: KVoiceBench, KOpenAudioBench, and KMMAU.

## Benchmark Results

Overall performance comparison of Raon-Speech and Raon-SpeechChat against baseline models across diverse benchmarks.

<div align="center">
  <img src="https://huggingface.co/KRAFTON/Raon-Speech-9B/resolve/main/assets/raon-speech-speechchat.png" alt="Raon-Speech Benchmark Results" width="800">
</div>

Raon-Speech is optimized for low-latency, real-time speech generation while maintaining strong performance across ASR, speech generation, spoken QA, audio understanding, and text QA tasks. In the benchmarks above, Raon-Speech shows consistently high cross-domain scores, while Raon-SpeechChat performs strongly on conversational speech capabilities such as pause handling, backchanneling, smooth turn-taking, interruption handling, overlap robustness, and multi-turn dialogue. 

On single-GPU streaming TTS setups, the model runs faster than real time on both RTX 6000 Pro and L40S, with sub-second time-to-first-token latency. Measured with LibriSpeech `test-clean` samples on single-GPU setups via streaming TTS. All values are averaged.

| Metric | RTX 6000 Pro Blackwell | L40S |
|---|---:|---:|
| `RTF` | `0.27` (`3.7x` real-time) | `0.45` (`2.2x` real-time) |
| `TTFT` | `617 ms` | `887 ms` |
| `TBT` | `135 ms` | `233 ms` |

- `RTF`: Real-Time Factor. Lower is faster; values below `1.0` indicate faster-than-real-time synthesis.
- `TTFT`: Time to First Token.
- `TBT`: Time Between Tokens.

## Apple Silicon (MLX) Port

This fork adds a pure-MLX port of Raon-Speech / Raon-SpeechChat targeting
Apple Silicon. All four tasks run on-device on M-series Macs:

| Task | Speed | Notes |
|---|---|---|
| TTS (hybrid quant) | **2.6× real-time** (RTF 0.38) | M5 Max, 9B model |
| STT | **6.5× real-time** | 1.2 s decode for 7.8 s audio |
| VoiceChat (STT → response → TTS) | **Faster than real-time** | chained pipeline |
| **SpeechChat (full-duplex realtime)** | **RTF 0.78** | M-series, 8-bit, ~16 ms headroom under the 80 ms frame budget |

Quick start:

```bash
# Install with the mlx extra (CPU + Metal; pulls fastapi/gradio/uvicorn for the demo)
uv sync --extra mlx --extra dev

# Pre-convert SpeechChat-9B to uniform 8-bit (10.32 GB, fast load):
python -m raon_mlx.utils.convert models/Raon-SpeechChat-9B \
    --output models/Raon-SpeechChat-9B-mlx-8bit --quant 8bit

# Realtime full-duplex Gradio demo on http://127.0.0.1:7862 :
python demo/gradio_mlx_duplex_demo.py \
    --model-path models/Raon-SpeechChat-9B-mlx-8bit \
    --hf-model-path models/Raon-SpeechChat-9B \
    --quantize 8bit

# Offline duplex run on a 24 kHz mono wav:
python scripts/run_offline_test.py

# Deterministic per-frame bench (8 runs, fixed seed):
PYTHONPATH=src python scripts/bench_duplex.py --runs 8

# TTS (CLI):
python -m raon_mlx.tts "Hello there." -o out.wav --model models/Raon-Speech-9B
```

The MLX port lives entirely under `src/raon_mlx/`; the upstream PyTorch
implementation under `src/raon/` is untouched. See
[`docs/README.md`](docs/README.md) for the doc index,
[`docs/mlx-port-roadmap.md`](docs/mlx-port-roadmap.md) for the architecture
notes and per-frame performance profile (including the 2026-04-28 perf
pass that took duplex from RTF ~0.88 to ~0.78), and
[`docs/mlx-test-checklist.md`](docs/mlx-test-checklist.md) for the
acceptance tests.

For full-duplex use, prefer **uniform 8-bit** (`--quantize 8bit`); 4-bit
thinker errors compound across the multi-frame loop and produce gibberish
audio. Single-utterance TTS works fine with the hybrid quant default.

Known soft-edge: rapid Stop→Start session cycles in the realtime demo can
hit a Metal command-buffer race (`SIGSEGV` exit 139). Wait ~1 s between
sessions or refresh the page.

## Requirements

- Python `>=3.11`
- CUDA GPU recommended (`bfloat16` / `float16`)
- PyTorch + Torchaudio matching your CUDA environment

## Model Loading (Local or Hugging Face Hub)

All model entry points accept either:

- local checkpoint directory
- Hugging Face `repo_id` (downloaded automatically by `from_pretrained`)

Examples:

```bash
# Edit preset variables in the script first, then run:
bash scripts/infer.sh
# or
bash scripts/duplex_infer.sh
```

```python
from raon import RaonPipeline

pipe = RaonPipeline("KRAFTON/Raon-Speech-9B", device="cuda", dtype="bfloat16")
# or
pipe = RaonPipeline("KRAFTON/Raon-SpeechChat-9B", device="cuda", dtype="bfloat16")
```

If you want to pre-download first:

```bash
hf download KRAFTON/Raon-Speech-9B --local-dir /path/to/model_dir
# or
hf download KRAFTON/Raon-SpeechChat-9B --local-dir /path/to/model_dir
```

## Execution Modes

### Mode 1: `raon` installed (recommended)

After `pip install -e .` (or `uv sync`), all entry points are supported:

- `scripts/infer.sh`, `scripts/duplex_infer.sh`
- `scripts/train.sh`, `scripts/duplex_train.sh`
- `demo/run_gradio_demo.sh`, `demo/run_gradio_duplex_demo.sh`
- Python API: `from raon import RaonPipeline`

### Mode 2: without `raon` install

Supported:

- Raon-Speech Gradio demo (`demo/gradio_demo.py`) has a fallback that loads Hub remote code when `raon` import fails.
  Run directly from HF repo:

```bash
# from repo root
bash demo/run_gradio_demo.sh
```

- Pure Transformers flow using Hub remote code (advanced):
  `AutoModel.from_pretrained(..., trust_remote_code=True)`.

  See examples: `examples/message_example.ipynb` and `examples/duplex_example.ipynb`.
  Minimal pipeline load without `raon` install:

```python
import torch
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

MODEL_ID = "KRAFTON/Raon-Speech-9B"

config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
RaonPipeline = get_class_from_dynamic_module(
    "modeling_raon.RaonPipeline",
    MODEL_ID,
    revision=getattr(config, "_commit_hash", None),
)

pipe = RaonPipeline(MODEL_ID, device="cuda", dtype="bfloat16")
```

Not supported as-is:

- `python -m raon.*` module commands (used by `scripts/*.sh`)
- Raon-SpeechChat (Full-duplex) realtime demo runtime (`demo/gradio_duplex_demo.py`) because runtime code imports `raon.*` modules.

## Environment Setup

### Option A: `venv` + `pip`

```bash
git clone https://github.com/krafton-ai/Raon-Speech
cd Raon-Speech
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Option B: `uv`

```bash
git clone https://github.com/krafton-ai/Raon-Speech
cd Raon-Speech
uv sync
```

### Demo dependencies (optional)

The realtime full-duplex demo needs extra packages (`sglang`, `gradio`, `fastapi`, `uvicorn`):

```bash
pip install -e ".[demo]"
# or
uv sync --extra demo
```

### FlashAttention (optional)

If you want to use FlashAttention during training, install it separately:

```bash
pip install flash-attn
```

## Project Layout

```text
Raon-Speech/
├── src/raon/                 # package code
│   ├── models/               # RaonModel / RaonDuplexModel
│   ├── modules/              # audio encoder, tokenizer, speaker encoder, etc.
│   ├── utils/                # processor, datasets, losses, prompts, special tokens
│   ├── train.py              # SpeechLLM training entry
│   ├── duplex_train.py       # Full-duplex training entry
│   ├── generate.py           # SpeechLLM JSONL inference entry
│   ├── duplex_generate.py    # Full-duplex inference entry
│   └── pipeline.py           # high-level API (RaonPipeline)
├── scripts/                  # shell wrappers
├── demo/                     # Gradio demos
├── config/                   # inference configs
├── data/                     # sample datasets
└── examples/                 # notebooks and scripts
```

## Model Architecture

- One shared backbone: `RaonModel` (LM backbone + audio encoder + Mimi codec path).
- Two model types:
  `raon` (Raon-Speech) and `raon_duplex` (Raon-SpeechChat).
- Main trainable blocks include text/audio alignment and audio code prediction
  (`input_adaptor`, `output_adaptor`, `audio_lm_head`, `proj_code`, `code_predictor`).

## SpeechLLM

### Data Format (JSONL)

Each line is one sample.

| Field | Type | Description |
|---|---|---|
| `conversations` | `list[dict]` | turns with `from` (`human`/`gpt`) and `value` |
| `audios` | `list[str]` | audio paths consumed by `<audio>` tags in order |
| `speaker_ref_audios` | `list[str]` | optional speaker reference audio for TTS |
| `channel` | `str` | `tts`, `stt`, `speech-chat`, `textqa` |
| `system` | `str` | optional system prompt |

Sample eval data is under `data/speechllm/eval`.

### Inference

```bash
bash scripts/infer.sh
```

- task defaults come from `config/infer.yaml`
- edit the preset variables at the top of `scripts/infer.sh` for quick runs
- `--attn_implementation` controls attention backend (default `sdpa`; use `fa` for FlashAttention)
- `--data_dir` is scanned for JSONL files, and each line is treated as one sample.

Advanced CLI example:

```bash
python -m raon.generate \
  --model_path /path/to/model \
  --data_dir /path/to/data_dir \
  --output_dir /path/to/output_dir \
  --config /path/to/config.yaml \
  --batch_size 4 \
  --attn_implementation sdpa
```

### Pipeline API

```python
from raon import RaonPipeline

pipe = RaonPipeline("/path/to/model", device="cuda", dtype="bfloat16", config="/path/to/config.yaml")

text = pipe.stt("/path/to/stt.wav")
audio, sr = pipe.tts("Hello!", speaker_audio="/path/to/speaker_ref.wav")
ans1 = pipe.speech_chat("/path/to/speech-chat.wav")
ans2 = pipe.textqa("What did the speaker say?", audio="/path/to/textqa.wav")

pipe.save_audio((audio, sr), "/path/to/tts.wav")
```

Continuation TTS is also supported:

```python
audio, sr = pipe.tts_continuation(
    target_text="Continue this sentence.",
    ref_audio="/path/to/speaker_ref.wav",
    ref_text="Optional transcription of ref audio.",
)
```

See:

- `examples/message_example.py`
- `examples/message_example.ipynb`

### Training

```bash
bash scripts/train.sh
```

Common options:

- edit the preset variables at the top of `scripts/train.sh`
- `NPROC_PER_NODE` controls multi-GPU torchrun
- `MASTER_PORT` sets the torchrun rendezvous port
- `USE_SPEAKER_EMBEDDING` toggles speaker-conditioning inputs

Notes:

- Current training code freezes these modules: `audio_encoder`, `input_adaptor`, `output_adaptor`, `audio_lm_head`, `proj_code`, `code_predictor`, `speaker_encoder`
- Default training attention implementation is `sdpa`; to use FlashAttention, pass `--attn_implementation fa`

## Full-Duplex

### Data Format (JSONL)

Training (`duplex_train.py`) expects one JSON object per line with stereo audio metadata.

Required top-level fields:

| Field | Type | Description |
|---|---|---|
| `audio_path` | `str` | Path to stereo wav (`2` channels). |
| `language` | `str` | Language code (for prompt selection), e.g. `eng`, `kor`. |
| `channel` | `str` | Duplex channel type, e.g. `full_duplex` or `duplex_instruct`. |
| `speak_first` | `list[int or bool]` | Per-channel initial speaking mode for the two channels. `1`/`true` means that channel is treated as speaking first; `0`/`false` means listen-first. |
| `include_in_training` | `list[int or bool]` | Per-channel training mask for the two channels. `1`/`true` includes that channel's targets in loss computation; `0`/`false` excludes them. |
| `turns` or `scripts` | `list` | Conversation timing/transcript annotations (one of the two formats below). |

Supported annotation format A (`turns`):

- Use this format when each utterance is already represented as a turn.
- Each turn contains `channel`, `start_sample`, `end_sample`, and `ipus`.
- Each IPU contains `words`, and each word has `word`, `start_sample`, `end_sample`.

Supported annotation format B (`scripts`):

- Use this format when you have per-channel word timestamps instead of explicit turns.
- `scripts` is `[ch0_words, ch1_words]`, and each word item has `word`, `start`, `end` in seconds.
- You can optionally provide `timeline` or `rough_timeline` for utterance boundaries.
- If those boundaries include `channel`, they are used directly.
- Otherwise, the loader derives per-channel utterance boundaries from the word timestamps in `scripts`.

Optional fields:

- `sample_rate` (used for `turns` timestamps; defaults to `24000` if omitted)
- `system_prompt` (if omitted, prompt is built from persona/context/name metadata when available)

Optional inference metadata sidecar:

- You can place a same-stem `.jsonl` file next to the input wav for prompt and speaker metadata.
- This is an input metadata format, not a separate CLI argument schema.

| Field | Type | Description |
|---|---|---|
| `audio_path` | `str` | Input audio path metadata (optional for CLI run, used in dataset-style records). |
| `speak_first` | `bool` | Override initial speaking mode. |
| `language` | `str` | Optional language hint. |
| `persona` | `str` | Persona key/text for prompt builder. |
| `context` | `str` | Extra context appended to system prompt. |
| `system_prompt` | `str` | Explicit prompt override. |
| `speaker_audio` | `str` | Speaker reference wav path (preferred key). |
| `speaker_ref_audios` | `list[str]` | Fallback speaker reference list (first element used). |

Sample data:

- train: `data/duplex/train`
- eval: `data/duplex/eval`
- persona catalog: `data/duplex/personas.json`

Training data uses timeline JSONL + stereo audio assets.  

### Inference

```bash
bash scripts/duplex_infer.sh
```

- edit the preset variables at the top of `scripts/duplex_infer.sh` for quick runs
- `--attn_implementation` controls attention backend (default `sdpa`; use `fa` for FlashAttention)
- `--data_dir` is scanned recursively for `*.jsonl`, and every line is treated as one sample.

Advanced CLI example:

```bash
python -m raon.duplex_generate \
  --model_path /path/to/model \
  --data_dir /path/to/data_dir \
  --output_dir /path/to/output_dir \
  --config /path/to/config.yaml \
  --speaker_audio /path/to/speaker_ref.wav \
  --persona /path/or/persona_key \
  --context "Optional extra context"
```

- `data_dir` is scanned recursively for `*.jsonl`, and CLI arguments override per-record metadata when both are provided.

### Pipeline API

```python
from raon import RaonPipeline

pipe = RaonPipeline("/path/to/duplex-model", device="cuda", dtype="bfloat16")
audio = pipe.load_audio("/path/to/input.wav")

summary = pipe.duplex(
    audio_input=audio,
    output_dir="/path/to/output_dir",
    speak_first=False,
    system_prompt="You are engaging in real-time conversation.",
    speaker_audio="/path/to/speaker_ref.wav",
)
print(summary)
```

See:

- `examples/duplex_example.ipynb`

### Training

```bash
bash scripts/duplex_train.sh
```

Common options:

- edit the preset variables at the top of `scripts/duplex_train.sh`
- `NPROC_PER_NODE` controls multi-GPU torchrun
- `MASTER_PORT` sets the torchrun rendezvous port
- `BATCH_SIZE` is fixed to `1`

Notes:

- same freeze list pattern as SpeechLLM training
- Default training attention implementation is `sdpa`; to use FlashAttention, pass `--attn_implementation fa`

## Gradio Demos

### Raon-Speech demo

```bash
bash demo/run_gradio_demo.sh
```

- edit the preset variables at the top of `demo/run_gradio_demo.sh`

### Raon-SpeechChat realtime demo

1. Export HF checkpoint to SGLang bundle:

```bash
bash scripts/export.sh
```

- edit the preset variables at the top of `scripts/export.sh`

2. Run demo:

```bash
bash demo/run_gradio_duplex_demo.sh
```

- edit the preset variables at the top of `demo/run_gradio_duplex_demo.sh`

## License

Apache License 2.0. See `LICENSE`.

## Citation

```bash
@misc{raonspeech,
    title  = {Raon-Speech Technical Report},
    author = {{KRAFTON}},
    month  = {April},
    year   = {2026}
}
```

