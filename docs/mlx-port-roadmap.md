# Raon-Speech MLX Port Roadmap

## Goal

Port Raon-Speech inference to Apple MLX for quantized, high-performance inference on Apple Silicon. Target: 4-bit quantized model running at or near real-time TTS on M-series chips.

## Current State

- MPS (PyTorch Metal backend) works at RTF 2.52 (0.4x real-time) on M5 Max with float16
- Models downloaded: Raon-Speech-9B, Raon-SpeechChat-9B
- Fork: `git@github.com:abrisene/Raon-Speech.git`

## Prior Art: PersonaPlex MLX Port

We have a complete MLX port of Moshi (Kyutai) at `/Users/dr/Models/Clients/personaplex/`. Moshi shares critical architecture with Raon — most importantly the Mimi audio codec. The PersonaPlex kernel provides:

### Reusable Components (from `personaplex_kernel/`)

| Component | Files | Applicability to Raon |
|-----------|-------|----------------------|
| **Mimi codec** | `models/mimi.py` | Direct reuse — same Mimi architecture (24kHz, 12.5Hz, Seanet, 2048 codebook) |
| **Seanet encoder/decoder** | `modules/seanet.py` | Direct reuse — streaming conv encoder/decoder |
| **Streaming Conv1d/ConvTranspose1d** | `modules/conv.py` | Direct reuse — causal padding, streaming step() |
| **SplitResidualVectorQuantizer** | `modules/quantization.py` | Direct reuse — same VQ architecture |
| **Transformer + KV cache** | `modules/transformer.py` | Adapt — needs GQA support for Qwen3 |
| **RoPE** | `modules/transformer.py` | Reusable — already in the Attention class |
| **Generation loop** | `models/generate.py` | Template — Raon's loop is simpler (no dual-stream) |
| **PyTorch weight loading** | `models/mimi.py`, `models/lm.py` | Template — conv weight transposition patterns |

### Key Patterns Already Solved

1. **Conv weight layout**: PyTorch `(outC, inC, kSize)` → MLX `(outC, kSize, inC)` via `swapaxes(-1, -2)`
2. **ConvTranspose weight layout**: PyTorch `(inC, outC, kSize)` → MLX `(outC, kSize, inC)` via `transpose(1, 2, 0)`
3. **Streaming state management**: `reset_state()` / `step()` pattern on all conv and transformer modules
4. **EuclideanCodebook**: Custom `update_in_place()` for precomputing embeddings after weight load
5. **KV cache**: Both standard and rotating variants

## Raon Architecture (from config.json)

### text_model (Thinker) — The Backbone (~8B params)
- **Architecture**: Qwen3 (`Qwen3ForCausalLM`)
- hidden_size: 4096
- num_attention_heads: 32
- num_key_value_heads: 8 (GQA 4:1)
- num_hidden_layers: 36
- intermediate_size: 12288 (SwiGLU)
- head_dim: 128
- rope_theta: 5000000
- vocab_size: 153723

### talker_config (4 Qwen3 layers, ~500M params)
- hidden_size: 2048
- num_attention_heads: 16
- num_key_value_heads: 8 (GQA 2:1)
- num_hidden_layers: 4
- intermediate_size: 6144
- Connected to thinker via MLP projection (4096 → 2048)

### code_predictor_config (~200M params)
- Model type: `qwen3_omni_moe_talker_code_predictor`
- hidden_size: 1024
- num_attention_heads: 16
- num_key_value_heads: 8
- num_hidden_layers: 5
- num_code_groups: 16
- vocab_size: 2048 (codebook size)

### audio_tokenizer_config (Mimi codec, ~30M params)
- **Source**: `kyutai/mimi` — same as PersonaPlex
- sample_rate: 24000
- frame_rate: 12.5
- codebook_size: 2048
- num_quantizers: 32 (1 semantic + 31 acoustic)
- hidden_size: 512
- num_filters: 64
- upsampling_ratios: [8, 6, 5, 4]
- Transformer: 8 layers, 8 heads, sliding_window: 250

### audio_encoder_config (AuT, ~300M params)
- Model type: `qwen3_omni_moe_audio_encoder`
- Whisper-like: 24 encoder layers, d_model: 1024, 16 heads
- output_dim: 2048
- sampling_rate: 24000, num_mel_bins: 128

### input_adaptor (~17M params)
- 2-layer MLP: 2048 → 4096

### output_adaptor (~4M params)
- 2-layer MLP: 512 → 4096

### speaker_encoder (ECAPA-TDNN, ~10M params)
- Pretrained: `speechbrain/spkrec-ecapa-voxceleb`
- pretrained_dim: 192 → output_size: 4096

## Port Plan

### Phase 1: Mimi Codec (1-2 days)

**Goal**: Raon's Mimi weights loading into PersonaPlex's MLX Mimi.

The architectures are identical. The difference is weight key naming:
- PersonaPlex loads from Kyutai's PyTorch format
- Raon stores Mimi weights inside the HF safetensors checkpoint under the `audio_tokenizer.` prefix

**Work**:
1. Copy PersonaPlex's `modules/conv.py`, `modules/seanet.py`, `modules/quantization.py`, `models/mimi.py` into a new `src/raon_mlx/` package
2. Write a weight extraction script that loads Raon's safetensors, filters `audio_tokenizer.*` keys, and maps them to the MLX Mimi's key names
3. Verify: encode a wav, decode it, compare output to PyTorch Mimi output (should be numerically close)

**Risk**: Raon may have modified the Mimi config slightly (e.g., `num_quantizers: 32` vs Moshi's typical 8-16). The architecture is the same but the VQ may have more codebooks. PersonaPlex's `SplitResidualVectorQuantizer` is parameterized by `nq` so this should just work.

### Phase 2: Qwen3 Backbone in MLX (2-3 days)

**Goal**: The 36-layer Qwen3 thinker running in MLX with KV cache.

**Approach**: Use `mlx-lm`'s existing Qwen3 implementation as reference, but build a standalone module (not the full `mlx-lm` generate pipeline) that can be composed with the other Raon components.

**Key differences from PersonaPlex's transformer**:
- **GQA**: num_kv_heads=8, num_heads=32 (PersonaPlex only had kv_repeat=1)
- **SwiGLU FFN**: gate_proj, up_proj, down_proj (PersonaPlex has similar gating but different weight naming)
- **RMSNorm** (same as PersonaPlex)
- **RoPE with theta=5M** (PersonaPlex uses 10K-100K)
- **No sliding window** on the backbone (simpler than Mimi's transformer)

**Work**:
1. Implement `Qwen3Attention` with GQA support (extend PersonaPlex's `Attention` class)
2. Implement `Qwen3MLP` with SwiGLU (gate_proj * silu × up_proj → down_proj)
3. Implement `Qwen3DecoderLayer` and `Qwen3Model` (36 layers + RMSNorm)
4. Write weight mapping from HF `text_model.*` keys
5. Verify: compare hidden states on a test input against PyTorch output

**Quantization**: This is the biggest module. Apply 4-bit quantization here using `mlx.nn.quantize()`. The 8B backbone drops from ~16GB to ~4.5GB at 4-bit.

### Phase 3: Talker + Code Predictor (1-2 days)

**Goal**: The audio code generation pipeline in MLX.

The talker is just 4 more Qwen3 layers (smaller hidden size 2048). The code predictor is 5 layers at hidden_size 1024 with a per-codebook sampling loop — analogous to PersonaPlex's DepFormer.

**Work**:
1. Reuse Qwen3 layer implementation from Phase 2 (different config)
2. Implement `ThinkerToTalkerProjection` (MLP: 4096 → 2048)
3. Implement `RaonCodePredictor` — 5-layer transformer that sequentially predicts 16 codebook entries
4. Write weight mapping for `talker.*`, `proj_code.*`, `code_predictor.*` keys
5. Implement the `audio_lm_head` (linear projection to codebook logits)

### Phase 4: Generation Loop (1-2 days)

**Goal**: End-to-end TTS and STT inference in MLX.

The generation loop is in `src/raon/models/wrapper.py`:
1. `_generation_prefill`: process input tokens, build KV cache
2. `_decoding_step` (hot loop): run thinker forward → sample text token → if audio: run talker → run code predictor → sample audio codes
3. `decode_audio`: feed accumulated audio codes through Mimi decoder → PCM

**Work**:
1. Implement prefill (process input_ids, build initial KV cache state)
2. Implement the autoregressive loop with text/audio token switching logic
3. Implement sampling (temperature, top-k, top-p, repetition-aware sampling)
4. Wire up Mimi decode at the end
5. Benchmark: measure tokens/sec and RTF

### Phase 5: Audio Encoder (1-2 days, can defer)

**Goal**: STT and SpeechChat support.

The audio encoder is a Whisper-like architecture (AuT wrapper). This is NOT needed for TTS — only for tasks that take audio input (STT, SpeechChat, TextQA with audio).

**Work**:
1. Implement the Whisper-like encoder in MLX (24 layers, 1024 dim, 16 heads)
2. Implement `input_adaptor` (2-layer MLP: 2048 → 4096)
3. Write weight mapping for `audio_encoder.*` and `input_adaptor.*` keys

**Can defer**: TTS is the primary use case. Audio input tasks can stay on PyTorch/MPS initially.

### Phase 6: Speaker Encoder (optional, can defer)

**Goal**: Voice-conditioned TTS.

ECAPA-TDNN from SpeechBrain. Small model (~10M params), runs once per utterance. Not on the critical path.

**Options**:
- Keep in PyTorch/MPS (single boundary crossing per utterance — negligible overhead)
- Port to MLX later if desired

### Phase 7: SpeechChat / Full-Duplex (stretch goal)

The Raon-SpeechChat-9B model (`RaonDuplexModel`) adds real-time duplex capabilities. This is a separate model with additional architecture on top. Tackle after the base model works.

## Expected Performance

### 4-bit Quantized on M5 Max (128GB)

- **Model size**: ~4.5GB (backbone) + ~1GB (other components) ≈ 5.5GB
- **Memory bandwidth**: ~546 GB/s (M5 Max)
- **Theoretical throughput**: 546 / 5.5 ≈ ~100 tokens/sec for backbone
- **Realistic (with overhead)**: ~40-60 tokens/sec
- **At 12.5 audio frames/sec**: likely **faster than real-time TTS**
- **Target RTF**: < 1.0 (real-time capable)

### Comparison

| Setup | RTF | Real-time? |
|-------|-----|-----------|
| RTX 6000 Pro (KRAFTON benchmark) | 0.27 | Yes (3.7x) |
| MPS float16 (current, M5 Max) | 2.52 | No (0.4x) |
| MLX 4-bit quantized (projected) | 0.3-0.8 | Likely yes |

## File Structure

```
src/raon_mlx/
├── __init__.py
├── models/
│   ├── mimi.py          # Mimi codec (from PersonaPlex)
│   ├── qwen3.py         # Qwen3 backbone
│   ├── raon.py          # Full RaonModel composition
│   └── generate.py      # Generation loop
├── modules/
│   ├── conv.py          # Streaming conv ops (from PersonaPlex)
│   ├── seanet.py        # Seanet encoder/decoder (from PersonaPlex)
│   ├── quantization.py  # VQ (from PersonaPlex)
│   ├── transformer.py   # Qwen3 attention + MLP
│   ├── kv_cache.py      # KV cache (from PersonaPlex)
│   ├── code_predictor.py
│   ├── adaptor.py       # Input/output adaptors
│   └── audio_encoder.py # AuT (Phase 5)
├── utils/
│   ├── sampling.py      # Token sampling
│   └── audio_io.py      # Audio I/O
├── convert_weights.py   # HF safetensors → MLX weight mapping
└── pipeline.py          # High-level API matching RaonPipeline
```

## Open Questions

1. **Mimi num_quantizers**: Raon uses 32 quantizers (1 semantic + 31 acoustic) but Moshi typically uses 8 or 16. PersonaPlex's VQ is parameterized, so this should work, but needs verification.
2. **Code predictor architecture**: The `qwen3_omni_moe_talker_code_predictor` model type may have MoE (mixture of experts) components. Need to inspect the actual code predictor implementation to confirm.
3. **Audio encoder necessity**: For TTS-only use, the audio encoder can be skipped entirely. Confirm this is acceptable for initial port.
4. **Quantization granularity**: 4-bit for the backbone, but what about the talker (4 layers), code predictor (5 layers), and Mimi transformer (8 layers)? These are small enough that quantization may hurt quality more than it helps speed.
