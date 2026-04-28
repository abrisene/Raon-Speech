# Raon-Speech MLX Port Roadmap

## Goal

Port Raon-Speech inference to Apple MLX for quantized, high-performance inference on Apple Silicon. Target: 4-bit quantized model running at or near real-time TTS on M-series chips.

## Current State (Updated 2026-04-04)

**MLX port is functional.** End-to-end TTS working at 2.3x real-time on M5 Max.

- Branch: `feat/mlx-port` (12 commits)
- Fork: `git@github.com:abrisene/Raon-Speech.git`
- Pre-converted model: `models/Raon-Speech-9B-mlx-hybrid` (6.46 GB, hybrid quant)

### What's Working
- TTS generation (text → speech)
- Speaker conditioning (voice cloning from reference audio)
- Model conversion (HF → quantized MLX safetensors)
- Streaming callback support (per-frame PCM output)
- CLI: `python -m raon_mlx.tts`

### What's Not Yet Implemented
- STT (needs audio encoder — Phase 5)
- SpeechChat / TextQA (needs audio encoder)
- Full-duplex / Raon-SpeechChat-9B model
- Repetition-aware sampling (RAS)
- TTS continuation from reference audio

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

### Phase 1: Mimi Codec ✅

**Goal**: Raon's Mimi weights loading into PersonaPlex's MLX Mimi.

The architectures are identical. The difference is weight key naming:
- PersonaPlex loads from Kyutai's PyTorch format
- Raon stores Mimi weights inside the HF safetensors checkpoint under the `audio_tokenizer.` prefix

**Work**:
1. Copy PersonaPlex's `modules/conv.py`, `modules/seanet.py`, `modules/quantization.py`, `models/mimi.py` into a new `src/raon_mlx/` package
2. Write a weight extraction script that loads Raon's safetensors, filters `audio_tokenizer.*` keys, and maps them to the MLX Mimi's key names
3. Verify: encode a wav, decode it, compare output to PyTorch Mimi output (should be numerically close)

**Risk**: Raon may have modified the Mimi config slightly (e.g., `num_quantizers: 32` vs Moshi's typical 8-16). The architecture is the same but the VQ may have more codebooks. PersonaPlex's `SplitResidualVectorQuantizer` is parameterized by `nq` so this should just work.

### Phase 2: Qwen3 Backbone in MLX ✅

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

### Phase 3: Talker + Code Predictor ✅

**Goal**: The audio code generation pipeline in MLX.

The talker is just 4 more Qwen3 layers (smaller hidden size 2048). The code predictor is 5 layers at hidden_size 1024 with a per-codebook sampling loop — analogous to PersonaPlex's DepFormer.

**Work**:
1. Reuse Qwen3 layer implementation from Phase 2 (different config)
2. Implement `ThinkerToTalkerProjection` (MLP: 4096 → 2048)
3. Implement `RaonCodePredictor` — 5-layer transformer that sequentially predicts 16 codebook entries
4. Write weight mapping for `talker.*`, `proj_code.*`, `code_predictor.*` keys
5. Implement the `audio_lm_head` (linear projection to codebook logits)

### Phase 4: Generation Loop ✅

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

### Phase 5: Audio Encoder (not started)

**Goal**: STT and SpeechChat support.

The audio encoder is a Whisper-like architecture (AuT wrapper). This is NOT needed for TTS — only for tasks that take audio input (STT, SpeechChat, TextQA with audio).

**Work**:
1. Implement the Whisper-like encoder in MLX (24 layers, 1024 dim, 16 heads)
2. Implement `input_adaptor` (2-layer MLP: 2048 → 4096)
3. Write weight mapping for `audio_encoder.*` and `input_adaptor.*` keys

**Can defer**: TTS is the primary use case. Audio input tasks can stay on PyTorch/MPS initially.

### Phase 6: Speaker Encoder ✅ (hybrid approach)

**Goal**: Voice-conditioned TTS.

ECAPA-TDNN from SpeechBrain. Small model (~10M params), runs once per utterance. Not on the critical path.

**Options**:
- Keep in PyTorch/MPS (single boundary crossing per utterance — negligible overhead)
- Port to MLX later if desired

### Phase 7: SpeechChat / Full-Duplex ✅

`Raon-SpeechChat-9B` runs end-to-end on MLX with intelligible assistant audio at
real-time RTF (~1.0 on M-series). Duplex loop owned by
`src/raon_mlx/models/duplex_generate.py`; FastAPI + WebSocket runtime in
`src/raon_mlx/realtime/`; Gradio demo at `demo/gradio_mlx_duplex_demo.py`.

Key duplex-specific things worth knowing:

1. **8-bit uniform quant is the floor for duplex**, not hybrid. Hybrid (4-bit
   thinker) compounds error across the multi-frame loop and produces gibberish
   audio even though SIL frames render cleanly and per-forward parity vs PT is
   exact (cos 0.9999968). Use `quant="8bit"` for SpeechChat. Hybrid remains
   fine for TTS-only paths.
2. **SIL detection is phase-based**, not `emitted_audio` (which is always True
   because every chunk contains AUDIO_OUTPUT_PLACEHOLDER). During SIL, feed
   silence_codes both to Mimi `decode_step` and as `prev_audio_feedback` so
   the talker cache stays on a silence trajectory.
3. **Code predictor codebooks 2–16 are greedy**, matching PT's `predict_codes`.
   Only codebook 1 samples (with the same temperature as text).
4. RAS (repetition-aware sampling) is plumbed in `generate_audio_codes` via
   `ras_enabled=False` default; off matches PT's typical behavior.

See `docs/mlx-duplex-debug-log.md` for the full diagnostic trail and the
single-forward parity proofs (thinker / talker / code predictor / output
adaptor all match PT cos 0.99997+).

## Measured Performance (M5 Max, 128GB)

### Backbone-only (Qwen3 thinker, 36 layers)

| Precision | tok/s | Backbone RTF |
|-----------|-------|-------------|
| bfloat16 (no quant) | 7.0 | 1.80 |
| **4-bit quantized** | **69.4** | **0.18** |

### Full TTS Pipeline (end-to-end)

| Config | RTF | Speed | Notes |
|--------|-----|-------|-------|
| PyTorch MPS float16 | 2.52 | 0.4x | Baseline |
| MLX thinker=4bit only | 1.08 | 0.9x | Code predictor bottleneck |
| MLX all=4bit | 0.49 | 2.0x | |
| **MLX thinker=4bit, talker+cp=8bit** | **0.46** | **2.2x** | **Best quality/speed tradeoff** |
| RTX 6000 Pro (KRAFTON benchmark) | 0.27 | 3.7x | Datacenter GPU reference |

### Comparison with original projections

- Projected: RTF 0.3-0.8 → **Achieved: RTF 0.46** (within range, toward the fast end)
- The hybrid quantization (4-bit backbone, 8-bit audio components) is both faster and higher quality than uniform 4-bit **for TTS-only**.
  For full-duplex SpeechChat use **uniform 8-bit** (`quant="8bit"`); 4-bit thinker errors compound across many frames and produce gibberish audio. See Phase 7.

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

## Voice Shaping (Future Work)

Beyond speaker reference audio and voice seed, there are several approaches to control voice characteristics:

1. **Voice presets** — pre-compute speaker embeddings from diverse reference samples, save as named presets (e.g. "deep male", "bright female", "narrator"). Users pick from a dropdown.
2. **Embedding interpolation** — blend two preset embeddings: `0.7 * voice_A + 0.3 * voice_B` to create novel voices between two references.
3. **Embedding arithmetic** — vector operations on speaker embeddings: `narrator - male + female = female narrator` (analogous to word2vec arithmetic).
4. **Direct embedding editing** — map perceptual qualities (pitch, breathiness, pace, warmth) to embedding dimensions via PCA or supervised probing, then expose as sliders.
5. **Voice seed** — currently implemented. Same seed = reproducible voice. Different seeds sample different points in the model's voice prior space.

## Real-Time Conversation (Next)

### Architecture

For real-time voice conversation, the pipeline is:

1. **User speaks** → audio chunks → audio encoder (AuT) → embeddings
2. **Model thinks** → thinker processes audio embeddings → generates response
3. **Model speaks** → talker + code predictor → audio codes → Mimi decode → PCM
4. **Streaming** → output audio as it's generated, overlap with continued listening

### Turn-Based Conversation (stepping stone)

Before full duplex, implement turn-based: user speaks → model responds with audio.
This uses the existing STT + TTS pipeline chained together, with the SpeechChat task
providing the model's response generation.

### Full Duplex (Raon-SpeechChat-9B)

The SpeechChat model adds simultaneous listen/speak capability:
- Dual-stream audio processing (user + assistant channels)
- Explicit interaction state modeling (speaking, listening, backchanneling)
- Turn-taking and interruption handling
- Requires the duplex model architecture (separate from base Speech model)

## Duplex Performance Profile (8-bit, M-series)

Per-section ms per frame, averaged over 30 frames after 5-frame warmup
(`scripts/profile_duplex.py`):

| Section                       | avg ms | notes                                       |
|------------------------------|--------|---------------------------------------------|
| 01 encode_user (Voxtral)     |   1.0  | streaming MLX encoder + input adaptor       |
| 02 build_embeds              |   0.2  | embedding lookup + placeholder splice       |
| 03 thinker (36L Qwen3, 8b)   |   1.6  | growing KV cache; small input (2-3 tokens)  |
| 04 talker (4L Qwen3, 8b)     |   0.4  |                                             |
| **05 update_seq + audio**    | **62.4** | **15 sequential code-predictor steps**      |
| 06 mimi decode_step          |   0.5  |                                             |
| **TOTAL**                    | **66.0** | RTF ~0.88 on 28.4 s test                    |

**Section 05 dominates (95% of frame time).** Inside it, ~95% is the
autoregressive code predictor: 15 sequential 5-layer transformer steps to
predict codebooks 2–16 given codebook 1. The architecture is sequential by
construction (each code conditions on the previous), so this can't be
batched without retraining a parallel-prediction head.

Already-applied optimizations:
- Reuse the code-predictor KV cache across frames (was reallocating ~5680
  cache slots per 28 s run). Net 13% speed-up.

Further optimization candidates (not yet tried):
- `mx.compile` the per-step inner function. The mutable KV cache prevents the
  obvious wrapping; would need to thread cache state in/out as MLX arrays.
- Drop the `audio_lm_head` EOS column (suppress_eos always sets it to -1e9)
  to shrink the matmul from `[2049, 2048]` to `[2048, 2048]`. Trivial in fp16,
  needs a quantized-aware row slice for 8-bit weights.
- Custom Metal kernel for the per-step transformer block, fused. Out of
  scope for now but the highest-ceiling option.
- Pre-allocate KV caches at session start to avoid the 4-or-so reallocations
  during a long conversation. Tested at `step=2048`; cold-start cost
  outweighed the steady-state win for this workload.

Tried and reverted:
- KV cache contiguous-write fast-path: replacing the per-position Python loop
  with a single slice assignment was *slower*, presumably because MLX's lazy
  graph prefers smaller writes that fuse with later ops.
- Skipping the init second-pass forward (matching PT's flow exactly): clean
  win for code clarity but no measurable perf change, because the bottleneck
  is the sequential code-predictor loop, not the cache machinery.

### Cleanup tried, reverted (worth knowing for future maintainers)

We attempted to align with `omlx` / `mlx_lm`'s simpler cache pattern: drop
`cache_position`, drop the init second-pass forward, let the first duplex
frame just append. The audio still rendered correctly, but **steady-state
per-frame time regressed by ~14 ms** on the offline test (and the first
frame jumped to ~3 s). Our best guess: MLX's lazy graph compiles a
different (worse) plan when positions are inferred from `cache.offset`
rather than passed in explicitly, possibly because the offset read forces
a small synchronization that the explicit-position path avoids.

If a future MLX version improves graph compilation or we move to a single
`mx.compile`d step function, this regression should disappear and the
cleanup is worth revisiting.

## Open Questions

1. **Mimi num_quantizers**: Raon uses 32 quantizers (1 semantic + 31 acoustic) but Moshi typically uses 8 or 16. PersonaPlex's VQ is parameterized, so this should work, but needs verification.
2. **Code predictor architecture**: The `qwen3_omni_moe_talker_code_predictor` model type may have MoE (mixture of experts) components. Need to inspect the actual code predictor implementation to confirm.
3. **Audio encoder necessity**: For TTS-only use, the audio encoder can be skipped entirely. Confirm this is acceptable for initial port.
4. **Quantization granularity**: 4-bit for the backbone, but what about the talker (4 layers), code predictor (5 layers), and Mimi transformer (8 layers)? These are small enough that quantization may hurt quality more than it helps speed.
