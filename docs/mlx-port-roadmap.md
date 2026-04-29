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

### Headline numbers (after 2026-04-28 perf pass)

| Metric                  | Baseline | Op-level pass | Encoder bf16 | Encoder 8-bit |
|-------------------------|----------|---------------|--------------|---------------|
| Mean RTF (8 det. runs)  | ~0.88    | 0.78          | 0.70         | **0.66**      |
| Avg frame time          | ~70 ms   | 63.7 ms       | 55.9 ms      | **52.8 ms**   |
| Headroom under 80 ms    | ~10 ms   | ~16 ms        | ~24 ms       | **~27 ms (~34%)** |

Bench harness: `scripts/bench_duplex.py` — fixed seed, 3-frame warmup,
deterministic across runs (so the RTF is comparable; offline test RTF varies
with the sampled response length).

The fine-grained profiler `scripts/profile_duplex_fine.py` materializes
deferred MLX work at every section boundary (since `mx.synchronize` alone
doesn't trigger lazy evaluation). It surfaced the streaming Voxtral encoder
running at ~14 ms/frame in fp32 — that's the second-largest cost after the
thinker. Switching the encoder default dtype to bfloat16 (matching the rest
of the model) halves that to ~7 ms with no audio quality regression.

### Per-section profile (`scripts/profile_duplex.py`)

| Section                       | avg ms | notes                                       |
|------------------------------|--------|---------------------------------------------|
| 01 encode_user (Voxtral)     |   1.0  | streaming MLX encoder + input adaptor       |
| 02 build_embeds              |   0.2  | embedding lookup + placeholder splice       |
| 03 thinker (36L Qwen3, 8b)   |   1.6  | *(misleading — see lazy-eval note below)*   |
| 04 talker (4L Qwen3, 8b)     |   0.4  |                                             |
| **05 update_seq + audio**    | **57.7** | **dominated by deferred thinker eval**    |
| 06 mimi decode_step          |   0.5  |                                             |
| TOTAL                        |  61.5  | (sum of section avgs)                       |

**Important caveat:** `mx.synchronize()` does **not** trigger evaluation of
lazy MLX graphs — it only waits for already-queued GPU work. The per-section
mx.synchronize bracketing in `profile_duplex.py` therefore mis-attributes the
thinker forward to whichever section first calls `.item()` (in our flow,
`predicted_token[0,0].item()` inside section 05).

When we force evaluation at end of section 03 (an explicit `mx.eval` on
`text_logits` and `thinker_pre_norm`), the breakdown shifts to its real shape:

| Section                              | avg ms |
|--------------------------------------|--------|
| 03 thinker forward + lm_head (real)  |  ~27   |
| 04 talker forward                    |   0.5  |
| 05 sample + (cond) audio codes       |  ~5    |
| 06 mimi decode_step                  |   0.5  |
| 01-02 encode + build embeds          |   1.2  |
| Other (Python overhead, state ops)   |  ~30   |

The 36-layer thinker forward + lm_head is the real per-frame bottleneck, not
the code predictor (which is only ~3.6 ms when called).

### Optimizations applied this pass

| Lever | File | Effect |
|---|---|---|
| `mx.fast.rope` for sequential offsets | `models/qwen3.py` | Fused Metal kernel; precomputed `inv_freq` for the `position_ids` fallback |
| Drop `mx.repeat` for GQA — `sdpa` natively supports it | `models/qwen3.py` | Removes 2 ops × 5 layers × 14 steps in code predictor |
| Fused `_swiglu` via `mx.compile(shapeless=True)` | `models/qwen3.py` | One op instead of two per MLP per layer |
| **`lm_head` over only the prediction row `[-2:-1]`** | `models/duplex_generate.py` | **3× cut on the 4096×153723 matmul (~9 ms saved)** |
| Reuse `state.machine_state.last_frame_tokens` (Python list) | `models/duplex_generate.py` | Eliminates a `.tolist()` host sync per frame |
| `mx.async_eval(text_logits, talker_out)` after talker | `models/duplex_generate.py` | Overlaps GPU dispatch with sample/state-machine Python path |
| Logit-mask cache (4 entries cover all states) | `utils/state_machine.py` | Skips the 600 KB numpy→mlx copy after first frame |
| Code-predictor KV cache reuse across frames | `models/generate.py` (prior pass) | ~13% on 28 s offline run |
| `KVCache` accepts `list[int]` for `cache_position` | `modules/kv_cache.py`, `models/qwen3.py`, `models/raon.py` | Plumbing only — passive option after benchmarking didn't show net win |
| **Streaming Voxtral encoder defaults to bfloat16** (was fp32) | `utils/streaming_encoder.py` | **~7 ms/frame, 10% RTF reduction.** Encoder runs every frame and was the largest unquantized op left. |
| **Streaming Voxtral encoder Linear layers 8-bit quant** | `utils/streaming_encoder.py` | **~3 ms/frame, 4-5% additional RTF reduction.** `nn.quantize` skips Conv1d so the conv stem stays bf16; only attention QKV/O + feed-forward Linears are quantized. |

### `mx.compile` exploration (no win at the layer-or-larger scale)

| Test | Raw | Compiled | Win |
|---|---|---|---|
| Single quantized linear | 0.31 ms | 0.30 ms | within noise |
| Full MLP (3 quant matmuls) | 0.58 ms | 0.62 ms | slight regression |
| 5-layer × 14-step autoregressive | 5.31 ms | 4.95 ms | **7%** |
| 36-layer single forward (thinker scale) | 20.64 ms | 19.99 ms | **3%** |

Conclusion: `mx.compile` only helps when there's a long lazy chain to fuse,
and the marginal gain shrinks as compute share grows. For the thinker, the
expected ~3% (~0.6 ms / frame) does not justify refactoring the cache to be
functional (concat-only) and threading 36 (K, V) tuples through every
callsite. We've left the existing path in place.

### Further optimization candidates (not yet tried)

- Reduce the per-frame thinker sequence from 3 → 1 token. Today we re-process
  `[AIP, X, AOP]` each frame so the cache is rewritten with real audio
  embeddings in AIP/AOP. With careful state separation we could update only
  the audio positions, cutting attention work meaningfully. Larger refactor
  of the duplex frame contract.
- 4-bit thinker: known to produce gibberish in this duplex setup (errors
  compound across frames); see Phase 7 notes. Would need quant-aware
  retraining or careful per-layer mixed precision to be viable.
- Drop the `audio_lm_head` EOS column to shrink `[2049, 2048]` → `[2048, 2048]`.
  Trivial in fp16; needs a quant-aware row slice for 8-bit weights.

### Tried and reverted

- KV cache contiguous-write fast-path: replacing the per-position Python loop
  with a single slice assignment was *slower*, presumably because MLX's lazy
  graph prefers smaller writes that fuse with later ops.
- Skipping the init second-pass forward (matching PT's flow exactly): clean
  win for code clarity but no measurable perf change.
- Aligning with `mlx_lm`'s simpler cache pattern (drop `cache_position`,
  drop the init second-pass forward): per-frame time regressed by ~14 ms
  on the offline test (and first-frame jumped to ~3 s). The explicit-position
  path appears to keep MLX's lazy graph from forcing a small sync that the
  inferred-offset path triggers. Worth revisiting if a future MLX version
  improves graph compilation.
- Async-eval at end of `duplex_step` (decoded_audio + prev_audio_feedback) to
  overlap with caller-side work: regressed in the bench harness because the
  next frame's encoder runs immediately on the same GPU and contends with
  the still-in-flight mimi decode. Kept the async-eval after the talker.
- Passing `cache_position` as a Python list (instead of `mx.array`) through
  the thinker stack: gave ~3% regression in the bench, presumably because
  the list path materializes a tuple per layer where the array path is a
  cheap view. Plumbing kept (the cache accepts both forms) but the duplex
  caller continues to pass an `mx.array`.

## Known Issues / Soft Edges

- **External speakers + duplex**: the duplex model is full-duplex by design
  and listens during its own speech. Browser-level AEC (we request
  `echoCancellation: true`) handles built-in mic + built-in speakers
  reasonably on macOS, but external speakers can produce a feedback loop.
  Use headphones, or add server-side mic gating during the assistant
  SPEECH phase if needed (loses barge-in).

## Recently Resolved

- ~~Metal command-buffer race during rapid session restart~~ — fixed
  2026-04-28. Symptom was `failed assertion 'A command encoder is already
  encoding to this command buffer'` + `SIGSEGV` (exit 139) when finishing
  a session and starting a new one in quick succession. Root cause: the
  next session's `init_duplex_state` was racing the previous session's
  last `mimi.decode_step` on shared `model.mimi` streaming state.
  Fix: a per-session step lock guards `handle_audio_frame`, and a new
  `MLXRealtimeDuplexSession.drain()` (called from
  `RealtimeRuntimeManager.finish_session` while still holding the manager
  lock) does `mx.synchronize()` + `mimi.reset_all()` + `mx.synchronize()`
  before the active-session slot is released — so the next session's
  init can never observe in-flight commands from the previous one.

## Open Questions

1. **Mimi num_quantizers**: Raon uses 32 quantizers (1 semantic + 31 acoustic) but Moshi typically uses 8 or 16. PersonaPlex's VQ is parameterized, so this should work, but needs verification.
2. **Code predictor architecture**: The `qwen3_omni_moe_talker_code_predictor` model type may have MoE (mixture of experts) components. Need to inspect the actual code predictor implementation to confirm.
3. **Audio encoder necessity**: For TTS-only use, the audio encoder can be skipped entirely. Confirm this is acceptable for initial port.
4. **Quantization granularity**: 4-bit for the backbone, but what about the talker (4 layers), code predictor (5 layers), and Mimi transformer (8 layers)? These are small enough that quantization may hurt quality more than it helps speed.
