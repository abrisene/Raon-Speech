# Raon-Speech MLX Duplex — Design Spec

**Date:** 2026-04-05
**Author:** Rhona Okafor
**Branch:** feat/mlx-port
**Goal:** Full-parity duplex (realtime bidirectional conversation) on MLX/Apple Silicon

---

## Overview

Port the PyTorch full-duplex inference engine to MLX, achieving feature parity with the
original `gradio_duplex_demo.py` + `demo/realtime/` infrastructure. The duplex engine
processes audio frame-by-frame (80ms at 24kHz/12.5fps), simultaneously encoding user
speech and generating assistant speech, controlled by a Mealy state machine that manages
listening/speaking phase transitions.

After validation, this engine becomes the Raon backend for the PersonaPlex app (to be renamed).

---

## Architecture

Three layers, matching the PyTorch structure:

```
┌──────────────────────────────────────────────────────┐
│ Layer 3: Transport                                   │
│ WebSocket binary protocol + FastAPI + Gradio shell   │
│ Adapted from PyTorch demo with MLX backend swap      │
├──────────────────────────────────────────────────────┤
│ Layer 2: Session Orchestration                       │
│ MLXRealtimeDuplexSession                             │
│ feed_audio() → step() → Frame events                │
│ Backlog management, artifacts, metrics               │
├──────────────────────────────────────────────────────┤
│ Layer 1: Duplex Generation Engine                    │
│ init_duplex_state() → duplex_step() per frame        │
│ Mimi encode → thinker → state machine →              │
│ talker → code predictor → Mimi decode                │
└──────────────────────────────────────────────────────┘
```

---

## Layer 1: Duplex Generation Engine

### Per-frame step function

Each call to `duplex_step()` processes one 80ms audio frame:

```
Input: 1920 PCM samples + DuplexDecodingState
                    │
    ┌───────────────▼──────────────────────┐
    │ 1. Mimi encode_step(pcm)             │
    │    encoder → transformer → quantize  │
    │    → audio codes [1, codebooks, 1]   │
    │                                      │
    │ 2. VQ decode → latent [1, 1, 512]    │
    │    input projection → [1, 1, 4096]   │
    │    = audio_input_embeds              │
    ├──────────────────────────────────────┤
    │ 3. Thinker forward (cached):         │
    │    input_ids = last frame tokens     │
    │    + audio_input_embeds injection    │
    │    + audio_output feedback from prev │
    │    → text_logits, talker_hidden      │
    ├──────────────────────────────────────┤
    │ 4. Apply penalties:                  │
    │    eos_penalty on PAD token          │
    │    sil_penalty on SIL token          │
    │    bc_penalty on BC token (SIL only) │
    │                                      │
    │ 5. State machine logit masking       │
    │    Enforces valid phase transitions  │
    │                                      │
    │ 6. Sample text prediction            │
    ├──────────────────────────────────────┤
    │ 7. If SPEECH phase (pre-transition): │
    │    talker → code predictor → codes   │
    │                                      │
    │ 8. State machine transition:         │
    │    → new phase + frame tokens        │
    │                                      │
    │ 9. If onset (SIL→SPEECH) + no codes: │
    │    Generate codes now                │
    │                                      │
    │ 10. If emitted_audio:                │
    │     Handle acoustic delay buffer     │
    │     Mimi decode_step → output PCM    │
    │     Build feedback embedding         │
    │                                      │
    │ 11. If SIL (no audio emitted):       │
    │     Push silence codes to Mimi       │
    │     Clear semantic buffer            │
    │     Decode silence PCM               │
    └───────────────┬──────────────────────┘
                    ▼
Output: 1920 PCM samples + text delta + updated DuplexDecodingState
```

### DuplexDecodingState

```python
@dataclass
class DuplexDecodingState:
    # Token sequence
    sequences: mx.array              # [1, seq_len] — full token history
    
    # KV caches
    thinker_cache: list[KVCache]     # 36 layers
    talker_cache: list[KVCache]      # 4 layers
    
    # Audio encoder streaming state
    mimi_encoder_cache: MimiEncoderState  # Conv + transformer caches
    
    # Audio output history
    audio_codes: mx.array            # [1, num_frames, 16] — generated codes
    audio_codes_mask: mx.array       # [1, num_frames] — which frames have codes
    
    # State machine
    machine_state: DuplexMachineState
    
    # Acoustic delay
    semantic_buffer: mx.array | None  # Buffered semantic code for delay alignment
    
    # Sampling config
    temperature: float
    top_k: int
    top_p: float
    eos_penalty: float
    sil_penalty: float
    bc_penalty: float
    
    # Speaker conditioning
    speaker_embeds: mx.array | None
    
    # Forced SIL warmup counter
    forced_sil_remaining: int
    
    # Sequence tracking
    last_sequence_len: int           # For text delta extraction
```

### State Machine — Full Parity

Ported from `src/raon/utils/state_machine.py`. Pure logic, torch tensor ops replaced
with mx equivalents.

**Config for Raon-Speech-9B (from RaonDuplexModel defaults):**
- `sequence_mode = "uta"` — frame tokens: [User_input, Text_prediction, Audio_output]
- `use_duplex_end_pad = True` — EPAD token for speech onset
- `use_sil_token = True` — dedicated SIL token for silence
- `no_audio_in_sil = False` — SIL frames still emit [U, A] (audio placeholder present)
- `use_backchannel_token = False` — not active for this checkpoint, implemented for parity

**Phases:** SIL (listening) and SPEECH (speaking)

**Transitions (UTA mode):**

| Current Phase | Prediction | New Phase | Frame Tokens | Audio? |
|--------------|------------|-----------|--------------|--------|
| SIL | SIL | SIL | [U, A] | yes (silence) |
| SIL | EPAD | SPEECH | [U, EPAD, A] | yes (onset) |
| SIL | BC | SPEECH | [U, BC, A] | yes (onset) |
| SIL | text | SPEECH | [U, text, A] | yes (onset) |
| SPEECH | SIL | SIL | [U, A] | yes (silence) |
| SPEECH | PAD | SPEECH | [U, A] | yes |
| SPEECH | EPAD | SPEECH | [U, EPAD, A] | yes |
| SPEECH | text | SPEECH | [U, text, A] | yes |

**Logit masking (enforces valid transitions):**
- SIL phase: only SIL, EPAD, BC allowed
- SPEECH after EPAD/BC onset: only text tokens (forces content after onset marker)
- SPEECH after text: text + PAD + EPAD + SIL
- SPEECH after PAD: PAD + EPAD + SIL
- Structural tokens always blocked: IM_START, IM_END, AUDIO_START, AUDIO_END,
  AUDIO_INPUT_PLACEHOLDER, AUDIO_OUTPUT_PLACEHOLDER, SPEAKER_EMBEDDING_PLACEHOLDER

**Forced SIL warmup:** When `speak_first=False` and `use_sil_token=True`, first step
forces SIL prediction via logit override (`forced_sil_remaining=1`).

### Acoustic Delay Buffer

When `max_delay > 0` (delays=[0, 1, 1, ..., 1]):
- Semantic code (CB0) has delay=0, acoustic codes (CB1-15) have delay=1
- `semantic_buffer` holds CB0 from current frame
- Output frame: `[prev_semantic, current_acoustic]`
- First speech frame: only semantic code, acoustics zeroed
- SIL transition: clear buffer

### Audio Input Encoding — Mimi First, AuT Fallback

**Primary approach:** Use Mimi encoder in streaming mode (from PersonaPlex port).
- `encode_step(pcm)` → encoder → transformer → downsample → quantize → codes
- VQ decode codes → latent [1, 1, 512]
- Project to thinker space: output_adaptor (512 → 4096) or dedicated input projection

**Risk:** Model was trained with AuT (24-layer transformer) input. Mimi encoder produces
different embeddings. If output quality degrades significantly:

**Fallback:** Port causal AuT to MLX (~500-600 lines):
- 24 transformer layers, 1024 hidden, 16 heads
- 3 Conv2d downsampling layers with causal left-padding and border caches
- Causal attention with KV cache (lower-triangular mask within chunk)
- Streaming state: STFT cache, running-max norm, 3 conv caches, 24 KV pairs
- Input adaptor: 2-layer MLP (2048 → 4096) + RMSNorm

### Silence Codes

When model outputs SIL (no speech), push pre-computed silence codes through Mimi decoder
to keep its convolutional state warm. Without this, the first speech frame after silence
has artifacts from stale conv state.

```python
def get_silence_codes(self) -> mx.array:
    """Return silence codebook values for Mimi decoder warmup."""
    # Encode a zero frame through Mimi to get valid silence codes
    ...
```

---

## Layer 2: Session Orchestration

### MLXRealtimeDuplexSession

Adapted from PyTorch `LocalRealtimeSession` (session.py lines 740-958).

**Lifecycle:**
1. `__init__` — load MLX model, init config, compute speaker embeddings
2. `start()` → `[Frame.ready()]`
3. `handle_audio_frame(pcm)` — feed + step loop:
   - `feed_audio(pcm_bytes)` — buffer with backlog management
   - While buffer has full frames: `step()` → emit Frame events
4. `finish(reason)` — close session, flush artifacts, return metadata
5. `close()` — cleanup

**Backlog management (from PyTorch, ported exactly):**
- `soft_backlog_seconds` — log warning when audio buffer exceeds threshold
- `hard_backlog_seconds` — take action (degrade or close)
- `degrade` action: drop oldest frames to target, continue
- `close` action: terminate session
- Track: dropped frames/bytes, max time behind, backlog events

**Artifacts:**
- User audio WAV (accumulated from feed_audio)
- Assistant audio WAV (accumulated from step outputs)
- Transcript text (accumulated from text deltas)
- Metadata JSON (session params, runtime stats)
- Session bundle ZIP

**Metrics per session:**
- frames_in, frames_out, bytes_in, bytes_out
- dropped_input_frames, dropped_input_bytes
- decode_step_total_seconds, decode_step_avg_ms, decode_step_max_ms
- user_audio_seconds, decode_rtf (real-time factor)
- decode_errors, consecutive_decode_errors

### MLX Model Loading

Replace PyTorch `get_runtime()` (SGLang backend) with MLX model loading:

```python
def get_mlx_runtime(model_path: str, quantize_thinker: int = 4) -> tuple[RaonMLX, tokenizer]:
    model = RaonMLX.from_pretrained(model_path)
    # Hybrid quantization: 4-bit thinker, 8-bit talker + code predictor
    nn.quantize(model.thinker, bits=quantize_thinker)
    nn.quantize(model.talker, bits=8)
    nn.quantize(model.code_predictor.model, bits=8)
    tokenizer = load_tokenizer(model_path)
    return model, tokenizer
```

---

## Layer 3: Transport

### Binary Frame Protocol

Copied from `demo/realtime/protocol/messages.py` — framework-agnostic.

| Kind | Byte | Direction | Payload |
|------|------|-----------|---------|
| READY | 0x00 | server→client | empty |
| AUDIO | 0x01 | bidirectional | float32 PCM samples |
| TEXT | 0x02 | server→client | UTF-8 text delta |
| ERROR | 0x05 | server→client | UTF-8 error message |
| CLOSE | 0x06 | bidirectional | UTF-8 reason |
| PING | 0x07 | client→server | arbitrary |
| PONG | 0x08 | server→client | echo payload |

### FastAPI + WebSocket

Adapted from `demo/realtime/api/app.py`:
- `RealtimeRuntimeManager` — singleton, enforces one active session
- `mount_realtime_websocket(app, manager)` — WebSocket at `/realtime/ws`
- `POST /realtime/session/start` — reserve session with config
- `POST /realtime/session/finish` — close and retrieve artifacts
- `GET /health` — status check

**Changes from PyTorch version:**
- `get_runtime()` → `get_mlx_runtime()` (load RaonMLX instead of SGLang)
- Remove CUDA-specific code (tf32, flash attention, cuda graph)
- Session factory creates `MLXRealtimeDuplexSession`

### Gradio Duplex Shell

Adapted from `demo/gradio_duplex_demo.py`:
- Same UI: mode selector, persona dropdown, sampling controls, start/finish buttons
- Same JS client (`gradio_stream.js`, `gradio_stop.js`) — no changes needed
- Mount Gradio on FastAPI app at `/`, WebSocket at `/realtime/ws`

---

## Special Tokens

Ported from `src/raon/utils/special_tokens.py`:

| Token | ID | Text | Purpose |
|-------|-----|------|---------|
| IM_START | 151644 | `<\|im_start\|>` | Message boundary |
| IM_END | 151645 | `<\|im_end\|>` | Message boundary |
| AUDIO_START | 151669 | `<\|audio_start\|>` | Audio generation begin |
| AUDIO_END | 151670 | `<\|audio_end\|>` | Audio generation end |
| SPEAKER_EMBEDDING | 151671 | `<\|speaker_embedding_placeholder\|>` | Speaker conditioning |
| DUPLEX_SIL | 151672 | `<\|audio_output_sil\|>` | Silence (listening) |
| AUDIO_OUTPUT_BC | 151673 | `<\|audio_output_backchannel\|>` | Backchannel onset |
| AUDIO_OUTPUT_PLACEHOLDER | 151675 | `<\|audio_output_placeholder\|>` | Audio frame marker |
| AUDIO_INPUT_PLACEHOLDER | 151676 | `<\|audio_input_placeholder\|>` | Audio input marker |
| AUDIO_OUTPUT_PAD | 151677 | `<\|audio_output_pad\|>` | Speech padding |
| AUDIO_OUTPUT_END_PAD | 151678 | `<\|audio_output_end_pad\|>` | Speech onset (EPAD) |

---

## File Plan

| File | Est. Lines | Purpose |
|------|-----------|---------|
| `src/raon_mlx/utils/special_tokens.py` | ~80 | Token constants |
| `src/raon_mlx/utils/state_machine.py` | ~270 | DuplexStateManager (full parity) |
| `src/raon_mlx/models/duplex_generate.py` | ~500 | DuplexDecodingState, init, step |
| `src/raon_mlx/realtime/__init__.py` | ~5 | Package |
| `src/raon_mlx/realtime/session.py` | ~300 | MLXRealtimeDuplexSession |
| `src/raon_mlx/realtime/protocol/__init__.py` | ~5 | Package |
| `src/raon_mlx/realtime/protocol/messages.py` | ~85 | Binary frame protocol |
| `src/raon_mlx/realtime/api/__init__.py` | ~5 | Package |
| `src/raon_mlx/realtime/api/app.py` | ~250 | FastAPI + WebSocket |
| `demo/gradio_mlx_duplex_demo.py` | ~200 | Gradio duplex shell |
| `demo/realtime/web/gradio_stream.js` | — | Reuse existing (no changes) |
| `demo/realtime/web/gradio_stop.js` | — | Reuse existing (no changes) |

**Total new code:** ~1,700 lines

---

## Build Order

1. `special_tokens.py` — token constants, no dependencies
2. `state_machine.py` — Mealy machine, depends on special_tokens
3. `duplex_generate.py` — core engine, depends on state_machine + RaonMLX + Mimi
4. **Offline test** — run duplex over pre-recorded WAV, verify output quality
5. `protocol/messages.py` — binary frame codec (copy)
6. `session.py` — session orchestration with backlog/artifacts
7. `api/app.py` — FastAPI + WebSocket
8. `gradio_mlx_duplex_demo.py` — Gradio UI
9. **Realtime test** — live mic through Gradio

---

## Risk: Mimi Input Quality

The model was trained with AuT encoder input (24-layer transformer, 2048-dim output
projected to 4096 via input_adaptor). We substitute Mimi encoder (convolutional +
8-layer transformer, 512-dim VQ latent projected to 4096 via output_adaptor).

**Detection:** If the thinker receives Mimi embeddings but doesn't understand them,
the output will be garbled or the model will stay in permanent SIL. We'll know in
the first offline test.

**Mitigation:** Port the causal AuT encoder to MLX. Estimated 500-600 additional lines.
Architecture: 3 causal Conv2d downsampling layers + 24 causal transformer layers +
input adaptor MLP. Streaming state: STFT cache, running-max norm, conv border caches,
24 KV cache pairs, frame counter.

---

## After Validation

Once duplex works on MLX:
1. Add `RaonMLXPipeline.duplex()` method for programmatic access
2. Port duplex engine into PersonaPlex as a Raon backend
3. PersonaPlex provides the production UI (WebSocket, voice UI, session management)
4. Gradio demo becomes a development/testing tool only
