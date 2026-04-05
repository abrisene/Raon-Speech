# Raon MLX Duplex Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Full-parity duplex (realtime bidirectional conversation) on MLX/Apple Silicon

**Architecture:** Three-layer system — duplex engine (state machine + per-frame generation), session orchestration (backlog + artifacts), transport (WebSocket + Gradio). Mimi encoder for audio input, with causal AuT as fallback.

**Tech Stack:** MLX, Python, FastAPI, WebSocket, Gradio, NumPy

**Spec:** `docs/duplex-mlx-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/raon_mlx/utils/special_tokens.py` | Token ID constants for duplex control tokens |
| `src/raon_mlx/utils/state_machine.py` | Mealy state machine: phase transitions + logit masking |
| `src/raon_mlx/models/duplex_generate.py` | DuplexDecodingState, init, step, sequence update |
| `src/raon_mlx/realtime/__init__.py` | Package init |
| `src/raon_mlx/realtime/protocol/__init__.py` | Package init |
| `src/raon_mlx/realtime/protocol/messages.py` | Binary frame protocol (AUDIO/TEXT/CLOSE) |
| `src/raon_mlx/realtime/session.py` | Session orchestration: feed/step/backlog/artifacts |
| `src/raon_mlx/realtime/api/__init__.py` | Package init |
| `src/raon_mlx/realtime/api/app.py` | FastAPI + WebSocket + runtime manager |
| `demo/gradio_mlx_duplex_demo.py` | Gradio UI shell for duplex |

---

### Task 1: Special Tokens

**Files:**
- Create: `src/raon_mlx/utils/special_tokens.py`

- [ ] **Step 1: Create special_tokens.py**

Port token constants from `src/raon/utils/special_tokens.py`. Pure dataclass + constants, no framework dependency.

```python
# All 12 special tokens with IDs matching the PyTorch version
# IM_START(151644), IM_END(151645), AUDIO_START(151669), AUDIO_END(151670),
# SPEAKER_EMBEDDING_PLACEHOLDER(151671), DUPLEX_SIL(151672), AUDIO_OUTPUT_BC(151673),
# AUDIO_OUTPUT_PLACEHOLDER(151675), AUDIO_INPUT_PLACEHOLDER(151676),
# AUDIO_OUTPUT_PAD(151677), AUDIO_OUTPUT_END_PAD(151678), PAD(151679)
```

- [ ] **Step 2: Verify token IDs match**

Run: `python -c "from raon_mlx.utils.special_tokens import *; print(DUPLEX_SIL.id, AUDIO_OUTPUT_END_PAD.id)"`
Expected: `151672 151678`

---

### Task 2: State Machine

**Files:**
- Create: `src/raon_mlx/utils/state_machine.py`

- [ ] **Step 1: Port DuplexStateManager**

Port the full state machine from `src/raon/utils/state_machine.py` (265 lines). Replace torch tensor ops with mx equivalents in `apply_logit_mask()`. The rest is pure Python logic.

Key classes:
- `DuplexPhase` enum (SIL, SPEECH)
- `DuplexMachineState` dataclass (phase, last_frame_tokens, num_input_tokens, emitted_audio)
- `DuplexStateConfig` frozen dataclass (all config flags)
- `DuplexStateManager` (initial_state, initial_forced_prediction_id, transition, apply_logit_mask)

The logit mask method uses `mx.full`, `mx.zeros_like` instead of torch equivalents.

- [ ] **Step 2: Verify state machine transitions**

Test the core transitions match PyTorch behavior:
```python
from raon_mlx.utils.state_machine import *
mgr = DuplexStateManager(DuplexStateConfig(
    use_duplex_end_pad=True, use_sil_token=True, sequence_mode="uta"))
state = mgr.initial_state()
# SIL + EPAD -> SPEECH
new_state, tokens, audio = mgr.transition(state, AUDIO_OUTPUT_END_PAD.id)
assert new_state.phase == DuplexPhase.SPEECH
assert tokens == [AUDIO_INPUT_PLACEHOLDER.id, AUDIO_OUTPUT_END_PAD.id, AUDIO_OUTPUT_PLACEHOLDER.id]
```

- [ ] **Step 3: Commit**

```bash
git add src/raon_mlx/utils/special_tokens.py src/raon_mlx/utils/state_machine.py
git commit -m "feat(duplex): add special tokens and state machine for duplex inference"
```

---

### Task 3: Duplex Generation Engine

**Files:**
- Create: `src/raon_mlx/models/duplex_generate.py`
- Modify: `src/raon_mlx/models/generate.py` (extract shared helpers if needed)

This is the core — ~500 lines. Three main functions:

- [ ] **Step 1: Create DuplexDecodingState dataclass**

```python
@dataclass
class DuplexDecodingState:
    sequences: mx.array           # [1, seq_len]
    thinker_cache: list[KVCache]
    talker_cache: list[KVCache]
    audio_codes: mx.array         # [1, num_frames, 16]
    audio_codes_mask: mx.array    # [1, num_frames]
    machine_state: DuplexMachineState
    semantic_buffer: mx.array | None
    temperature: float
    top_k: int
    top_p: float
    eos_penalty: float
    sil_penalty: float
    bc_penalty: float
    speaker_embeds: mx.array | None
    forced_sil_remaining: int
    last_sequence_len: int
    state_config: DuplexStateConfig
```

- [ ] **Step 2: Implement init_duplex_state()**

Initialize duplex decoding:
1. Tokenize system prompt via tokenizer.apply_chat_template
2. Append [IM_START, AUDIO_START] tokens
3. Optionally insert speaker token before IM_START
4. Run thinker prefill to build KV cache
5. Create DuplexStateManager with model config
6. Force initial prediction (SIL for listen-first, EPAD for speak-first)
7. Run _update_duplex_sequences (first frame)
8. Handle initial audio: if SIL, push silence codes; if speech, handle acoustic delay
9. Reset Mimi encoder/decoder streaming state
10. Return DuplexDecodingState

Key: Read model config flags from the checkpoint (use_duplex_end_pad=True, use_sil_token=True, sequence_mode="uta").

- [ ] **Step 3: Implement duplex_step()**

Per-frame step function:
1. Encode user audio: `mimi.encode_step(pcm)` → codes → `quantizer.decode(codes)` → latent → `output_adaptor(latent)` → audio_input_embeds [1, 1, 4096]
2. Build thinker input: last N frame tokens as input_ids, inject audio_input_embeds at AUDIO_INPUT_PLACEHOLDER position, inject audio feedback at AUDIO_OUTPUT_PLACEHOLDER position
3. Thinker forward (cached): get text_logits and pre-norm hidden state
4. If forced_sil_remaining > 0: force SIL logits
5. Call _update_duplex_sequences_and_generate_audio_codes (next step)
6. If emitted_audio and speech: handle acoustic delay, decode audio via mimi.decode_step
7. If SIL: push silence codes, decode silence, clear semantic buffer
8. Build feedback embedding from generated codes (_get_audio_output_embed)
9. Return updated state + output PCM + text delta

- [ ] **Step 4: Implement _update_duplex_sequences_and_generate_audio_codes()**

The orchestrator (port from wrapper.py lines 902-1021):
1. Extract text logits from position -2 (before [A])
2. Apply penalties: eos_penalty on PAD, sil_penalty on SIL, bc_penalty on BC
3. Apply state machine logit mask
4. Sample text prediction
5. If currently in SPEECH: generate audio codes via generate_audio_codes()
6. Run state machine transition
7. If onset frame (SIL→SPEECH) and no codes yet: generate codes now
8. If emitted_audio: append codes, handle audio_end sentinel clamping
9. Append frame tokens to sequences
10. Return updated state components

- [ ] **Step 5: Implement get_silence_codes()**

Encode a zero-PCM frame through Mimi to get valid silence codebook values:
```python
def get_silence_codes(model: RaonMLX) -> mx.array:
    silence_pcm = mx.zeros((1, 1, 1920))
    codes = model.mimi.encode(silence_pcm)  # [1, codebooks, 1]
    return codes[:, :16, 0]  # [1, 16] — only first 16 codebooks
```

- [ ] **Step 6: Implement run_duplex_offline()**

Offline test harness: reads a WAV file, runs duplex frame-by-frame, saves output audio.
```python
def run_duplex_offline(model, tokenizer, audio_path, output_dir, ...):
    # Load and chunk audio into 1920-sample frames
    # init_duplex_state()
    # For each frame: duplex_step()
    # Concatenate output PCM, save WAV
```

- [ ] **Step 7: Test offline duplex**

Run: `python -c "from raon_mlx.models.duplex_generate import run_duplex_offline; ..."`

Use a short test WAV (e.g., `data/duplex/eval/audio/spk_ref.wav` if it exists, or generate silence).
Verify: output WAV is generated, has non-zero samples during speech frames, model transitions between SIL and SPEECH phases.

- [ ] **Step 8: Commit**

```bash
git add src/raon_mlx/models/duplex_generate.py
git commit -m "feat(duplex): add MLX duplex generation engine with state machine"
```

---

### Task 4: Binary Frame Protocol

**Files:**
- Create: `src/raon_mlx/realtime/__init__.py`
- Create: `src/raon_mlx/realtime/protocol/__init__.py`
- Create: `src/raon_mlx/realtime/protocol/messages.py`

- [ ] **Step 1: Copy protocol from PyTorch**

Copy `demo/realtime/protocol/messages.py` verbatim — it's pure Python + numpy, no torch dependency. Create package init files.

- [ ] **Step 2: Commit**

```bash
git add src/raon_mlx/realtime/
git commit -m "feat(duplex): add binary frame protocol for realtime WebSocket"
```

---

### Task 5: Session Orchestration

**Files:**
- Create: `src/raon_mlx/realtime/session.py`

- [ ] **Step 1: Implement MLXRealtimeDuplexSession**

Adapt from PyTorch `LocalRealtimeSession` (session.py lines 740-958). Key changes:
- Replace `get_runtime()` with MLX model loading
- Replace `RealtimeDuplexSession` inner class with our duplex engine calls
- Keep all backlog management, artifact collection, metrics

Core methods:
- `__init__`: load MLX model, init config, speaker embeds
- `start()` → `[Frame.ready()]`
- `handle_audio_frame(pcm)`: feed_audio + step loop → Frame events
- `finish(reason)`: close, flush artifacts, return metadata
- `close()`: cleanup

Backlog management (port from PyTorch session.py lines 504-551):
- Audio buffer with frame alignment
- Soft/hard backlog thresholds
- Degrade action: drop oldest frames
- Close action: terminate session

Artifacts (simplified — save WAVs, transcript, metadata JSON):
- Accumulate user/assistant PCM in numpy arrays
- Write WAV files on finish
- Write transcript text file
- Write metadata JSON with runtime stats

- [ ] **Step 2: Implement get_mlx_runtime()**

Singleton model loader:
```python
def get_mlx_runtime(model_path, quantize="hybrid"):
    # Load RaonMLX model
    # Apply quantization (hybrid: 4-bit thinker, 8-bit talker/cp)
    # Load tokenizer from HF checkpoint
    # Reset Mimi state
    return model, tokenizer
```

- [ ] **Step 3: Commit**

```bash
git add src/raon_mlx/realtime/session.py
git commit -m "feat(duplex): add MLX realtime session with backlog management"
```

---

### Task 6: FastAPI + WebSocket

**Files:**
- Create: `src/raon_mlx/realtime/api/__init__.py`
- Create: `src/raon_mlx/realtime/api/app.py`

- [ ] **Step 1: Port FastAPI app**

Adapt from `demo/realtime/api/app.py`. Key changes:
- `RealtimeRuntimeManager`: replace `_resolve_session_factory` to use our `MLXRealtimeDuplexSession`
- `get_runtime_manager`: use MLX model path
- `_prepare_runtime_model`: remove CUDA-specific code (tf32, flash attention, compile)
- Keep: WebSocket handler, session start/finish endpoints, health check

The WebSocket handler (`websocket_duplex`) is framework-agnostic — it calls session methods via the manager. Minimal changes needed.

- [ ] **Step 2: Commit**

```bash
git add src/raon_mlx/realtime/api/
git commit -m "feat(duplex): add FastAPI WebSocket server for MLX duplex"
```

---

### Task 7: Gradio Duplex Demo

**Files:**
- Create: `demo/gradio_mlx_duplex_demo.py`

- [ ] **Step 1: Adapt Gradio shell**

Adapt from `demo/gradio_duplex_demo.py`. Changes:
- Import from `raon_mlx.realtime.api.app` instead of `demo.realtime.api.app`
- Default model path to pre-converted MLX model
- Remove `--compile-audio-modules` and `--compile-max-sequence-length` args (CUDA-specific)
- Keep: all UI controls, JS streaming, start/finish flow, artifact downloads
- Reuse existing `demo/realtime/web/gradio_stream.js` and `gradio_stop.js`

- [ ] **Step 2: End-to-end test**

Run: `python demo/gradio_mlx_duplex_demo.py --model-path <mlx-model-path>`
Verify: Gradio UI loads, WebSocket connects, microphone captures, model responds with audio

- [ ] **Step 3: Commit**

```bash
git add demo/gradio_mlx_duplex_demo.py
git commit -m "feat(duplex): add Gradio MLX duplex demo"
```

---

### Task 8: Pipeline Integration + Final Test

**Files:**
- Modify: `src/raon_mlx/pipeline.py`

- [ ] **Step 1: Add duplex() method to RaonMLXPipeline**

```python
def duplex(self, audio_input, output_dir, *, system_prompt=None, 
           speak_first=False, temperature=0.9, top_k=66, top_p=0.99,
           speaker_audio=None, eos_penalty=0.0, sil_penalty=0.0, 
           bc_penalty=0.0) -> dict:
    from .models.duplex_generate import run_duplex_offline
    return run_duplex_offline(
        model=self.model, tokenizer=self.tokenizer,
        audio_path=audio_input, output_dir=output_dir, ...)
```

- [ ] **Step 2: Full integration test**

Test offline duplex through the pipeline:
```python
pipe = RaonMLXPipeline("models/Raon-Speech-9B")
result = pipe.duplex("test_audio.wav", "output/duplex_test")
# Verify output files exist, audio is non-empty
```

Test realtime duplex through Gradio with live mic.

- [ ] **Step 3: Commit**

```bash
git add src/raon_mlx/pipeline.py
git commit -m "feat(duplex): add duplex() to MLX pipeline API"
```

---

## Execution Notes

- **Build order is strict** — each task depends on the previous
- **Task 3 is the hard one** — ~500 lines, most of the logic. Budget accordingly.
- **Offline test (Task 3, Step 7) is the quality gate** — if Mimi input produces garbage, stop and port causal AuT before continuing to Tasks 4-7
- **Tasks 4-7 are infrastructure** — mostly adapting existing code, lower risk
- **The JS client works unchanged** — don't touch gradio_stream.js or gradio_stop.js
