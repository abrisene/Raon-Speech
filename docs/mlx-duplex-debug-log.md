# MLX Duplex Debug Log

_Date: 2026-04-06_

This note records the debugging work done on the MLX realtime duplex path in `feat/mlx-port` so the next pass does not have to rediscover the same ground.

## Problem statement

The MLX realtime duplex demo boots and streams, but assistant speech is still stuttery / gibberish. Assistant-side generated text sometimes becomes more coherent, but speech quality remains bad.

## High-level conclusions so far

1. The initial WebSocket problem was real, but it was only the first blocker.
2. The current `transcript.txt` artifact in the MLX duplex path is **assistant-generated text**, not STT of the user.
3. The local SpeechChat MLX artifact is **not** obviously stale or corrupted; a fresh conversion produced a bit-for-bit identical file.
4. The remaining problem appears to be in the **runtime semantics of the MLX duplex path**, not in transport and not in the checkpoint conversion artifact itself.
5. The base MLX path is healthier than duplex: MLX TTS works, but MLX STT currently fails with an audio-input embedding shape mismatch, which suggests the audio-input side is unstable more broadly than just realtime duplex.

## What we verified / tried

### 1. Transport and session bootstrapping

- Confirmed the repo’s realtime path is FastAPI + WebSocket + Gradio.
- Started the MLX duplex server locally and verified:
  - `/health` works
  - `POST /realtime/session/start` returns a `session_id`
  - `ws://127.0.0.1:7862/realtime/ws?...` returns READY (`0x00`)
- Conclusion: transport is not the core problem anymore.

### 2. Prompt / session / tokenizer plumbing

Applied and tested fixes to make the MLX realtime session behave more like the reference path:

- load tokenizer/config via `RaonProcessor` rather than a bare tokenizer
- resolve duplex prompt keys like `eng:full_duplex:listen-first`
- pass `persona` / `persona_context`
- derive duplex token/config flags from the HF checkpoint rather than hardcoding them
- fix the text-vocab boundary used during duplex text-side sampling

These changes improved text behavior somewhat, but did **not** fix speech quality.

### 3. Browser capture and UI issues

Observed two separate frontend behaviors:

- some runs produced immediate close with `capture-start-failed`
- some runs captured audio successfully but still produced gibberish assistant output

Useful findings:

- recent sessions at one point had `user.wav` files that were all zeros or near-zero
- the MLX demo default browser noise gate was too aggressive (`0.03`), so this was lowered to `0.0`
- later sessions confirmed that capture can succeed and write nonzero `user.wav`

Conclusion: capture problems exist, but they do not fully explain the remaining gibberish when capture succeeds.

### 4. Transcript semantics

Confirmed from `src/raon_mlx/realtime/session.py` that the MLX duplex `transcript.txt` is built from `text_ids` emitted by the assistant generation path. It is **not** user STT.

This means nonsense transcript output is evidence that the assistant-side generation path is wrong, not that user transcription is merely inaccurate.

### 5. Speech-side sampling and code generation experiments

Tried several runtime-level fixes on the MLX duplex path, including:

- use top-k / top-p for text-side sampling (instead of ignoring them)
- make downstream audio codebooks greedy instead of sampled
- force audio-code temperature toward deterministic behavior
- avoid injecting zero vectors when the streaming encoder has not yet emitted a new audio frame

These changes sometimes improved apparent text coherence, but speech was still bad.

### 6. Code predictor parity check

Built an isolated comparison between the PyTorch reference code predictor and the MLX code predictor using the same talker hidden state.

Result:

- earlier MLX cached code-predictor behavior diverged from PyTorch
- after changing the MLX path to recompute from accumulated embeddings instead of using the buggy cache path, the generated discrete audio codes matched the PyTorch reference exactly for the same hidden state

Conclusion: the MLX code predictor itself can match the reference. The remaining issue is elsewhere.

### 7. Audio-output feedback embedding parity check

Compared `_get_audio_output_embed(...)` in MLX with the equivalent PyTorch path.

Result:

- they matched essentially exactly

Conclusion: the output-adaptor / latent-to-thinker feedback path is probably not the culprit.

### 8. Mimi decoder checks

Tested whether Mimi itself is the wrong codec choice.

Findings:

- the model config explicitly uses Mimi (`audio_tokenizer_config._name_or_path = "kyutai/mimi"`)
- for output-side audio, Mimi is the intended codec path
- MLX Mimi **batch decode** is reasonably close to the PyTorch Mimi decoder on the same discrete codes
- MLX Mimi **streaming decode path** diverged much more strongly from the batch decode / reference behavior

This led to an experiment where live duplex output used batch-decode-the-accumulated-codes and emitted the newest 80ms tail instead of relying on MLX `decode_step()`. That did not solve the overall speech problem.

Conclusion: Mimi is probably the right codec family, but the remaining issue is not solved by simply changing how we call the decoder.

### 9. Rebaseline to branch HEAD

Because local experimental edits had accumulated, the core MLX duplex engine files were restored to the branch HEAD versions, while keeping useful session/UI fixes around them.

Useful history finding:

- local commit `aa06b69` explicitly claims clean English duplex output after fixing encoder weights + SpeechChat conversion
- upstream `origin/main` has no relevant newer MLX duplex work
- the interesting MLX duplex work exists on this branch, not on upstream main

Conclusion: there is no missing upstream fix on the original remote that obviously explains the problem.

### 10. Fresh SpeechChat MLX conversion

Created a fresh conversion:

- `models/Raon-SpeechChat-9B-mlx-hybrid-fresh`

Compared it against the existing local artifact.

Result:

- same SHA-256 hash
- same file size
- same minimal config

Conclusion: the local SpeechChat MLX artifact is not stale or uniquely corrupted. A fresh conversion produces the same artifact.

### 11. Baseline MLX sanity checks outside duplex

Ran quick checks on the base MLX paths:

- MLX TTS works (`raon_mlx.tts` completed and wrote audio)
- MLX STT currently fails with a shape mismatch during audio-input embedding insertion in `stt_generate`

Conclusion: the MLX audio-input / speech-understanding side appears to be unstable more generally, not just in realtime duplex.

## Current most likely area of failure

The strongest remaining hypothesis is that the **MLX runtime semantics for live duplex state updates are still wrong** even though several lower-level pieces now look sane in isolation.

The likely regions are:

- audio-input embedding timing / insertion semantics
- thinker/talker cache overwrite behavior in the live duplex loop
- frame-to-frame runtime assumptions that differ from the PyTorch reference
- broader MLX audio-input path issues hinted at by the current STT shape-mismatch failure

## Files touched during debugging

Core files repeatedly investigated or edited:

- `src/raon_mlx/models/duplex_generate.py`
- `src/raon_mlx/models/generate.py`
- `src/raon_mlx/realtime/session.py`
- `src/raon_mlx/realtime/api/app.py`
- `src/raon_mlx/utils/streaming_encoder.py`
- `demo/gradio_mlx_duplex_demo.py`
- `demo/realtime/web/gradio_stream.js`

Reference files used for parity checks:

- `src/raon/models/wrapper.py`
- `src/raon/models/raon.py`
- `src/raon/modules/audio_tokenizer.py`
- `src/raon/modules/code_predictor.py`

## Artifacts / notes worth keeping

- experimental core-engine patch backup:
  - `.omx/context/duplex-engine-experimental-backup.patch`
- PRD / test spec created during this debugging loop:
  - `.omx/plans/prd-mlx-duplex-realtime.md`
  - `.omx/plans/test-spec-mlx-duplex-realtime.md`

## Suggested next steps

1. Fix / re-verify the MLX STT audio-input embedding path, since that is a cleaner entry point into the same class of audio-input bugs.
2. Instrument the duplex loop frame-by-frame against the PyTorch reference, especially around audio-input embedding availability and cache overwrite semantics.
3. Compare the first N live duplex steps between MLX and PyTorch on the same saved `user.wav` input rather than relying on ear tests alone.
4. Keep transport/UI fixes, but treat the remaining blocker as a runtime-semantic bug, not a websocket/checkpoint-conversion problem.
