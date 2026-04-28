# MLX Port Test Checklist

Audio files to evaluate. All generated on M5 Max with hybrid quantization (4-bit thinker, 8-bit talker+CP) unless noted.

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# TTS (from HF checkpoint, quantizes on-the-fly):
python -m raon_mlx.tts "Your text here" --model models/Raon-Speech-9B --quant hybrid -o output/test.wav

# TTS with speaker cloning:
python -m raon_mlx.tts "Your text here" --model models/Raon-Speech-9B --speaker data/duplex/eval/audio/spk_ref.wav -o output/clone.wav

# STT:
python -m raon_mlx.stt audio.wav --hf-model models/Raon-Speech-9B

# Gradio demo (all tasks in browser):
python demo/gradio_mlx_demo.py --model models/Raon-Speech-9B --port 7880

# Optional: pre-convert for faster TTS loading (6.46 GB):
python -m raon_mlx.utils.convert models/Raon-Speech-9B -o models/Raon-Speech-9B-mlx-hybrid
python -m raon_mlx.tts "Your text here" --model models/Raon-Speech-9B-mlx-hybrid -o output/test.wav

# Python API:
python -c "
from raon_mlx.pipeline import RaonMLXPipeline
pipe = RaonMLXPipeline('models/Raon-Speech-9B', quant='hybrid')
audio, sr = pipe.tts('Hello world!')
pipe.save_audio((audio, sr), 'output/test.wav')
text = pipe.stt('output/test.wav')
print(text)
"
```

## Tests to Run

### 1. Basic TTS Quality

Listen and evaluate: is the speech intelligible? Natural-sounding?

```bash
# Short
python -m raon_mlx.tts "Hello." -o output/test_short.wav --model models/Raon-Speech-9B-mlx-hybrid

# Medium
python -m raon_mlx.tts "Hello David. This is Raon Speech running on MLX with Apple Silicon." -o output/test_medium.wav --model models/Raon-Speech-9B-mlx-hybrid

# Long
python -m raon_mlx.tts "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet, which makes it useful for testing text to speech systems." -o output/test_long.wav --model models/Raon-Speech-9B-mlx-hybrid
```

- [ ] Short text produces speech (not silence, not noise)
- [ ] Medium text is intelligible English
- [ ] Long text maintains coherence throughout (doesn't degrade)
- [ ] No obvious artifacts (clicks, pops, metallic sounds)

### 2. Speaker Conditioning

Compare default voice vs speaker-conditioned voice.

```bash
# Default (random) voice
python -m raon_mlx.tts "Testing the default voice without speaker conditioning." -o output/test_default_voice.wav --model models/Raon-Speech-9B-mlx-hybrid

# Cloned voice from reference
python -m raon_mlx.tts "Testing speaker conditioning with a reference voice." --speaker data/duplex/eval/audio/spk_ref.wav -o output/test_cloned_voice.wav --model models/Raon-Speech-9B-mlx-hybrid
```

- [ ] Default voice sounds like speech
- [ ] Cloned voice sounds *different* from default
- [ ] Cloned voice has characteristics of the reference audio
- [ ] Both complete without errors

### 3. PyTorch MPS vs MLX Comparison

Generate the same text with both backends and compare quality.

```bash
# MLX
python -m raon_mlx.tts "Compare this output between MLX and PyTorch." -o output/compare_mlx.wav --model models/Raon-Speech-9B-mlx-hybrid

# PyTorch MPS (run in Python):
python -c "
from raon import RaonPipeline
pipe = RaonPipeline('models/Raon-Speech-9B', device='mps', dtype='float16')
audio, sr = pipe.tts('Compare this output between MLX and PyTorch.')
pipe.save_audio((audio, sr), 'output/compare_pytorch.wav')
"
```

- [ ] Both produce intelligible speech
- [ ] Quality is comparable (MLX may sound slightly different due to quantization)
- [ ] MLX is noticeably faster

### 4. Quantization Modes

Test all quantization options for quality comparison.

```bash
TEXT="Testing quantization quality differences across bit widths."

# No quantization (slowest, best quality baseline)
python -m raon_mlx.tts "$TEXT" --model models/Raon-Speech-9B --quant none -o output/quant_none.wav

# 8-bit all
python -m raon_mlx.tts "$TEXT" --model models/Raon-Speech-9B --quant 8bit -o output/quant_8bit.wav

# Hybrid (recommended)
python -m raon_mlx.tts "$TEXT" --model models/Raon-Speech-9B --quant hybrid -o output/quant_hybrid.wav

# 4-bit all
python -m raon_mlx.tts "$TEXT" --model models/Raon-Speech-9B --quant 4bit -o output/quant_4bit.wav
```

- [ ] No quant: clear speech (reference quality)
- [ ] 8-bit: comparable to no quant
- [ ] Hybrid: comparable to 8-bit
- [ ] 4-bit: acceptable quality (may have minor artifacts)
- [ ] Speed increases with more aggressive quantization

### 5. Edge Cases

```bash
# Very short
python -m raon_mlx.tts "Hi." -o output/edge_very_short.wav --model models/Raon-Speech-9B-mlx-hybrid

# Numbers and punctuation
python -m raon_mlx.tts "I have 3 cats, 2 dogs, and 1,000 reasons to be happy!" -o output/edge_numbers.wav --model models/Raon-Speech-9B-mlx-hybrid

# Question
python -m raon_mlx.tts "What do you think about running speech models on a laptop?" -o output/edge_question.wav --model models/Raon-Speech-9B-mlx-hybrid

# Korean (bilingual model)
python -m raon_mlx.tts "안녕하세요. 이것은 한국어 테스트입니다." -o output/edge_korean.wav --model models/Raon-Speech-9B-mlx-hybrid
```

- [ ] Very short text produces something (not empty/error)
- [ ] Numbers are spoken correctly
- [ ] Question intonation is present
- [ ] Korean produces Korean speech

### 6. Performance

```bash
# Time a longer generation
time python -m raon_mlx.tts "This is a longer sentence designed to test the sustained generation performance of the MLX port. We expect it to maintain faster than real-time generation throughout the entire utterance without degradation." -o output/perf_long.wav --model models/Raon-Speech-9B-mlx-hybrid
```

- [ ] RTF < 1.0 (real-time or faster)
- [ ] Model loads instantly from pre-converted weights
- [ ] No memory errors or crashes

### 7. Gradio Demo

```bash
python demo/gradio_mlx_demo.py --model models/Raon-Speech-9B --port 7880
# Open http://localhost:7880
```

- [ ] Demo launches without errors
- [ ] TTS task: enter text, click Generate, hear audio
- [ ] STT task: upload/record audio, click Generate, see transcription
- [ ] SpeechChat task: upload audio, get text response
- [ ] Speaker reference: upload ref audio for TTS, voice changes

### 8. Pipeline API

```python
from raon_mlx.pipeline import RaonMLXPipeline
pipe = RaonMLXPipeline("models/Raon-Speech-9B", quant="hybrid")

# TTS roundtrip
audio, sr = pipe.tts("Testing the pipeline API.")
pipe.save_audio((audio, sr), "output/pipe_test.wav")
text = pipe.stt("output/pipe_test.wav")
print(text)  # Should be close to "Testing the pipeline API."
```

- [ ] Pipeline loads without errors
- [ ] TTS produces audio
- [ ] STT transcribes correctly
- [ ] TTS → STT roundtrip produces recognizable text

## Known Issues

- HF tokenizer prints `OrderedVocab` warnings to stderr (cosmetic, from Rust tokenizer)
- Audio duration varies — shorter text sometimes produces proportionally shorter audio than expected
- Voice is random each generation without speaker conditioning (expected behavior)
- Korean output may occur occasionally without explicit language conditioning
- **Full-duplex SpeechChat requires `quant="8bit"` (uniform), not hybrid.** 4-bit thinker errors compound across the multi-frame loop and produce gibberish audio even though SIL frames render cleanly. Use the `Raon-SpeechChat-9B-mlx-8bit` pre-converted artifact or pass `--quantize 8bit` to the demo. Single-utterance TTS is unaffected and hybrid remains the recommended quant there.

## Duplex / SpeechChat

```bash
# One-shot conversion to 8-bit pre-quantized artifact (10.32 GB, faster load):
python -m raon_mlx.utils.convert models/Raon-SpeechChat-9B \
    --output models/Raon-SpeechChat-9B-mlx-8bit --quant 8bit

# Offline duplex run on a 24kHz mono wav (writes assistant.wav, conversation.wav,
# transcript.txt, frame_log.txt, summary.json):
python scripts/run_offline_test.py
# overrides via env: RAON_USER_WAV, RAON_MODEL_PATH, RAON_HF_PATH, RAON_QUANT,
# RAON_OUT, RAON_OUT_SUFFIX

# Realtime FastAPI + Gradio demo on http://127.0.0.1:7862 :
python demo/gradio_mlx_duplex_demo.py \
    --host 127.0.0.1 --port 7862 \
    --model-path models/Raon-SpeechChat-9B-mlx-8bit \
    --hf-model-path models/Raon-SpeechChat-9B \
    --quantize 8bit
```

Acceptance criteria:
- [ ] Offline run on `output/duplex_smoke_after_encoder_fix/user.wav` produces a
      conversation.wav whose right channel resolves as intelligible English speech.
- [ ] `frame_log.txt` shows `out_rms ≈ 0.0002` on `[SIL]` frames (silence) and
      `0.01–0.10` on `[SPEECH]` frames.
- [ ] Realtime demo streams without Metal command-buffer assertions during a
      single sustained session (rapid open/close cycles can race the decoder
      thread; this is a known soft-edge).

## Audio Files Generated During Development

These files may still be in `output/` from the development session:

| File | Description |
|------|-------------|
| `test_tts.wav` | First PyTorch MPS test |
| `mimi_mlx_roundtrip.wav` | Mimi codec encode/decode verification |
| `mlx_tts_test.wav` | First MLX TTS (before feedback fix — garbled) |
| `mlx_tts_v4.wav` | After 16-codebook fix |
| `mlx_tts_long.wav` | Working MLX TTS, long text |
| `mlx_preconverted.wav` | From pre-converted MLX model |
| `mlx_speaker_clone.wav` | First speaker-conditioned output |
| `pt_mps_long.wav` | PyTorch MPS reference for comparison |
