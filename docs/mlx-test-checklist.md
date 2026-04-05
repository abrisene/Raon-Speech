# MLX Port Test Checklist

Audio files to evaluate. All generated on M5 Max with hybrid quantization (4-bit thinker, 8-bit talker+CP) unless noted.

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Convert model (one time, ~2s, produces 6.46 GB):
python -m raon_mlx.utils.convert models/Raon-Speech-9B -o models/Raon-Speech-9B-mlx-hybrid

# Generate TTS:
python -m raon_mlx.tts "Your text here" --model models/Raon-Speech-9B-mlx-hybrid -o output/test.wav

# With speaker cloning:
python -m raon_mlx.tts "Your text here" --model models/Raon-Speech-9B-mlx-hybrid --speaker data/duplex/eval/audio/spk_ref.wav -o output/clone.wav

# From HF checkpoint (slower, quantizes on-the-fly):
python -m raon_mlx.tts "Your text here" --model models/Raon-Speech-9B --quant hybrid -o output/test.wav
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

## Known Issues

- HF tokenizer prints `OrderedVocab` warnings to stderr (cosmetic, from Rust tokenizer)
- Audio duration varies — shorter text sometimes produces proportionally shorter audio than expected
- Voice is random each generation without speaker conditioning (expected behavior)
- Korean output may occur occasionally without explicit language conditioning

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
