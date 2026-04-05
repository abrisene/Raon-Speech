# Qwen3.5 Backbone Upgrade — Next Steps

## Status (2026-04-05)

Architecture work is complete. The Qwen3.5-9B backbone:
- Loads and runs in MLX at 63 tok/s (8-bit)
- hidden_size=4096 matches all Raon adaptors
- Full pipeline plumbing verified: thinker → projection → talker → audio_lm_head
- Configurable: `RaonMLX(backbone="qwen3.5")` or `RaonMLX(backbone="qwen3")`

**What's missing: adaptor retraining.** The Qwen3.5 backbone produces different internal representations than Qwen3, so the adaptors (which translate between text and audio embedding spaces) need to be retrained.

## Training Plan

### Stage 1: Adaptor-Only Training (~72M params)

**What to train:**
- `input_adaptor` (17M) — maps audio encoder output → thinker space
- `output_adaptor` (4M) — maps Mimi VQ latents → thinker space  
- `thinker_to_talker_proj` (50M) — projects thinker hidden → talker input
- `speaker_encoder.projection` (0.8M) — projects speaker embeddings → thinker space

**What to freeze:**
- Qwen3.5-9B backbone (9B params) — frozen, provides representations
- Talker (500M) — frozen initially
- Code predictor (200M) — frozen
- Mimi codec (30M) — frozen
- Audio encoder / AuT (300M) — frozen

**Hardware:**
- 4090 (24GB): feasible with gradient checkpointing
- Memory estimate: ~22-24GB (frozen backbone fp16 + adaptor gradients + activations)
- Time: 2-4 hours with ~10K hours of data

**Training config changes needed in `scripts/train.sh`:**
```bash
# Freeze everything except adaptors
FREEZE_MODULES="text_model,audio_encoder,audio_tokenizer,talker,code_predictor,audio_lm_head,proj_code"
# Unfreeze adaptors (these are NOT in the freeze list)
# input_adaptor, output_adaptor, thinker_to_talker_proj, speaker_encoder.projection
```

### Stage 2: + Talker Fine-tuning (~572M params)

**Additionally unfreeze:**
- Talker (4 Qwen3 layers, 500M params)

**Hardware:**
- 4090: tight at 24GB, needs gradient checkpointing + 8-bit optimizer
- A100 40GB: comfortable
- Time: 4-8 hours on 4090, 1-2 hours on A100

### Stage 3 (Optional): Full Fine-tune

- Unfreeze everything at very low LR
- Needs A100 80GB+ or multi-GPU
- Only if Stage 2 quality isn't sufficient

## Training Data

### Minimum Viable
- ~1K hours paired speech-text data
- Task distribution: STT, TTS, SpeechChat, TextQA
- At least some English and Korean if bilingual is desired

### Recommended
- ~10K+ hours for good quality
- Available open datasets:
  - **LibriSpeech** (960h, English, clean read speech)
  - **GigaSpeech** (10Kh, English, diverse sources)
  - **Common Voice** (multilingual, variable quality)
  - **VoxPopuli** (400Kh, multilingual, European Parliament)
  - **KsponSpeech** (969h, Korean spontaneous speech)
  - **AI Hub Korean** (various Korean speech corpora)

### Data Format
KRAFTON's training pipeline expects JSONL with:
```json
{
  "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}],
  "audios": ["path/to/audio.wav"],
  "channel": "tts|stt|speech-chat|textqa",
  "speaker_ref_audios": ["path/to/ref.wav"]
}
```

## Vocabulary Mapping

Qwen3.5 has 248,320 tokens vs Raon's 153,723. The audio special tokens need to be added:

| Token | Raon ID | Action |
|-------|---------|--------|
| `<\|audio_start\|>` | 151669 | Map to new ID in Qwen3.5 vocab |
| `<\|audio_end\|>` | 151670 | Map to new ID |
| `<\|audio_output_placeholder\|>` | 151675 | Map to new ID |
| `<\|audio_input_placeholder\|>` | 151676 | Map to new ID |
| `<\|audio_output_pad\|>` | 151677 | Map to new ID |
| `<\|speaker_embedding_placeholder\|>` | 151671 | Map to new ID |
| `<\|im_start\|>` | 151644 | Already in Qwen3.5 (check ID) |
| `<\|im_end\|>` | 151645 | Already in Qwen3.5 (check ID) |

**Implementation:** Extend Qwen3.5's tokenizer with audio special tokens, resize embedding layer, initialize new token embeddings randomly. The `lm_head` and `audio_lm_head` would need corresponding updates.

## Implementation Checklist

### Before Training
- [ ] Write training config for adaptor-only training with frozen Qwen3.5 backbone
- [ ] Map audio special tokens into Qwen3.5 vocabulary
- [ ] Resize embedding layer and lm_head for new vocab
- [ ] Verify training pipeline runs with Qwen3.5 backbone (on small data, 1 epoch)
- [ ] Prepare training data in KRAFTON's JSONL format

### Training
- [ ] Stage 1: Train adaptors only (2-4 hours, 4090)
- [ ] Evaluate: TTS quality, STT accuracy, SpeechChat coherence
- [ ] Stage 2: If needed, fine-tune talker (4-8 hours, 4090 w/ gradient checkpointing)
- [ ] Evaluate again

### After Training
- [ ] Convert trained checkpoint to MLX format
- [ ] Benchmark: compare Qwen3 vs Qwen3.5 on TTS RTF, STT accuracy
- [ ] Update pipeline to auto-detect backbone type from checkpoint config
- [ ] Merge into `feat/mlx-port` if quality is acceptable

## Expected Benefits

| Metric | Qwen3 (current) | Qwen3.5 (projected) |
|--------|-----------------|---------------------|
| Backbone tok/s (4-bit) | 69 | ~80+ (fewer layers, SSM is faster) |
| KV cache memory | O(n) all layers | O(n) for 8 layers, O(1) for 24 |
| Text quality | Good | Better (newer training data) |
| Total model size | 18.1 GB (fp16) | ~18 GB (similar param count) |
| Quantized size | 6.46 GB (hybrid) | ~6.5 GB (similar) |

## Cost Summary

| Approach | Hardware | Time | Cost |
|----------|----------|------|------|
| Stage 1 (adaptors) | 4090 local | 2-4h | ~$0 |
| Stage 1 | Lambda A100 | 1-2h | ~$2-4 |
| Stage 2 (+ talker) | 4090 local | 4-8h | ~$0 |
| Stage 2 | Lambda A100 | 1-2h | ~$3-5 |
| Full fine-tune | A100 80GB | 1-2 days | ~$30-60 |
