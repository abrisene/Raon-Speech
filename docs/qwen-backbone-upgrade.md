# Qwen Backbone Upgrade Analysis

## Current Backbone

Raon-Speech-9B uses a **Qwen3-8B** backbone as its "thinker" — the main language model that drives both text understanding and audio token generation.

```
model_type: qwen3
hidden_size: 4096
num_attention_heads: 32
num_key_value_heads: 8   (GQA 4:1)
num_hidden_layers: 36
intermediate_size: 12288  (SwiGLU)
head_dim: 128
rope_theta: 5000000
vocab_size: 153723        (standard Qwen3 vocab + ~1700 audio special tokens)
```

## Why Upgrade?

A stronger backbone means better text understanding, better instruction following, better multilingual capability, and potentially better text-to-speech quality (the backbone decides *what* to say and *how* to express it; the talker and codec handle the audio realization).

## Candidate: Qwen3.5

Qwen3.5 is the expected successor to Qwen3. If it follows the pattern of previous .5 releases:

- **Architecture**: Likely identical to Qwen3 (same attention mechanism, same SwiGLU, same RoPE). The .5 releases typically improve training data, training recipe, and alignment — not the transformer architecture.
- **Sizes**: Likely available at 0.6B, 1.7B, 4B, 8B, 14B, 32B, 72B (same as Qwen3).

### Best-case scenario: Qwen3.5-8B with identical dimensions

If Qwen3.5-8B has hidden_size=4096, 32 heads, 8 KV heads, 36 layers — the same as Qwen3-8B — then the swap is relatively straightforward because all the adaptor layers have matching dimensions.

### What needs retraining

Even with identical architecture, the internal representations differ. The adaptors learned to map *this specific Qwen3-8B's* activation patterns to audio space. New backbone = garbage adaptors.

**Components affected by a same-dimensions swap:**

| Component | Params | Retraining needed? | Why |
|-----------|--------|-------------------|-----|
| text_model (backbone) | ~8B | No — this IS the swap | Frozen, provides new representations |
| input_adaptor | ~17M | **Yes** | Maps audio encoder output → backbone hidden space |
| output_adaptor | ~4M | **Yes** | Maps Mimi embeddings → backbone hidden space |
| thinker→talker projection | ~50M | **Yes** | Projects backbone output → talker input |
| speaker_encoder projection | ~10M | **Yes** | Projects speaker embeddings → backbone hidden space |
| talker (4 Qwen3 layers) | ~500M | Maybe | Receives projected backbone output |
| code_predictor | ~200M | Probably not | Operates downstream of talker |
| audio_encoder (AuT) | ~300M | No | Independent, feeds into input_adaptor |
| Mimi codec | ~30M | No | Independent audio tokenizer |

**Minimum trainable params**: ~80M (adaptors + projection only)
**Recommended trainable params**: ~580M (adaptors + projection + talker)

### Training recipe

**Stage 1: Adaptor alignment** (cheapest)
1. Load Raon-Speech-9B checkpoint
2. Replace `text_model` weights with Qwen3.5-8B
3. Freeze: backbone, audio_encoder, Mimi, code_predictor, speaker_encoder pretrained
4. Unfreeze: input_adaptor, output_adaptor, thinker→talker projection, speaker_encoder projection
5. Train on speech-text pairs until adaptors converge
6. **Cost estimate**: Single A100/H100, hours not days. ~80M trainable params.

**Stage 2: Talker fine-tuning** (recommended)
1. Start from Stage 1 checkpoint
2. Additionally unfreeze talker (4 Qwen3 layers)
3. Train at lower learning rate
4. **Cost estimate**: Single GPU, 1-2 days. ~580M trainable params.

**Stage 3: Full fine-tuning** (optional, expensive)
1. Start from Stage 2 checkpoint
2. Unfreeze everything at very low LR
3. **Cost estimate**: Multi-GPU, multiple days. Full 9B+ params.

KRAFTON's training code (`scripts/train.sh`) supports configurable module freezing, so this is implementable with their existing pipeline.

## Candidate: Larger Qwen3/3.5 (14B, 32B)

### Architecture mismatch

Larger Qwen models have different hidden dimensions:

| Model | hidden_size | heads | KV heads | layers | intermediate |
|-------|-------------|-------|----------|--------|-------------|
| Qwen3-8B (current) | 4096 | 32 | 8 | 36 | 12288 |
| Qwen3-14B | 5120 | 40 | 8 | 48 | 13824 |
| Qwen3-32B | 5120 | 40 | 8 | 64 | 25600 |
| Qwen3-72B | 8192 | 64 | 8 | 80 | 29568 |

**Every component that touches hidden_size breaks:**

- input_adaptor: output_size must change from 4096 → new hidden_size
- output_adaptor: output_size must change from 4096 → new hidden_size
- thinker→talker projection: input_size changes from 4096 → new hidden_size
- speaker_encoder: output_size must change from 4096 → new hidden_size
- text_model embeddings: vocab_size stays 153723, but embedding_dim changes
- lm_head: must match new hidden_size

**The talker and code predictor are dimensionally independent** — they have their own hidden sizes (2048 and 1024 respectively) and connect to the backbone only through the projection layer.

### Training cost

For a larger backbone:
- Cannot reuse any adaptor weights (dimension mismatch)
- Must initialize new adaptors from scratch
- The projection layer becomes the critical bridge
- Training cost scales with backbone size (more activations to compute)
- **Estimate**: Multi-GPU, several days minimum for adaptor training alone

### Memory implications

| Model | float16 size | 4-bit MLX size | Fits 128GB? |
|-------|-------------|----------------|-------------|
| 8B (current) | ~16GB | ~4.5GB | Easily |
| 14B | ~28GB | ~8GB | Yes |
| 32B | ~64GB | ~18GB | Yes |
| 72B | ~144GB | ~40GB | Tight (with overhead) |

The M5 Max with 128GB could theoretically run a 4-bit quantized 32B model with room to spare. A 72B would be very tight.

## Recommendation

### Near-term (after MLX port)

**Qwen3.5-8B swap** — if/when available with matching dimensions:
- Lowest risk, lowest cost
- Improves text quality without architectural surgery
- ~80M params to retrain, single GPU, hours
- Do this first

### Medium-term

**Qwen3.5-14B** — if better text quality is needed:
- Moderate architectural changes (adaptor dimensions)
- ~200M params to retrain from scratch
- Multi-GPU training, a few days
- At 4-bit MLX: ~8GB, runs comfortably on Apple Silicon

### Long-term / research

**Qwen3.5-32B** — maximum quality:
- Same architectural changes as 14B but much more compute
- At 4-bit MLX: ~18GB, still fits in 128GB
- Inference will be slower (more layers to evaluate per token)
- May need 8-bit instead of 4-bit to preserve quality at this scale

## Open Questions

1. **Qwen3.5 release timeline and architecture details** — not yet public as of April 2026
2. **Does backbone quality actually bottleneck speech quality?** The audio is generated by the talker + code predictor, which are smaller independent models. A better backbone helps with *what* to say, not *how* to say it. For pure TTS (text → speech with no reasoning), the backbone may matter less than for SpeechChat.
3. **KRAFTON's training data** — the open-sourced training pipeline is there, but retraining requires access to large-scale speech-text paired data. KRAFTON trained on 1M+ hours. A backbone swap with adaptor-only training may need less data, but this is uncharted.
4. **Vocab compatibility** — Raon adds ~1700 special tokens to the Qwen3 vocabulary. A new Qwen version with a different base vocabulary size would require adjusting the token embedding matrix. This is solvable (initialize new tokens randomly, keep audio special tokens) but adds complexity.
