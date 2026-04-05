# Qwen3.5 Architecture Analysis for Raon-Speech Backbone Upgrade

## Summary

Qwen3.5 is NOT a drop-in replacement for Qwen3. It's a fundamentally different hybrid architecture combining **linear attention (Mamba-style SSM)** with **full attention** layers. This makes the backbone swap significantly more complex than originally estimated.

## Architecture Comparison

### Raon's Current Backbone: Qwen3-8B (pure transformer)

```
36 layers, all full attention
hidden_size: 4096, heads: 32, kv_heads: 8, head_dim: 128
vocab: 153723, rope_theta: 5M
```

Every layer: `RMSNorm → Attention(GQA + QK-norm + RoPE) → RMSNorm → SwiGLU MLP`

### Qwen3.5-9B (hybrid SSM-transformer)

```
32 layers: [linear, linear, linear, full] × 8
hidden_size: 4096, full_attention_interval: 4
Full attn: heads: 16, kv_heads: 4, head_dim: 256
Linear attn: key_heads: 16, value_heads: 32, key_dim: 128, value_dim: 128
vocab: 248320
```

Layer pattern repeats every 4:
- **Linear layers** (3/4 of network): `RMSNorm → LinearAttention(Mamba SSM + conv1d) → RMSNorm → SwiGLU MLP`
- **Full layers** (1/4 of network): `RMSNorm → Attention(GQA + QK-norm + output gate) → RMSNorm → SwiGLU MLP`

### Key Differences

| Feature | Qwen3 | Qwen3.5 |
|---------|-------|---------|
| Architecture | Pure transformer | Hybrid SSM + transformer |
| Layer types | All full attention | 75% linear + 25% full |
| Attention heads | 32 × 128 | 16 × 256 (full only) |
| KV heads | 8 | 4 |
| Layers | 36 | 32 |
| SSM components | None | A_log, dt_bias, conv1d |
| Output gate | No | Yes (`attn_output_gate`) |
| Multi-token prediction | No | Yes (`mtp_num_hidden_layers: 1`) |
| Vocab | 151936 (base) / 153723 (Raon) | 248320 |

## What's Compatible

- `hidden_size: 4096` — **matches** (adaptor dimensions compatible)
- `intermediate_size: 12288` — **matches** (SwiGLU MLP dimensions compatible)
- SwiGLU MLP structure — **identical** (gate_proj, up_proj, down_proj)
- RMSNorm — **identical**
- Full attention layers — **similar** (larger head_dim, fewer heads, has output gate)

## What's New / Incompatible

### 1. Linear Attention (Mamba SSM) — 75% of layers

The linear attention layers use a selective state space model:
- `A_log`: log of the state transition matrix
- `dt_bias`: discretization timestep bias  
- `conv1d`: 1D causal convolution (kernel_size=4)
- `in_proj_qkv/a/b/z`: multiple input projections
- Runs as a recurrence, not attention — no KV cache for these layers

This is the biggest implementation effort. We'd need:
- Mamba-style SSM forward pass in MLX
- State management for streaming (SSM state replaces KV cache)
- `mx.associative_scan` or custom scan for efficient parallel evaluation

### 2. Attention Output Gate

Full attention layers have `attn_output_gate: True` — the attention output is gated before the residual add. Need to check exact implementation.

### 3. Different Head Configuration

Full attention: 16 heads × 256 dim (vs Qwen3's 32 × 128). Same total dimension (4096) but different parallelism.

### 4. Multi-Token Prediction

`mtp_num_hidden_layers: 1` — built-in speculative decoding head. Not needed for inference but the model may have been trained with it, affecting representations.

### 5. Vocabulary Size

248320 vs 153723. The embedding and lm_head dimensions change. Raon's audio special tokens would need to be added to the larger vocab.

## Effort Estimate

| Component | Effort | Notes |
|-----------|--------|-------|
| Linear attention (Mamba SSM) | **2-3 days** | Core new implementation, scan ops |
| Full attention (modified) | **0.5 days** | Adapt existing Qwen3 attn for larger head_dim + output gate |
| Hybrid layer routing | **0.5 days** | Route between linear/full based on layer_types |
| Vocabulary expansion | **0.5 days** | Map Raon audio tokens into larger vocab |
| Adaptor retraining | **1-2 days** | Train adaptors on new backbone (needs GPU) |
| Testing + validation | **1 day** | End-to-end STT/TTS quality verification |
| **Total** | **5-7 days** | Plus training compute for adaptors |

## Recommendation

### Short term: Stay on Qwen3

The current Qwen3-8B backbone works well. The MLX port is fast (2.6x real-time).
Qwen3 is a proven, well-understood architecture.

### Medium term: Implement Qwen3.5 backbone in MLX

The hybrid SSM architecture could provide significant benefits:
- **Faster inference**: linear attention is O(n) not O(n²), and 75% of layers use it
- **Lower memory**: no KV cache for linear layers (SSM state is fixed-size)
- **Better quality**: Qwen3.5-9B likely has better text understanding than Qwen3-8B

The hidden_size matches, so adaptors could be retrained with minimal compute.

### What to build first

1. Implement the Mamba SSM layer in MLX (reusable for other hybrid models)
2. Build hybrid Qwen3.5 model with layer-type routing
3. Test with the local Qwen3.5-0.8B model (quick iteration)
4. Once architecture works, download Qwen3.5-9B and retrain adaptors

## Local Models Available

- `~/models/Text/mlx-community/Qwen3.5-0.8B-MLX-8bit` — small but useful for testing architecture
- Qwen3.5-9B available on HuggingFace (4.8M downloads)
