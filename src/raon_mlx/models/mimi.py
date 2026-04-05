# Mimi audio codec model for MLX.
# Adapted from PersonaPlex/Kyutai Moshi MLX port.
# Original: Copyright (c) Kyutai, all rights reserved.

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ..modules import (
    ConvDownsample1d,
    ConvTrUpsample1d,
    EuclideanCodebook,
    ProjectedTransformer,
    SeanetConfig,
    SeanetDecoder,
    SeanetEncoder,
    SplitResidualVectorQuantizer,
    TransformerConfig,
)
from ..modules.conv import ConvTranspose1d


@dataclass
class MimiConfig:
    channels: int
    sample_rate: float
    frame_rate: float
    renormalize: bool
    seanet: SeanetConfig
    transformer: TransformerConfig
    quantizer_nq: int
    quantizer_bins: int
    quantizer_dim: int


def mimi_raon() -> MimiConfig:
    """Mimi config matching Raon-Speech-9B's audio_tokenizer_config."""
    seanet = SeanetConfig(
        dimension=512,
        channels=1,
        causal=True,
        nfilters=64,
        nresidual_layers=1,
        ratios=[8, 6, 5, 4],
        ksize=7,
        residual_ksize=3,
        last_ksize=3,
        dilation_base=2,
        pad_mode="constant",
        true_skip=True,
        compress=2,
    )
    transformer = TransformerConfig(
        d_model=seanet.dimension,
        num_heads=8,
        num_layers=8,
        causal=True,
        norm_first=True,
        bias_ff=False,
        bias_attn=False,
        layer_scale=0.01,
        positional_embedding="rope",
        use_conv_bias=True,
        gating=False,
        norm="layer_norm",
        context=250,
        max_period=10000,
        max_seq_len=8192,
        kv_repeat=1,
        dim_feedforward=2048,
        conv_layout=True,
        use_conv_block=False,
        cross_attention=False,
        conv_kernel_size=3,
    )
    return MimiConfig(
        channels=1,
        sample_rate=24000,
        frame_rate=12.5,
        renormalize=True,
        seanet=seanet,
        transformer=transformer,
        quantizer_nq=32,  # Raon uses 32 quantizers (1 semantic + 31 acoustic)
        quantizer_bins=2048,
        quantizer_dim=256,
    )


class Mimi(nn.Module):
    def __init__(self, cfg: MimiConfig):
        super().__init__()
        dim = cfg.seanet.dimension
        self.cfg = cfg
        encoder_frame_rate = cfg.sample_rate / math.prod(cfg.seanet.ratios)
        downsample_stride = int(encoder_frame_rate / cfg.frame_rate)
        self.encoder = SeanetEncoder(cfg.seanet)
        self.decoder = SeanetDecoder(cfg.seanet)
        self.quantizer = SplitResidualVectorQuantizer(
            dim=cfg.quantizer_dim,
            input_dim=dim,
            output_dim=dim,
            nq=cfg.quantizer_nq,
            bins=cfg.quantizer_bins,
        )
        self.encoder_transformer = ProjectedTransformer(
            cfg.transformer, input_dim=dim, output_dims=[dim],
        )
        self.decoder_transformer = ProjectedTransformer(
            cfg.transformer, input_dim=dim, output_dims=[dim],
        )
        self.downsample = ConvDownsample1d(stride=downsample_stride, dim=dim, causal=True)
        self.upsample = ConvTrUpsample1d(stride=downsample_stride, dim=dim, causal=True)
        self.encoder_cache = self.encoder_transformer.make_cache()
        self.decoder_cache = self.decoder_transformer.make_cache()

    def reset_state(self):
        self.encoder.reset_state()
        self.decoder.reset_state()
        for c in self.decoder_cache:
            c.reset()
        for c in self.encoder_cache:
            c.reset()

    def reset_all(self):
        self.reset_state()
        self.upsample.reset_state()
        self.downsample.reset_state()

    def encode(self, xs: mx.array) -> mx.array:
        self.encoder.reset_state()
        for c in self.encoder_cache:
            c.reset()
        xs = self.encoder(xs)
        xs = self.encoder_transformer(xs, cache=self.encoder_cache)[0]
        xs = self.downsample(xs)
        return self.quantizer.encode(xs)

    def decode(self, xs: mx.array) -> mx.array:
        self.decoder.reset_state()
        for c in self.decoder_cache:
            c.reset()
        xs = self.quantizer.decode(xs)
        xs = self.upsample(xs)
        xs = self.decoder_transformer(xs, cache=self.decoder_cache)[0]
        return self.decoder(xs)

    def encode_step(self, xs: mx.array) -> mx.array:
        xs = self.encoder.step(xs)
        xs = self.encoder_transformer(xs, cache=self.encoder_cache)[0]
        xs = self.downsample.step(xs)
        xs = self.quantizer.encode(xs)
        return xs

    def decode_step(self, xs: mx.array) -> mx.array:
        xs = self.quantizer.decode(xs)
        xs = self.upsample.step(xs)
        xs = self.decoder_transformer(xs, cache=self.decoder_cache)[0]
        xs = self.decoder.step(xs)
        return xs

    def warmup(self):
        pcm = mx.zeros((1, 1, 1920 * 4))
        codes = self.encode(pcm)
        pcm_out = self.decode(codes)
        mx.synchronize()

    @property
    def frame_rate(self) -> float:
        return self.cfg.frame_rate

    @property
    def sample_rate(self) -> float:
        return self.cfg.sample_rate

    def load_raon_weights(self, weights: dict[str, mx.array], strict: bool = True):
        """Load Mimi weights extracted from a Raon checkpoint.

        Args:
            weights: Dict of {key: mx.array} with keys already stripped of the
                'audio_tokenizer.' prefix.
            strict: If True, raise on missing/unexpected keys.
        """
        # First pass: fuse q/k/v projections into in_proj for each transformer layer
        qkv_groups: dict[str, dict[str, mx.array]] = {}
        for k, v in weights.items():
            if ".self_attn.q_proj." in k or ".self_attn.k_proj." in k or ".self_attn.v_proj." in k:
                # Group by transformer prefix + layer index
                parts = k.split(".self_attn.")
                group_key = parts[0]  # e.g. "decoder_transformer.layers.0"
                proj_type = parts[1].split(".")[0]  # q_proj, k_proj, v_proj
                if group_key not in qkv_groups:
                    qkv_groups[group_key] = {}
                qkv_groups[group_key][proj_type] = v

        # Build fused in_proj weights
        fused_qkv: list[tuple[str, mx.array]] = []
        for group_key, projs in qkv_groups.items():
            if len(projs) == 3:
                # Fuse q, k, v into single in_proj: [q; k; v] along output dim
                fused = mx.concatenate([projs["q_proj"], projs["k_proj"], projs["v_proj"]], axis=0)
                prefix = "decoder_transformer" if group_key.startswith("decoder") else "encoder_transformer"
                idx = group_key.split("layers.")[1].split(".")[0]
                mlx_key = f"{prefix}.transformer.layers.{idx}.self_attn.in_proj.weight"
                fused_qkv.append((mlx_key, fused))

        # Second pass: map all other keys
        mapped = list(fused_qkv)
        for k, v in weights.items():
            new_k = _map_hf_mimi_key(k)
            if new_k is None:
                continue
            # Conv weight transposition: HF (outC, inC, kSize) -> MLX (outC, kSize, inC)
            if new_k.endswith(".conv.weight") or new_k.endswith(".input_proj.weight") or new_k.endswith(".output_proj.weight"):
                if v.ndim == 3:
                    v = v.swapaxes(-1, -2)
            # ConvTranspose: HF (inC, outC, kSize) -> MLX (outC, kSize, inC)
            if new_k.endswith(".convtr.weight"):
                if v.ndim == 3:
                    v = v.transpose(1, 2, 0)
            mapped.append((new_k, v))

        self.load_weights(mapped, strict=strict)

        # Post-load: recompute codebook embeddings and expanded conv transpose weights
        def _post_load(module, name, _):
            if isinstance(module, EuclideanCodebook) and name == "initialized":
                module.update_in_place()
            if isinstance(module, ConvTranspose1d) and name == "weight":
                module.update_in_place()
            return True

        self.filter_and_map(_post_load)


def _map_hf_mimi_key(k: str) -> str | None:
    """Map a HuggingFace MimiModel key to the MLX Mimi key namespace.

    HF MimiModel key format (after stripping 'audio_tokenizer.' prefix):
      - decoder.layers.{flat_idx}.conv.{weight|bias}
      - decoder.layers.{flat_idx}.block.{1|3}.conv.{weight|bias}
      - decoder_transformer.layers.{N}.self_attn.{q|k|v|o}_proj.weight
      - decoder_transformer.layers.{N}.input_layernorm.{weight|bias}
      - decoder_transformer.layers.{N}.mlp.fc1.weight
      - decoder_transformer.layers.{N}.self_attn_layer_scale.scale
      - quantizer.semantic_residual_vector_quantizer.*
      - quantizer.acoustic_residual_vector_quantizer.*
      - downsample.conv.weight
      - upsample.conv.weight

    MLX Mimi key format:
      - decoder.init_conv1d.conv.conv.{weight|bias}
      - decoder.layers.{N}.upsample.convtr.convtr.{weight|bias}
      - decoder.layers.{N}.residuals.0.block.{0|1}.conv.conv.{weight|bias}
      - decoder_transformer.transformer.layers.{N}.self_attn.in_proj.weight (fused qkv)
      - decoder_transformer.transformer.layers.{N}.norm1.{weight|bias}
      - decoder_transformer.transformer.layers.{N}.gating.linear1.weight
      - quantizer.rvq_first.*
      - quantizer.rvq_rest.*
      - downsample.conv.conv.conv.weight
      - upsample.convtr.convtr.convtr.weight
    """
    # --- Seanet encoder/decoder: flat sequential indexing → structured layers ---

    # Decoder: flat indices 0=init, {2,5,8,11}=upsample, {3,6,9,12}=residual block, 14=final
    if k.startswith("decoder.layers."):
        rest = k.removeprefix("decoder.layers.")
        idx_str = rest.split(".")[0]
        idx = int(idx_str)
        suffix = rest[len(idx_str) + 1:]  # everything after the index

        if idx == 0:
            return f"decoder.init_conv1d.conv.{suffix}"
        elif idx == 14:
            return f"decoder.final_conv1d.conv.{suffix}"
        # Upsample ConvTranspose at positions 2, 5, 8, 11
        elif idx in (2, 5, 8, 11):
            layer_idx = {2: 0, 5: 1, 8: 2, 11: 3}[idx]
            # HF stores as .conv.{weight|bias} but MLX model is .convtr.convtr.{weight|bias}
            suffix = suffix.replace("conv.", "convtr.")
            return f"decoder.layers.{layer_idx}.upsample.convtr.{suffix}"
        # Residual blocks at positions 3, 6, 9, 12
        elif idx in (3, 6, 9, 12):
            layer_idx = {3: 0, 6: 1, 9: 2, 12: 3}[idx]
            # block.1 -> block.0, block.3 -> block.1
            suffix = suffix.replace("block.1.", "block.0.conv.").replace("block.3.", "block.1.conv.")
            return f"decoder.layers.{layer_idx}.residuals.0.{suffix}"
        return None

    # Encoder: flat indices 0=init, {1,4,7,10}=residual, {3,6,9,12}=downsample, 14=final
    if k.startswith("encoder.layers."):
        rest = k.removeprefix("encoder.layers.")
        idx_str = rest.split(".")[0]
        idx = int(idx_str)
        suffix = rest[len(idx_str) + 1:]

        if idx == 0:
            return f"encoder.init_conv1d.conv.{suffix}"
        elif idx == 14:
            return f"encoder.final_conv1d.conv.{suffix}"
        # Residual blocks at positions 1, 4, 7, 10
        elif idx in (1, 4, 7, 10):
            layer_idx = {1: 0, 4: 1, 7: 2, 10: 3}[idx]
            suffix = suffix.replace("block.1.", "block.0.conv.").replace("block.3.", "block.1.conv.")
            return f"encoder.layers.{layer_idx}.residuals.0.{suffix}"
        # Downsample convs at positions 3, 6, 9, 12
        elif idx in (3, 6, 9, 12):
            layer_idx = {3: 0, 6: 1, 9: 2, 12: 3}[idx]
            # HF .conv.{w|b} -> MLX .conv.conv.{w|b} (StreamableConv1d -> NormConv1d -> Conv1d)
            return f"encoder.layers.{layer_idx}.downsample.conv.{suffix}"
        return None

    # --- Transformers: separate q/k/v/o projections → fused in_proj ---
    # Handled specially in load_raon_weights, not here. Return None for individual
    # q/k/v proj keys — they get fused externally.
    if ".self_attn.q_proj." in k or ".self_attn.k_proj." in k or ".self_attn.v_proj." in k:
        return None  # handled by fusion in load_raon_weights

    if ".self_attn.o_proj." in k:
        # o_proj -> out_proj, and nest inside transformer.layers
        prefix = "decoder_transformer" if k.startswith("decoder") else "encoder_transformer"
        rest = k.split("layers.")[1]
        idx = rest.split(".")[0]
        return f"{prefix}.transformer.layers.{idx}.self_attn.out_proj.weight"

    # Layer norms
    if ".input_layernorm." in k:
        prefix = "decoder_transformer" if k.startswith("decoder") else "encoder_transformer"
        rest = k.split("layers.")[1]
        idx = rest.split(".")[0]
        param = "weight" if "weight" in k else "bias"
        return f"{prefix}.transformer.layers.{idx}.norm1.{param}"

    if ".post_attention_layernorm." in k:
        prefix = "decoder_transformer" if k.startswith("decoder") else "encoder_transformer"
        rest = k.split("layers.")[1]
        idx = rest.split(".")[0]
        param = "weight" if "weight" in k else "bias"
        return f"{prefix}.transformer.layers.{idx}.norm2.{param}"

    # MLP: mlp.fc1 -> gating.linear1, mlp.fc2 -> gating.linear2
    if ".mlp.fc1." in k:
        prefix = "decoder_transformer" if k.startswith("decoder") else "encoder_transformer"
        rest = k.split("layers.")[1]
        idx = rest.split(".")[0]
        return f"{prefix}.transformer.layers.{idx}.gating.linear1.weight"

    if ".mlp.fc2." in k:
        prefix = "decoder_transformer" if k.startswith("decoder") else "encoder_transformer"
        rest = k.split("layers.")[1]
        idx = rest.split(".")[0]
        return f"{prefix}.transformer.layers.{idx}.gating.linear2.weight"

    # Layer scales
    if ".self_attn_layer_scale." in k:
        prefix = "decoder_transformer" if k.startswith("decoder") else "encoder_transformer"
        rest = k.split("layers.")[1]
        idx = rest.split(".")[0]
        return f"{prefix}.transformer.layers.{idx}.layer_scale_1.scale"

    if ".mlp_layer_scale." in k:
        prefix = "decoder_transformer" if k.startswith("decoder") else "encoder_transformer"
        rest = k.split("layers.")[1]
        idx = rest.split(".")[0]
        return f"{prefix}.transformer.layers.{idx}.layer_scale_2.scale"

    # --- Quantizer ---
    if k.startswith("quantizer.semantic_residual_vector_quantizer."):
        rest = k.removeprefix("quantizer.semantic_residual_vector_quantizer.")
        rest = rest.replace("codebook.embed_sum", "codebook.embedding_sum")
        # HF layers.N -> MLX vq.layers.N (add vq. prefix for codebook layers)
        if rest.startswith("layers."):
            rest = "vq." + rest
        return f"quantizer.rvq_first.{rest}"

    if k.startswith("quantizer.acoustic_residual_vector_quantizer."):
        rest = k.removeprefix("quantizer.acoustic_residual_vector_quantizer.")
        rest = rest.replace("codebook.embed_sum", "codebook.embedding_sum")
        if rest.startswith("layers."):
            rest = "vq." + rest
        return f"quantizer.rvq_rest.{rest}"

    # --- Downsample / Upsample (between encoder/decoder and transformer) ---
    # ConvDownsample1d: .conv (StreamableConv1d) -> .conv (NormConv1d) -> .conv (Conv1d)
    if k == "downsample.conv.weight":
        return "downsample.conv.conv.conv.weight"
    # ConvTrUpsample1d: .convtr (StreamableConvTranspose1d) -> .convtr (NormConvTranspose1d) -> .convtr (ConvTranspose1d)
    if k == "upsample.conv.weight":
        return "upsample.convtr.convtr.convtr.weight"

    return None
