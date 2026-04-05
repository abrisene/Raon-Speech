# Raon-Speech full model assembly for MLX.
# Combines: thinker (Qwen3), talker (Qwen3), code predictor (Qwen3),
# thinker-to-talker projection, audio_lm_head, proj_code, Mimi codec.

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .mimi import Mimi, MimiConfig, mimi_raon
from .qwen3 import Qwen3Config, Qwen3Model, Qwen3RMSNorm, Qwen3DecoderLayer
from ..modules.kv_cache import KVCache, create_additive_causal_mask


# ---- Configs ----

def thinker_config() -> Qwen3Config:
    """36-layer Qwen3-8B backbone."""
    return Qwen3Config(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=36,
        intermediate_size=12288,
        head_dim=128,
        vocab_size=153723,
        rms_norm_eps=1e-6,
        rope_theta=5000000.0,
    )


def talker_config() -> Qwen3Config:
    """4-layer Qwen3 talker."""
    return Qwen3Config(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_hidden_layers=4,
        intermediate_size=6144,
        head_dim=128,
        vocab_size=153723,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )


def code_predictor_config() -> Qwen3Config:
    """5-layer code predictor."""
    return Qwen3Config(
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_hidden_layers=5,
        intermediate_size=3072,
        head_dim=128,
        vocab_size=2048,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
    )


# ---- Sub-models ----

class OutputAdaptor(nn.Module):
    """2-layer MLP that projects Mimi VQ latents (512) to thinker embedding space (4096).

    During autoregressive audio generation, the thinker receives this instead of
    the raw AUDIO_OUTPUT_PLACEHOLDER token embedding.
    """

    def __init__(self, input_size: int = 512, output_size: int = 4096, norm_eps: float = 1e-6):
        super().__init__()
        self.proj_0 = nn.Linear(input_size, output_size, bias=False)
        self.proj_2 = nn.Linear(output_size, output_size, bias=False)
        self.post_norm = Qwen3RMSNorm(output_size, eps=norm_eps)

    def __call__(self, xs: mx.array) -> mx.array:
        xs = nn.gelu(self.proj_0(xs))
        xs = self.proj_2(xs)
        return self.post_norm(xs)


class ThinkerToTalkerProjection(nn.Module):
    """MLP projection from thinker hidden_size (4096) to talker hidden_size (2048)."""

    def __init__(self, thinker_dim: int = 4096, talker_dim: int = 2048, intermediate: int = 6144):
        super().__init__()
        self.linear_fc1 = nn.Linear(thinker_dim, intermediate, bias=True)
        self.linear_fc2 = nn.Linear(intermediate, talker_dim, bias=True)

    def __call__(self, xs: mx.array) -> mx.array:
        return self.linear_fc2(nn.silu(self.linear_fc1(xs)))


class Talker(nn.Module):
    """4-layer Qwen3 talker that continues generation from projected thinker output.

    Shares the thinker's KV cache positions but has its own layers.
    """

    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        self.layers = [Qwen3DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(
        self,
        xs: mx.array,
        cache: list[KVCache] | None = None,
        mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        cache_position: mx.array | None = None,
    ) -> mx.array:
        if mask is None and xs.shape[1] > 1:
            if cache_position is not None:
                offset = int(cache_position[0].item())
            else:
                offset = cache[0].offset if cache is not None else 0
            mask = create_additive_causal_mask(xs.shape[1], offset)
            mask = mask.astype(xs.dtype)

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            xs = layer(xs, cache=layer_cache, mask=mask, position_ids=position_ids, cache_position=cache_position)
        return self.norm(xs)

    def make_cache(self) -> list[KVCache]:
        return [
            KVCache(head_dim=self.cfg.head_dim, n_kv_heads=self.cfg.num_key_value_heads)
            for _ in self.layers
        ]


class CodePredictor(nn.Module):
    """Predicts audio codes from talker hidden states.

    Architecture: 5-layer Qwen3-style transformer with per-codebook prediction heads.
    - codec_embedding: maps previous codebook codes to embeddings (16 groups × 2048 codes)
    - model: 5-layer transformer
    - fused_lm_head: [15, 2048, 1024] — prediction heads for codebooks 2-16
    - First codebook uses audio_lm_head (separate, on the parent model)
    """

    def __init__(self, cfg: Qwen3Config, num_code_groups: int = 16, codebook_size: int = 2048):
        super().__init__()
        self.cfg = cfg
        self.num_code_groups = num_code_groups
        self.codebook_size = codebook_size

        # Embedding for all codebook codes: 16 groups × 2048 = 32768 entries
        self.codec_embedding = nn.Embedding(num_code_groups * codebook_size, cfg.hidden_size)

        # 5-layer transformer
        self.model = _CodePredictorTransformer(cfg)

        # Fused prediction heads for codebooks 2-16 (15 heads)
        # Shape: [15, codebook_size, hidden_size] — stored as a 3D weight
        self.fused_lm_head = mx.zeros((num_code_groups - 1, codebook_size, cfg.hidden_size))

    def make_cache(self) -> list[KVCache]:
        return self.model.make_cache()


class _CodePredictorTransformer(nn.Module):
    """Inner transformer for the code predictor."""

    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        self.layers = [Qwen3DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        cache: list[KVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        if inputs_embeds is not None:
            xs = inputs_embeds
        else:
            raise ValueError("_CodePredictorTransformer requires inputs_embeds")

        if mask is None and xs.shape[1] > 1:
            offset = cache[0].offset if cache is not None else 0
            mask = create_additive_causal_mask(xs.shape[1], offset)
            mask = mask.astype(xs.dtype)

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            xs = layer(xs, cache=layer_cache, mask=mask)
        return self.norm(xs)

    def make_cache(self) -> list[KVCache]:
        return [
            KVCache(head_dim=self.cfg.head_dim, n_kv_heads=self.cfg.num_key_value_heads)
            for _ in self.layers
        ]


# ---- Full Model ----

class RaonMLX(nn.Module):
    """Full Raon-Speech model in MLX.

    Components:
    - thinker: 36-layer Qwen3 backbone
    - talker: 4-layer Qwen3
    - thinker_to_talker_proj: MLP 4096 -> 2048
    - code_predictor: 5-layer transformer + codebook heads
    - proj_code: linear 2048 -> 1024
    - audio_lm_head: linear 2048 -> 2049 (first codebook logits)
    - lm_head: linear 4096 -> 153723 (text logits)
    - mimi: audio codec for decoding generated codes to PCM
    """

    def __init__(
        self,
        thinker_cfg: Qwen3Config | None = None,
        talker_cfg: Qwen3Config | None = None,
        cp_cfg: Qwen3Config | None = None,
        mimi_cfg: MimiConfig | None = None,
    ):
        super().__init__()
        thinker_cfg = thinker_cfg or thinker_config()
        talker_cfg = talker_cfg or talker_config()
        cp_cfg = cp_cfg or code_predictor_config()
        mimi_cfg = mimi_cfg or mimi_raon()

        self.thinker = Qwen3Model(thinker_cfg)
        self.talker = Talker(talker_cfg)
        self.thinker_to_talker_proj = ThinkerToTalkerProjection(
            thinker_dim=thinker_cfg.hidden_size,
            talker_dim=talker_cfg.hidden_size,
            intermediate=6144,
        )
        self.code_predictor = CodePredictor(cp_cfg)
        self.proj_code = nn.Linear(talker_cfg.hidden_size, cp_cfg.hidden_size, bias=True)
        self.audio_lm_head = nn.Linear(talker_cfg.hidden_size, 2049, bias=False)
        self.lm_head = nn.Linear(thinker_cfg.hidden_size, thinker_cfg.vocab_size, bias=False)
        self.input_adaptor = OutputAdaptor(input_size=2048, output_size=thinker_cfg.hidden_size)  # Same arch as output adaptor
        self.output_adaptor = OutputAdaptor(input_size=512, output_size=thinker_cfg.hidden_size)
        self.speaker_projection = nn.Linear(192, thinker_cfg.hidden_size, bias=False)
        self.mimi = Mimi(mimi_cfg)

        # Configs for reference
        self.thinker_cfg = thinker_cfg
        self.talker_cfg = talker_cfg
        self.cp_cfg = cp_cfg

    def load_mlx_weights(self, mlx_model_path: str):
        """Load pre-converted MLX weights (from convert.py).

        Reads config.json to determine quantization, applies matching quantization
        to the model structure (so weight shapes match), then loads the saved weights.

        Args:
            mlx_model_path: Path to directory containing model.safetensors + config.json from convert.
        """
        import json
        from pathlib import Path
        from ..modules.quantization import EuclideanCodebook
        from ..modules.conv import ConvTranspose1d

        config_path = Path(mlx_model_path) / "config.json"
        with open(config_path) as f:
            cfg = json.load(f)

        # Apply quantization to match saved weight shapes
        thinker_bits = cfg.get("thinker_bits", 16)
        talker_bits = cfg.get("talker_bits", 16)
        cp_bits = cfg.get("cp_bits", 16)

        if thinker_bits in (4, 8):
            nn.quantize(self.thinker, bits=thinker_bits, group_size=64)
        if talker_bits in (4, 8):
            nn.quantize(self.talker, bits=talker_bits, group_size=64)
        if cp_bits in (4, 8):
            nn.quantize(self.code_predictor.model, bits=cp_bits, group_size=64)

        weights_path = Path(mlx_model_path) / "model.safetensors"
        weights = mx.load(str(weights_path))
        self.load_weights(list(weights.items()), strict=False)

        # Post-load fixups for codebook and conv transpose
        def _post_load(module, name, _):
            if isinstance(module, EuclideanCodebook) and name == "initialized":
                module.update_in_place()
            if isinstance(module, ConvTranspose1d) and name == "weight":
                module.update_in_place()
            return True
        self.filter_and_map(_post_load)

    def load_weights_from_raon(self, model_path: str):
        """Load all weights from a Raon-Speech HF checkpoint directory.

        Args:
            model_path: Path to model directory containing safetensors files.
        """
        import json
        from pathlib import Path

        index_path = Path(model_path) / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)

        # Collect all weights by component
        components: dict[str, dict[str, mx.array]] = {
            "text_model": {},
            "talker": {},
            "code_predictor": {},
            "audio_tokenizer": {},
            "thinker_to_talker_proj": {},
            "top_level": {},
        }

        loaded_shards: set[str] = set()
        for key, shard in index["weight_map"].items():
            if shard not in loaded_shards:
                all_w = mx.load(str(Path(model_path) / shard))
                for k, v in all_w.items():
                    if k.startswith("text_model."):
                        components["text_model"][k.removeprefix("text_model.")] = v
                    elif k.startswith("talker."):
                        components["talker"][k.removeprefix("talker.")] = v
                    elif k.startswith("code_predictor."):
                        components["code_predictor"][k.removeprefix("code_predictor.")] = v
                    elif k.startswith("audio_tokenizer."):
                        components["audio_tokenizer"][k.removeprefix("audio_tokenizer.")] = v.astype(mx.float32)
                    elif k.startswith("thinker_to_talker_proj."):
                        components["thinker_to_talker_proj"][k.removeprefix("thinker_to_talker_proj.")] = v
                    else:
                        components["top_level"][k] = v
                del all_w
                loaded_shards.add(shard)

        # Load thinker (Qwen3 backbone)
        self.thinker.load_raon_weights(components["text_model"])

        # Load talker — keys already match (layers.N.self_attn.q_proj.weight etc.)
        self.talker.load_weights(list(components["talker"].items()), strict=False)

        # Load thinker-to-talker projection
        self.thinker_to_talker_proj.load_weights(
            list(components["thinker_to_talker_proj"].items()), strict=False
        )

        # Load code predictor — need to handle fused_lm_head (3D tensor)
        cp_mapped = []
        for k, v in components["code_predictor"].items():
            cp_mapped.append((k, v))
        self.code_predictor.load_weights(cp_mapped, strict=False)

        # Load Mimi codec
        self.mimi.load_raon_weights(components["audio_tokenizer"], strict=False)

        # Load input adaptor
        ia = {k.removeprefix("input_adaptor."): v
              for k, v in components["top_level"].items() if k.startswith("input_adaptor.")}
        if ia:
            ia_mapped = []
            for k, v in ia.items():
                k = k.replace("proj.0.", "proj_0.").replace("proj.2.", "proj_2.")
                ia_mapped.append((k, v))
            self.input_adaptor.load_weights(ia_mapped, strict=False)

        # Load output adaptor
        oa = {k.removeprefix("output_adaptor."): v
              for k, v in components["top_level"].items() if k.startswith("output_adaptor.")}
        if oa:
            # Map HF keys: proj.0.weight -> proj_0.weight, proj.2.weight -> proj_2.weight
            oa_mapped = []
            for k, v in oa.items():
                k = k.replace("proj.0.", "proj_0.").replace("proj.2.", "proj_2.")
                oa_mapped.append((k, v))
            self.output_adaptor.load_weights(oa_mapped, strict=False)

        # Load top-level weights
        top = components["top_level"]
        if "lm_head.weight" in top:
            self.lm_head.load_weights([("weight", top["lm_head.weight"])], strict=False)
        if "audio_lm_head.weight" in top:
            self.audio_lm_head.load_weights([("weight", top["audio_lm_head.weight"])], strict=False)
        if "proj_code.weight" in top:
            self.proj_code.load_weights([
                ("weight", top["proj_code.weight"]),
                ("bias", top["proj_code.bias"]),
            ], strict=False)
        if "speaker_encoder.projection.weight" in top:
            self.speaker_projection.load_weights([
                ("weight", top["speaker_encoder.projection.weight"]),
            ], strict=False)
