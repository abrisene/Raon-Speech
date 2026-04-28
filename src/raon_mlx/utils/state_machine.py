# Duplex state machine for MLX inference.
# Ported from src/raon/utils/state_machine.py — full parity Mealy machine.

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

from .special_tokens import (
    AUDIO_END,
    AUDIO_INPUT_PLACEHOLDER,
    AUDIO_OUTPUT_BC,
    AUDIO_OUTPUT_END_PAD,
    AUDIO_OUTPUT_PAD,
    AUDIO_OUTPUT_PLACEHOLDER,
    AUDIO_START,
    BLOCKED_STRUCTURAL,
    DUPLEX_SIL,
    IM_END,
    IM_START,
    SPEAKER_EMBEDDING_PLACEHOLDER,
)


class DuplexPhase(enum.Enum):
    SIL = "SIL"
    SPEECH = "SPEECH"


@dataclass
class DuplexMachineState:
    phase: DuplexPhase
    last_frame_tokens: list[int]

    @property
    def num_input_tokens(self) -> int:
        return len(self.last_frame_tokens)

    @property
    def emitted_audio(self) -> bool:
        return (
            AUDIO_OUTPUT_PLACEHOLDER.id in self.last_frame_tokens
            or AUDIO_START.id in self.last_frame_tokens
        )


@dataclass(frozen=True)
class DuplexStateConfig:
    use_duplex_end_pad: bool = False
    use_sil_token: bool = False
    no_audio_in_sil: bool = False
    sequence_mode: Literal["tua", "uta"] | None = None
    duplex_pad_token_id: int = AUDIO_OUTPUT_PAD.id
    duplex_end_pad_token_id: int = AUDIO_OUTPUT_END_PAD.id
    duplex_sil_token_id: int = DUPLEX_SIL.id
    use_backchannel_token: bool = False
    duplex_bc_token_id: int = AUDIO_OUTPUT_BC.id

    @property
    def effective_sequence_mode(self) -> Literal["tua", "uta"]:
        return self.sequence_mode or "tua"


class DuplexStateManager:
    """Mealy state machine for duplex inference: transitions + logit masking."""

    def __init__(self, config: DuplexStateConfig) -> None:
        self._config = config
        # Logit-mask cache. The mask itself depends only on (phase, context-class)
        # and a fixed vocab size, so we lazily memoize it after first construction
        # and reuse across frames — avoids a 600 KB numpy → mlx copy per frame.
        # Key: (phase_str, context_class, vocab_size)
        self._mask_cache: dict[tuple[str, str, int], "mx.array"] = {}

    @property
    def config(self) -> DuplexStateConfig:
        return self._config

    def initial_state(self, speak_first: bool = False) -> DuplexMachineState:
        return DuplexMachineState(
            phase=DuplexPhase.SIL,
            last_frame_tokens=[AUDIO_INPUT_PLACEHOLDER.id, AUDIO_OUTPUT_PLACEHOLDER.id],
        )

    def initial_forced_prediction_id(self, speak_first: bool) -> int | None:
        cfg = self._config
        if speak_first:
            if cfg.use_duplex_end_pad:
                return cfg.duplex_end_pad_token_id
            return None
        if cfg.use_sil_token:
            return cfg.duplex_sil_token_id
        return None

    def transition(
        self,
        state: DuplexMachineState,
        predicted_id: int,
    ) -> tuple[DuplexMachineState, list[int], bool]:
        """Compute next state and emitted frame tokens from a text prediction.

        Returns:
            (new_state, frame_tokens, emitted_audio)
        """
        cfg = self._config
        aip = AUDIO_INPUT_PLACEHOLDER.id
        aop = AUDIO_OUTPUT_PLACEHOLDER.id
        sil_id = cfg.duplex_sil_token_id
        epad_id = cfg.duplex_end_pad_token_id
        pad_id = cfg.duplex_pad_token_id
        bc_id = cfg.duplex_bc_token_id
        is_uta = cfg.effective_sequence_mode == "uta"

        is_sil_prediction = predicted_id == sil_id

        if state.phase == DuplexPhase.SIL:
            if is_sil_prediction:
                tokens = [aip, aop]
                return DuplexMachineState(DuplexPhase.SIL, tokens), tokens, True

            if cfg.use_duplex_end_pad and predicted_id == epad_id:
                tokens = [aip, epad_id, aop] if is_uta else [epad_id, aip, aop]
                return DuplexMachineState(DuplexPhase.SPEECH, tokens), tokens, True

            if cfg.use_backchannel_token and predicted_id == bc_id:
                tokens = [aip, bc_id, aop] if is_uta else [bc_id, aip, aop]
                return DuplexMachineState(DuplexPhase.SPEECH, tokens), tokens, True

            # SIL -> SPEECH via direct text
            if is_uta:
                tokens = [aip, predicted_id, aop]
            else:
                tokens = [predicted_id, aip, aop]
            return DuplexMachineState(DuplexPhase.SPEECH, tokens), tokens, True

        # SPEECH phase
        if is_sil_prediction:
            tokens = [aip, aop]
            return DuplexMachineState(DuplexPhase.SIL, tokens), tokens, True

        if predicted_id == pad_id:
            tokens = [aip, aop]
            return DuplexMachineState(DuplexPhase.SPEECH, tokens), tokens, True

        if predicted_id == epad_id:
            tokens = [aip, epad_id, aop] if is_uta else [epad_id, aip, aop]
            return DuplexMachineState(DuplexPhase.SPEECH, tokens), tokens, True

        # SPEECH -> SPEECH (text token)
        if is_uta:
            tokens = [aip, predicted_id, aop]
        else:
            tokens = [predicted_id, aip, aop]
        return DuplexMachineState(DuplexPhase.SPEECH, tokens), tokens, True

    def apply_logit_mask(
        self,
        user_logits: mx.array,
        state: DuplexMachineState,
        vocab_size: int,
    ) -> mx.array:
        """Mask logits to enforce valid state-machine transitions.

        Builds an additive mask (0 for allowed, -inf for blocked). The mask only
        depends on (phase, context-class, vocab) and a fixed config — cached
        on the manager to skip the numpy → mlx copy after the first frame.

        Args:
            user_logits: Shape [1, 1, vocab] or [1, seq, vocab].
            state: Current machine state.
            vocab_size: Text vocabulary size (tokens >= this are special).

        Returns:
            Masked logits with invalid tokens set to -inf.
        """
        V = user_logits.shape[-1]
        ctx_class = self._classify_mask_context(state)
        cache_key = (state.phase.value, ctx_class, V)

        mask = self._mask_cache.get(cache_key)
        if mask is None:
            mask = self._build_mask(state.phase, ctx_class, V, vocab_size)
            self._mask_cache[cache_key] = mask
        return user_logits + mask

    def _classify_mask_context(self, state: DuplexMachineState) -> str:
        """Classify the SPEECH-phase context token into one of the mask buckets.

        SIL phase ignores this. For SPEECH we partition into:
          'onset'    — last context was EPAD/BC (only text allowed next)
          'text'     — last context was a regular text token
          'pad'      — last context was PAD/SIL/blocked (only PAD/EPAD/SIL allowed)
          'na'       — SIL phase, classifier unused
        """
        if state.phase == DuplexPhase.SIL:
            return "na"
        cfg = self._config
        epad_id = cfg.duplex_end_pad_token_id
        bc_id = cfg.duplex_bc_token_id
        sil_id = cfg.duplex_sil_token_id
        pad_id = cfg.duplex_pad_token_id

        onset_ids = {epad_id}
        if cfg.use_backchannel_token:
            onset_ids.add(bc_id)

        ctx = self._extract_context_token(state)
        if ctx is None:
            return "pad"
        if ctx in onset_ids:
            return "onset"
        if ctx not in (BLOCKED_STRUCTURAL | onset_ids | {pad_id, sil_id}):
            return "text"
        return "pad"

    def _build_mask(
        self,
        phase: DuplexPhase,
        ctx_class: str,
        V: int,
        vocab_size: int,
    ) -> mx.array:
        """Construct the additive logit mask for a given (phase, context-class)."""
        import numpy as np

        cfg = self._config
        sil_id = cfg.duplex_sil_token_id
        epad_id = cfg.duplex_end_pad_token_id
        pad_id = cfg.duplex_pad_token_id
        bc_id = cfg.duplex_bc_token_id

        onset_ids = {epad_id}
        if cfg.use_backchannel_token:
            onset_ids.add(bc_id)

        mask_np = np.full(V, -np.inf, dtype=np.float32)

        if phase == DuplexPhase.SIL:
            if 0 <= sil_id < V:
                mask_np[sil_id] = 0.0
            if cfg.use_duplex_end_pad and 0 <= epad_id < V:
                mask_np[epad_id] = 0.0
            if cfg.use_backchannel_token and 0 <= bc_id < V:
                mask_np[bc_id] = 0.0
        elif ctx_class == "onset":
            mask_np[:vocab_size] = 0.0
            for block_id in BLOCKED_STRUCTURAL | {sil_id, pad_id} | onset_ids:
                if 0 <= block_id < V:
                    mask_np[block_id] = -np.inf
        elif ctx_class == "text":
            mask_np[:vocab_size] = 0.0
            if 0 <= pad_id < V:
                mask_np[pad_id] = 0.0
            if 0 <= epad_id < V:
                mask_np[epad_id] = 0.0
            if 0 <= sil_id < V:
                mask_np[sil_id] = 0.0
            for block_id in BLOCKED_STRUCTURAL:
                if 0 <= block_id < V:
                    mask_np[block_id] = -np.inf
        else:  # 'pad'
            if 0 <= pad_id < V:
                mask_np[pad_id] = 0.0
            if 0 <= epad_id < V:
                mask_np[epad_id] = 0.0
            if 0 <= sil_id < V:
                mask_np[sil_id] = 0.0

        return mx.array(mask_np)

    def _extract_context_token(self, state: DuplexMachineState) -> int | None:
        tokens = state.last_frame_tokens
        if len(tokens) != 3:
            return None
        is_uta = self._config.effective_sequence_mode == "uta"
        if is_uta:
            return tokens[1]
        else:
            return tokens[0]
