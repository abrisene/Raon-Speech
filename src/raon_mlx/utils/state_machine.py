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

        Builds an additive mask (0 for allowed, -inf for blocked) using
        numpy for construction, then converts to mx.array for the final add.
        Avoids -inf + inf = NaN issues with pure MLX at() operations.

        Args:
            user_logits: Shape [1, 1, vocab] or [1, seq, vocab].
            state: Current machine state.
            vocab_size: Text vocabulary size (tokens >= this are special).

        Returns:
            Masked logits with invalid tokens set to -inf.
        """
        import numpy as np

        cfg = self._config
        sil_id = cfg.duplex_sil_token_id
        epad_id = cfg.duplex_end_pad_token_id
        pad_id = cfg.duplex_pad_token_id
        bc_id = cfg.duplex_bc_token_id

        onset_ids = {epad_id}
        if cfg.use_backchannel_token:
            onset_ids.add(bc_id)

        V = user_logits.shape[-1]
        # Build mask in numpy: 0.0 = allowed, -inf = blocked
        mask_np = np.full(V, -np.inf, dtype=np.float32)

        if state.phase == DuplexPhase.SIL:
            # Only SIL, EPAD, and optionally BC allowed
            if 0 <= sil_id < V:
                mask_np[sil_id] = 0.0
            if cfg.use_duplex_end_pad and 0 <= epad_id < V:
                mask_np[epad_id] = 0.0
            if cfg.use_backchannel_token and 0 <= bc_id < V:
                mask_np[bc_id] = 0.0

        elif state.phase == DuplexPhase.SPEECH:
            context_token = self._extract_context_token(state)

            if context_token is not None and context_token in onset_ids:
                # After EPAD/BC onset: only text tokens allowed
                mask_np[:vocab_size] = 0.0
                # Block structural + control tokens
                for block_id in BLOCKED_STRUCTURAL | {sil_id, pad_id} | onset_ids:
                    if 0 <= block_id < V:
                        mask_np[block_id] = -np.inf

            elif (
                context_token is not None
                and context_token not in (BLOCKED_STRUCTURAL | onset_ids | {pad_id, sil_id})
            ):
                # After text: text + PAD + EPAD + SIL allowed
                mask_np[:vocab_size] = 0.0
                if 0 <= pad_id < V:
                    mask_np[pad_id] = 0.0
                if 0 <= epad_id < V:
                    mask_np[epad_id] = 0.0
                if 0 <= sil_id < V:
                    mask_np[sil_id] = 0.0
                # Block structural tokens
                for block_id in BLOCKED_STRUCTURAL:
                    if 0 <= block_id < V:
                        mask_np[block_id] = -np.inf

            else:
                # PAD frame: PAD + EPAD + SIL allowed
                if 0 <= pad_id < V:
                    mask_np[pad_id] = 0.0
                if 0 <= epad_id < V:
                    mask_np[epad_id] = 0.0
                if 0 <= sil_id < V:
                    mask_np[sil_id] = 0.0

        # Broadcast to match user_logits shape and add
        mask = mx.array(mask_np)
        return user_logits + mask

    def _extract_context_token(self, state: DuplexMachineState) -> int | None:
        tokens = state.last_frame_tokens
        if len(tokens) != 3:
            return None
        is_uta = self._config.effective_sequence_mode == "uta"
        if is_uta:
            return tokens[1]
        else:
            return tokens[0]
