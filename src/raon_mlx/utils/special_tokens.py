# Special token definitions for Raon duplex inference.
# Ported from src/raon/utils/special_tokens.py — IDs must match exactly.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialToken:
    id: int
    text: str

    def __int__(self) -> int:
        return self.id

    def __str__(self) -> str:
        return self.text


PAD = SpecialToken(id=151679, text="<|endoftext|>")
IM_START = SpecialToken(id=151644, text="<|im_start|>")
IM_END = SpecialToken(id=151645, text="<|im_end|>")
AUDIO_START = SpecialToken(id=151669, text="<|audio_start|>")
AUDIO_END = SpecialToken(id=151670, text="<|audio_end|>")
SPEAKER_EMBEDDING_PLACEHOLDER = SpecialToken(id=151671, text="<|speaker_embedding_placeholder|>")
DUPLEX_SIL = SpecialToken(id=151672, text="<|audio_output_sil|>")
AUDIO_OUTPUT_BC = SpecialToken(id=151673, text="<|audio_output_backchannel|>")
AUDIO_OUTPUT_PLACEHOLDER = SpecialToken(id=151675, text="<|audio_output_placeholder|>")
AUDIO_INPUT_PLACEHOLDER = SpecialToken(id=151676, text="<|audio_input_placeholder|>")
AUDIO_OUTPUT_PAD = SpecialToken(id=151677, text="<|audio_output_pad|>")
AUDIO_OUTPUT_END_PAD = SpecialToken(id=151678, text="<|audio_output_end_pad|>")

ALL_SPECIAL_TOKENS: list[SpecialToken] = [
    PAD, IM_START, IM_END, AUDIO_START, AUDIO_END,
    SPEAKER_EMBEDDING_PLACEHOLDER, DUPLEX_SIL, AUDIO_OUTPUT_BC,
    AUDIO_OUTPUT_PLACEHOLDER, AUDIO_INPUT_PLACEHOLDER,
    AUDIO_OUTPUT_PAD, AUDIO_OUTPUT_END_PAD,
]

# Structural tokens that must never be sampled as text predictions in duplex mode.
BLOCKED_STRUCTURAL: frozenset[int] = frozenset({
    AUDIO_INPUT_PLACEHOLDER.id,
    AUDIO_OUTPUT_PLACEHOLDER.id,
    AUDIO_START.id,
    AUDIO_END.id,
    IM_START.id,
    IM_END.id,
    SPEAKER_EMBEDDING_PLACEHOLDER.id,
})
