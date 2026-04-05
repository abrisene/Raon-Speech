# Realtime binary message protocol for duplex demo sessions.
# Copied from demo/realtime/protocol/messages.py — framework-agnostic.

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np


class MessageKind(enum.IntEnum):
    READY = 0x00
    AUDIO = 0x01
    TEXT = 0x02
    ERROR = 0x05
    CLOSE = 0x06
    PING = 0x07
    PONG = 0x08


@dataclass(slots=True)
class Frame:
    kind: MessageKind
    payload: bytes

    def encode(self) -> bytes:
        return bytes([self.kind]) + self.payload

    @classmethod
    def decode(cls, data: bytes) -> Frame:
        if len(data) < 1:
            raise ValueError("Empty frame")
        return cls(kind=MessageKind(data[0]), payload=data[1:])

    @classmethod
    def ready(cls) -> Frame:
        return cls(kind=MessageKind.READY, payload=b"")

    @classmethod
    def audio(cls, pcm: np.ndarray) -> Frame:
        arr = np.asarray(pcm, dtype=np.float32).reshape(-1)
        return cls(kind=MessageKind.AUDIO, payload=arr.tobytes())

    @classmethod
    def text(cls, content: str) -> Frame:
        return cls(kind=MessageKind.TEXT, payload=content.encode("utf-8"))

    @classmethod
    def error(cls, message: str) -> Frame:
        return cls(kind=MessageKind.ERROR, payload=message.encode("utf-8"))

    @classmethod
    def close(cls, reason: str | None = None) -> Frame:
        payload = reason.encode("utf-8") if reason else b""
        return cls(kind=MessageKind.CLOSE, payload=payload)

    def audio_samples(self) -> np.ndarray:
        return np.frombuffer(self.payload, dtype=np.float32)

    def text_content(self) -> str:
        return self.payload.decode("utf-8")
