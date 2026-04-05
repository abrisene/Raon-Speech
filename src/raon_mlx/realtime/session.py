# MLX realtime duplex session orchestration.
# Adapted from demo/realtime/runtime/session.py with MLX backend.

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .protocol.messages import Frame

logger = logging.getLogger(__name__)

SAMPLES_PER_FRAME = 1920
SAMPLE_RATE = 24000
FRAME_BYTES = SAMPLES_PER_FRAME * 4  # float32


@dataclass
class AudioConfig:
    sampling_rate: int = SAMPLE_RATE
    frame_size: int = SAMPLES_PER_FRAME
    frame_bytes: int = FRAME_BYTES
    bytes_per_second: int = SAMPLE_RATE * 4
    mic_gain: float = 1.0
    noise_gate: float = 0.0
    output_gain: float = 1.0
    output_clip: float = 1.0
    input_clip: float = 1.0
    max_buffer_bytes: int = FRAME_BYTES * 50  # ~4 seconds
    soft_backlog_seconds: float = 2.0
    hard_backlog_seconds: float = 5.0
    hard_backlog_action: str = "degrade"
    degrade_target_seconds: float = 0.5


@dataclass
class SamplingConfig:
    temperature: float = 0.9
    top_k: int = 66
    top_p: float = 0.99
    eos_penalty: float = 0.0
    sil_penalty: float = 0.0
    bc_penalty: float = 0.0


@dataclass
class SessionConfig:
    session_id: str = ""
    prompt: str = "You are engaging in real-time conversation."
    speak_first: bool = False
    speaker_audio: str | None = None
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)


@dataclass
class SessionMetrics:
    frames_in: int = 0
    frames_out: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    dropped_input_frames: int = 0
    dropped_input_bytes: int = 0
    backlog_soft_events: int = 0
    backlog_hard_events: int = 0
    max_time_behind_seconds: float = 0.0
    decode_errors: int = 0
    consecutive_decode_errors: int = 0
    decode_step_total_seconds: float = 0.0
    decode_step_max_seconds: float = 0.0


class MLXRealtimeDuplexSession:
    """High-level session wrapper for MLX duplex realtime decoding."""

    def __init__(
        self,
        *,
        session_id: str,
        model_path: str,
        hf_model_path: str | None = None,
        result_root: str = "./output/mlx_duplex_demo",
        session: dict[str, Any] | None = None,
        runtime: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        session_payload = dict(session or {})
        sampling_payload = dict(session_payload.get("sampling") or {})
        audio_payload = dict(session_payload.get("audio") or {})

        # Resolve HF model path for tokenizer + encoder
        # If model_path is an MLX converted dir, read source_model from config
        if hf_model_path is None:
            import json as _json
            cfg_path = Path(model_path) / "config.json"
            if cfg_path.exists():
                with open(cfg_path) as _f:
                    _cfg = _json.load(_f)
                hf_model_path = _cfg.get("source_model") or model_path
            else:
                hf_model_path = model_path
        self._hf_model_path = hf_model_path

        self._config = SessionConfig(
            session_id=session_id,
            prompt=str(session_payload.get("prompt", "You are engaging in real-time conversation.")),
            speak_first=bool(session_payload.get("speak_first", False)),
            speaker_audio=session_payload.get("speaker_audio"),
        )
        self._config.sampling.temperature = float(sampling_payload.get("temperature", 0.9))
        self._config.sampling.top_k = int(sampling_payload.get("top_k", 66))
        self._config.sampling.top_p = float(sampling_payload.get("top_p", 0.99))
        self._config.sampling.eos_penalty = float(sampling_payload.get("eos_penalty", 0.0))
        self._config.sampling.sil_penalty = float(sampling_payload.get("sil_penalty", 0.0))
        self._config.sampling.bc_penalty = float(sampling_payload.get("bc_penalty", 0.0))
        self._config.audio.mic_gain = float(
            audio_payload.get("mic_gain", audio_payload.get("input_gain", 1.0))
        )
        self._config.audio.noise_gate = float(
            audio_payload.get("noise_gate", audio_payload.get("silence_rms_threshold", 0.0))
        )

        # Load MLX model
        model, tokenizer = get_mlx_runtime(
            model_path=model_path,
            hf_model_path=self._hf_model_path,
            quantize=str((runtime or {}).get("quantize", "hybrid")),
        )
        self._model = model
        self._tokenizer = tokenizer
        self._model_path = model_path

        # Compute speaker embeddings if provided
        speaker_embeds = None
        speaker_audio_value = str(self._config.speaker_audio or "").strip()
        if speaker_audio_value:
            try:
                speaker_embeds = self._load_speaker_embeds(speaker_audio_value)
                logger.info("Speaker reference loaded: %s", speaker_audio_value)
            except Exception as exc:
                logger.warning("Speaker reference failed: %s (%s)", speaker_audio_value, exc)

        # Initialize duplex state
        from ..models.duplex_generate import init_duplex_state
        self._duplex_state = init_duplex_state(
            model=model,
            tokenizer=tokenizer,
            hf_model_path=self._hf_model_path,
            system_prompt=self._config.prompt,
            speak_first=self._config.speak_first,
            temperature=self._config.sampling.temperature,
            top_k=self._config.sampling.top_k,
            top_p=self._config.sampling.top_p,
            eos_penalty=self._config.sampling.eos_penalty,
            sil_penalty=self._config.sampling.sil_penalty,
            bc_penalty=self._config.sampling.bc_penalty,
            speaker_embeds=speaker_embeds,
        )

        # Session state
        self._raw_input_bytes = bytearray()
        self._metrics = SessionMetrics()
        self._result_root = Path(result_root)
        self._user_audio_chunks: list[np.ndarray] = []
        self._assistant_audio_chunks: list[np.ndarray] = []
        self._transcript_parts: list[str] = []
        self._closed = False
        self._close_reason: str | None = None
        self._result_payload: dict[str, Any] | None = None

    def _load_speaker_embeds(self, audio_path: str):
        import mlx.core as mx
        from ..utils.speaker import compute_speaker_embedding
        return compute_speaker_embedding(audio_path)

    @property
    def session_id(self) -> str:
        return self._config.session_id

    def start(self) -> list[Frame]:
        return [Frame.ready()]

    def handle_audio_frame(self, pcm: np.ndarray) -> list[Frame]:
        if self._closed:
            return [Frame.close(self._close_reason or "finished")]

        import mlx.core as mx
        from ..models.duplex_generate import duplex_step

        pcm_np = np.asarray(pcm, dtype=np.float32).reshape(-1)
        self._user_audio_chunks.append(pcm_np.copy())

        # Buffer management
        self._raw_input_bytes.extend(pcm_np.tobytes())
        self._metrics.bytes_in += len(pcm_np) * 4

        # Backlog check
        audio_cfg = self._config.audio
        max_bytes = audio_cfg.max_buffer_bytes
        if max_bytes > 0 and len(self._raw_input_bytes) > max_bytes:
            drop = len(self._raw_input_bytes) - max_bytes
            drop -= drop % FRAME_BYTES
            if drop > 0:
                del self._raw_input_bytes[:drop]
                self._metrics.dropped_input_bytes += drop
                self._metrics.dropped_input_frames += drop // FRAME_BYTES

        backlog_seconds = len(self._raw_input_bytes) / max(1, audio_cfg.bytes_per_second)
        if audio_cfg.soft_backlog_seconds > 0 and backlog_seconds > audio_cfg.soft_backlog_seconds:
            self._metrics.backlog_soft_events += 1
        if audio_cfg.hard_backlog_seconds > 0 and backlog_seconds > audio_cfg.hard_backlog_seconds:
            self._metrics.backlog_hard_events += 1
            if audio_cfg.hard_backlog_action == "close":
                self._close_reason = "overloaded_backlog"
                return [Frame.close("overloaded_backlog")]
            # Degrade: drop to target
            target_bytes = int(audio_cfg.degrade_target_seconds * audio_cfg.bytes_per_second)
            if len(self._raw_input_bytes) > target_bytes:
                drop = len(self._raw_input_bytes) - target_bytes
                drop -= drop % FRAME_BYTES
                if drop > 0:
                    del self._raw_input_bytes[:drop]
                    self._metrics.dropped_input_bytes += drop
                    self._metrics.dropped_input_frames += drop // FRAME_BYTES

        if backlog_seconds > self._metrics.max_time_behind_seconds:
            self._metrics.max_time_behind_seconds = backlog_seconds

        # Process available frames
        frames: list[Frame] = []
        while len(self._raw_input_bytes) >= FRAME_BYTES:
            chunk_bytes = bytes(self._raw_input_bytes[:FRAME_BYTES])
            del self._raw_input_bytes[:FRAME_BYTES]
            pcm_frame = np.frombuffer(chunk_bytes, dtype=np.float32).copy()

            # Apply mic gain and noise gate
            if audio_cfg.mic_gain != 1.0:
                pcm_frame = pcm_frame * audio_cfg.mic_gain
            if audio_cfg.input_clip > 0:
                pcm_frame = np.clip(pcm_frame, -audio_cfg.input_clip, audio_cfg.input_clip)
            if audio_cfg.noise_gate > 0:
                rms = float(np.sqrt(np.mean(pcm_frame * pcm_frame))) if pcm_frame.size else 0.0
                if rms < audio_cfg.noise_gate:
                    pcm_frame = np.zeros_like(pcm_frame)

            self._metrics.frames_in += 1

            # Run duplex step
            audio_input = mx.array(pcm_frame[None, None, :])  # [1, 1, 1920]
            decode_started = time.perf_counter()
            try:
                self._duplex_state, output_audio, text_ids = duplex_step(
                    self._model, self._duplex_state, audio_input,
                )
                self._metrics.consecutive_decode_errors = 0
            except Exception as exc:
                self._metrics.decode_errors += 1
                self._metrics.consecutive_decode_errors += 1
                logger.exception("duplex_step error session=%s", self.session_id)
                if self._metrics.consecutive_decode_errors >= 3:
                    self._close_reason = "internal_error"
                    frames.append(Frame.close("internal_error"))
                    break
                frames.append(Frame.error(f"decode error: {exc}"))
                # Output silence on error
                frames.append(Frame.audio(np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)))
                continue
            finally:
                elapsed = time.perf_counter() - decode_started
                self._metrics.decode_step_total_seconds += elapsed
                if elapsed > self._metrics.decode_step_max_seconds:
                    self._metrics.decode_step_max_seconds = elapsed

            # Emit audio
            out_np = np.array(output_audio[0, 0], copy=False).astype(np.float32)
            if audio_cfg.output_gain != 1.0:
                out_np = out_np * audio_cfg.output_gain
            if audio_cfg.output_clip > 0:
                out_np = np.clip(out_np, -audio_cfg.output_clip, audio_cfg.output_clip)
            self._assistant_audio_chunks.append(out_np.copy())
            frames.append(Frame.audio(out_np))
            self._metrics.frames_out += 1
            self._metrics.bytes_out += len(out_np) * 4

            # Emit text delta
            if text_ids:
                text_delta = self._tokenizer.decode(text_ids, skip_special_tokens=False)
                if text_delta:
                    self._transcript_parts.append(text_delta)
                    frames.append(Frame.text(text_delta))

        return frames

    def request_close(self, reason: str) -> list[Frame]:
        self.finish(reason)
        return [Frame.close(reason)]

    def finish(self, reason: str = "client_finish") -> dict[str, Any]:
        if self._result_payload is not None:
            return self._result_payload

        self._close_reason = reason
        self._closed = True

        # Save artifacts
        output_dir = self._result_root / self._config.session_id
        output_dir.mkdir(parents=True, exist_ok=True)

        files: dict[str, str] = {}
        import soundfile as sf

        if self._user_audio_chunks:
            user_audio = np.concatenate(self._user_audio_chunks)
            user_path = str(output_dir / "user.wav")
            sf.write(user_path, user_audio, SAMPLE_RATE)
            files["user_wav"] = user_path

        if self._assistant_audio_chunks:
            asst_audio = np.concatenate(self._assistant_audio_chunks)
            asst_path = str(output_dir / "assistant.wav")
            sf.write(asst_path, asst_audio, SAMPLE_RATE)
            files["assistant_wav"] = asst_path

        transcript = "".join(self._transcript_parts)
        if transcript:
            tx_path = str(output_dir / "transcript.txt")
            Path(tx_path).write_text(transcript, encoding="utf-8")
            files["transcript"] = tx_path

        m = self._metrics
        frame_seconds = SAMPLES_PER_FRAME / SAMPLE_RATE
        user_seconds = m.frames_in * frame_seconds
        runtime_stats = {
            "frames_in": m.frames_in,
            "frames_out": m.frames_out,
            "bytes_in": m.bytes_in,
            "bytes_out": m.bytes_out,
            "dropped_input_frames": m.dropped_input_frames,
            "dropped_input_bytes": m.dropped_input_bytes,
            "backlog_soft_events": m.backlog_soft_events,
            "backlog_hard_events": m.backlog_hard_events,
            "max_time_behind_seconds": m.max_time_behind_seconds,
            "decode_errors": m.decode_errors,
            "decode_step_total_seconds": m.decode_step_total_seconds,
            "decode_step_avg_ms": (m.decode_step_total_seconds / m.frames_in * 1000) if m.frames_in else 0,
            "decode_step_max_ms": m.decode_step_max_seconds * 1000,
            "user_audio_seconds": user_seconds,
            "decode_rtf": (m.decode_step_total_seconds / user_seconds) if user_seconds > 0 else 0,
        }

        metadata = {
            "session_id": self._config.session_id,
            "model_path": self._model_path,
            "close_reason": reason,
            "runtime_stats": runtime_stats,
            "files": files,
        }
        meta_path = str(output_dir / "metadata.json")
        Path(meta_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        files["metadata"] = meta_path

        self._result_payload = {
            "session_id": self._config.session_id,
            "close_reason": reason,
            "metadata": metadata,
            "user_wav": files.get("user_wav"),
            "assistant_wav": files.get("assistant_wav"),
            "transcript_txt": files.get("transcript"),
            "metadata_json": meta_path,
        }
        return self._result_payload

    def artifact_response(self) -> dict[str, Any] | None:
        return self._result_payload

    def close(self) -> None:
        if not self._closed:
            self.finish(self._close_reason or "client_finish")


# ---- MLX Runtime Singleton ----

_RUNTIME_LOCK = threading.Lock()
_RUNTIME_CACHE: dict[str, tuple] = {}


def get_mlx_runtime(
    *,
    model_path: str,
    hf_model_path: str | None = None,
    quantize: str = "hybrid",
) -> tuple:
    """Return (model, tokenizer) singleton for the given model path."""
    import mlx.core as mx
    import mlx.nn as nn

    cache_key = f"{model_path}:{quantize}"
    with _RUNTIME_LOCK:
        if cache_key in _RUNTIME_CACHE:
            return _RUNTIME_CACHE[cache_key]

    from ..models.raon import RaonMLX

    logger.info("Loading MLX model from %s (quantize=%s)...", model_path, quantize)
    model = RaonMLX()

    model_dir = Path(model_path)
    if (model_dir / "model.safetensors").exists() and (model_dir / "config.json").exists():
        model.load_mlx_weights(model_path)
    else:
        model.load_weights_from_raon(model_path)

    # Apply quantization if not already applied by load_mlx_weights
    if quantize == "hybrid":
        try:
            nn.quantize(model.thinker, bits=4, group_size=64)
        except Exception:
            pass
        try:
            nn.quantize(model.talker, bits=8, group_size=64)
        except Exception:
            pass
        try:
            nn.quantize(model.code_predictor.model, bits=8, group_size=64)
        except Exception:
            pass
    elif quantize == "8bit":
        nn.quantize(model.thinker, bits=8, group_size=64)
        nn.quantize(model.talker, bits=8, group_size=64)
        nn.quantize(model.code_predictor.model, bits=8, group_size=64)
    elif quantize == "4bit":
        nn.quantize(model.thinker, bits=4, group_size=64)

    # Load tokenizer — try HF model path first (has tokenizer files),
    # fall back to model_path if not specified
    from transformers import AutoTokenizer
    tokenizer_path = hf_model_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)

    # Warmup Mimi
    model.mimi.reset_all()

    logger.info("MLX model loaded.")

    runtime = (model, tokenizer)
    with _RUNTIME_LOCK:
        _RUNTIME_CACHE[cache_key] = runtime
    return runtime


def create_session(**kwargs: Any) -> MLXRealtimeDuplexSession:
    """Factory used by the runtime manager to build the active session."""
    return MLXRealtimeDuplexSession(**kwargs)
