"""High-level pipeline API for Raon-Speech MLX inference.

Drop-in replacement for raon.pipeline.RaonPipeline that runs on MLX.
Supports TTS, STT, and SpeechChat tasks.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import warnings
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import soundfile as sf

from .models.raon import RaonMLX
from .models.generate import tts_generate, stt_generate, voice_chat_generate

logger = logging.getLogger(__name__)


class RaonMLXPipeline:
    """High-level MLX inference API matching the RaonPipeline interface.

    Example::

        pipe = RaonMLXPipeline("models/Raon-Speech-9B")
        text = pipe.stt("audio.wav")
        waveform, sr = pipe.tts("Hello, world!")
    """

    def __init__(
        self,
        model_path: str,
        hf_model_path: str | None = None,
        quant: str = "hybrid",
    ) -> None:
        """Load model and processor.

        Args:
            model_path: Path to HF checkpoint directory.
            hf_model_path: Path to HF checkpoint for tokenizer/audio encoder.
                Defaults to model_path.
            quant: Quantization mode: "none", "4bit", "8bit", "hybrid".
        """
        self.model_path = model_path
        self.hf_model_path = hf_model_path or model_path
        self.sampling_rate = 24000

        # Suppress tokenizer warnings
        warnings.filterwarnings("ignore")
        logging.disable(logging.WARNING)
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        _old = os.dup(2)
        _null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_null, 2)

        from raon.utils.processor import RaonProcessor
        self.processor = RaonProcessor.from_pretrained(self.hf_model_path)

        import time as _t
        _t.sleep(0.05)
        os.dup2(_old, 2)
        os.close(_null)
        os.close(_old)
        logging.disable(logging.NOTSET)

        # Load MLX model
        logger.info("Loading MLX model from %s", model_path)
        self.model = RaonMLX()
        self.model.load_weights_from_raon(model_path)

        if quant == "4bit":
            nn.quantize(self.model.thinker, bits=4, group_size=64)
            nn.quantize(self.model.talker, bits=4, group_size=64)
            nn.quantize(self.model.code_predictor.model, bits=4, group_size=64)
        elif quant == "8bit":
            nn.quantize(self.model.thinker, bits=8, group_size=64)
            nn.quantize(self.model.talker, bits=8, group_size=64)
            nn.quantize(self.model.code_predictor.model, bits=8, group_size=64)
        elif quant == "hybrid":
            nn.quantize(self.model.thinker, bits=4, group_size=64)
            nn.quantize(self.model.talker, bits=8, group_size=64)
            nn.quantize(self.model.code_predictor.model, bits=8, group_size=64)

        logger.info("MLX pipeline ready (quant=%s)", quant)

    def _tokenize(self, messages, force_audio_output=False, max_audio_chunk_length=None):
        """Tokenize messages using the HF processor."""
        return self.processor(
            messages,
            add_generation_prompt=True,
            force_audio_output=force_audio_output,
            device="cpu",
            max_audio_chunk_length=max_audio_chunk_length,
        )

    def _get_speaker_embedding(self, speaker_audio: str | None) -> mx.array | None:
        if speaker_audio is None:
            return None
        from .utils.speaker import extract_speaker_embedding
        return extract_speaker_embedding(
            speaker_audio,
            projection_weight=self.model.speaker_projection.weight,
        )

    def _encode_audio(self, audio_input, audio_input_lengths) -> tuple[mx.array, mx.array]:
        from .utils.audio_encoder import encode_audio_tensor
        return encode_audio_tensor(
            audio_tensor=audio_input,
            audio_lengths=audio_input_lengths,
            model_path=self.hf_model_path,
            input_adaptor_proj_0=self.model.input_adaptor.proj_0.weight,
            input_adaptor_proj_2=self.model.input_adaptor.proj_2.weight,
            input_adaptor_post_norm_weight=self.model.input_adaptor.post_norm.weight,
        )

    # ------------------------------------------------------------------
    # Public API (matches RaonPipeline interface)
    # ------------------------------------------------------------------

    def stt(self, audio: str, prompt: str | None = None) -> str:
        """STT: audio file path -> transcribed text."""
        from raon.utils.processor import get_default_stt_prompt

        effective_prompt = prompt or get_default_stt_prompt()
        messages = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": effective_prompt},
            ],
        }]
        inputs = self._tokenize(messages, max_audio_chunk_length=192000)
        input_ids = mx.array(inputs["input_ids"].numpy())

        audio_embeds, audio_mask = self._encode_audio(
            inputs["audio_input"], inputs["audio_input_lengths"],
        )

        token_ids = stt_generate(
            self.model, input_ids, audio_embeds, audio_mask,
            max_new_tokens=512, temperature=0.2,
        )
        return self.processor.tokenizer.decode(token_ids, skip_special_tokens=True)

    def tts(self, text: str, speaker_audio: str | None = None, seed: int | None = None) -> tuple:
        """TTS: text -> (waveform_tensor, sampling_rate)."""
        from raon.utils.processor import get_default_tts_prompt
        from raon.utils.special_tokens import SPEAKER_EMBEDDING_PLACEHOLDER

        speaker_prefix = str(SPEAKER_EMBEDDING_PLACEHOLDER) if speaker_audio else ""
        prompt = get_default_tts_prompt()
        messages = [{"role": "user", "content": f"{speaker_prefix}{prompt}:\n{text}"}]

        inputs = self._tokenize(messages, force_audio_output=True)
        input_ids = mx.array(inputs["input_ids"].numpy())
        speaker_embedding = self._get_speaker_embedding(speaker_audio)

        if seed is not None:
            mx.random.seed(seed)

        pcm, sr = tts_generate(
            self.model, input_ids,
            max_new_tokens=512,
            audio_temperature=1.2,
            top_k=20,
            speaker_embedding=speaker_embedding,
        )
        if pcm.shape[1] > 0:
            _ = pcm[0, 0].item()  # materialize

        # Return as numpy array matching RaonPipeline interface
        audio_np = np.array(pcm[0]).astype(np.float32)
        return audio_np, sr

    def speech_chat(self, audio: str) -> str:
        """SpeechChat: audio file path -> text response."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
            ],
        }]
        inputs = self._tokenize(messages, max_audio_chunk_length=192000)
        input_ids = mx.array(inputs["input_ids"].numpy())

        audio_embeds, audio_mask = self._encode_audio(
            inputs["audio_input"], inputs["audio_input_lengths"],
        )

        token_ids = stt_generate(
            self.model, input_ids, audio_embeds, audio_mask,
            max_new_tokens=1024, temperature=0.7,
        )
        return self.processor.tokenizer.decode(token_ids, skip_special_tokens=True)

    def textqa(self, text: str, audio: str | None = None) -> str:
        """TextQA: text question (+ optional audio) -> text response."""
        if audio:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": text},
                ],
            }]
        else:
            messages = [{"role": "user", "content": text}]

        inputs = self._tokenize(messages, max_audio_chunk_length=192000)
        input_ids = mx.array(inputs["input_ids"].numpy())

        if audio and "audio_input" in inputs and inputs["audio_input"] is not None:
            audio_embeds, audio_mask = self._encode_audio(
                inputs["audio_input"], inputs["audio_input_lengths"],
            )
        else:
            audio_embeds = mx.zeros((1, 0, 4096))
            audio_mask = mx.zeros((1, 0), dtype=mx.bool_)

        token_ids = stt_generate(
            self.model, input_ids, audio_embeds, audio_mask,
            max_new_tokens=1024, temperature=0.7,
        )
        return self.processor.tokenizer.decode(token_ids, skip_special_tokens=True)

    def voice_chat(
        self,
        audio: str,
        speaker_audio: str | None = None,
        seed: int | None = None,
    ) -> tuple[str, np.ndarray | None, int]:
        """Voice chat: audio in → text + audio out.

        Chains STT → text response → TTS: the model listens, thinks, then speaks.

        Args:
            audio: Path to input audio file.
            speaker_audio: Optional speaker reference for voice conditioning.
            seed: Optional random seed for reproducible voice.

        Returns:
            Tuple of (text_response, audio_waveform_or_None, sample_rate).
        """
        # Step 1: Understand the audio (SpeechChat → text response)
        text_response = self.speech_chat(audio)

        if not text_response.strip():
            return "", None, self.sampling_rate

        # Step 2: Speak the response (TTS)
        audio_out, sr = self.tts(text_response, speaker_audio=speaker_audio, seed=seed)
        return text_response, audio_out, sr

    def duplex(
        self,
        audio_input: str,
        output_dir: str,
        *,
        system_prompt: str | None = None,
        speak_first: bool = False,
        temperature: float = 0.9,
        top_k: int = 66,
        top_p: float = 0.99,
        eos_penalty: float = 0.0,
        sil_penalty: float = 0.0,
        bc_penalty: float = 0.0,
        speaker_audio: str | None = None,
    ) -> dict:
        """Run full-duplex inference on an audio file.

        Processes audio frame-by-frame (80ms at 24kHz), simultaneously encoding
        user speech and generating assistant speech. Saves output audio, transcript,
        and summary to output_dir.

        Args:
            audio_input: Path to input audio WAV file.
            output_dir: Directory to save output files.
            system_prompt: System prompt text. Defaults to duplex conversation prompt.
            speak_first: If True, model speaks first.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Top-p sampling.
            eos_penalty: Penalty on PAD token to encourage longer speech.
            sil_penalty: Penalty on SIL token to reduce silence.
            bc_penalty: Penalty on BC token.
            speaker_audio: Optional speaker reference audio path.

        Returns:
            Summary dict with durations, RTF, and transcript.
        """
        from .models.duplex_generate import run_duplex_offline

        if system_prompt is None:
            system_prompt = "You are engaging in real-time conversation."

        speaker_embeds = None
        if speaker_audio is not None:
            from .utils.speaker import compute_speaker_embedding
            speaker_embeds = compute_speaker_embedding(speaker_audio)

        return run_duplex_offline(
            model=self.model,
            tokenizer=self.processor.tokenizer,
            audio_path=audio_input,
            output_dir=output_dir,
            hf_model_path=self.hf_model_path,
            system_prompt=system_prompt,
            speak_first=speak_first,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_penalty=eos_penalty,
            sil_penalty=sil_penalty,
            bc_penalty=bc_penalty,
            speaker_embeds=speaker_embeds,
        )

    @staticmethod
    def save_audio(audio_data: tuple, path: str) -> None:
        """Save (waveform, sr) tuple to a WAV file."""
        audio, sr = audio_data
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        audio = np.asarray(audio, dtype=np.float32)
        sf.write(path, audio, sr)
