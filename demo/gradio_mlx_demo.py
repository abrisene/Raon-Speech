"""Gradio demo for Raon-Speech running on MLX (Apple Silicon).

Usage:
    python demo/gradio_mlx_demo.py
    python demo/gradio_mlx_demo.py --model models/Raon-Speech-9B --quant hybrid
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from raon_mlx.pipeline import RaonMLXPipeline

_temp_files: list[str] = []


def audio_to_tempfile(audio_data: tuple[int, np.ndarray] | None) -> str | None:
    if audio_data is None:
        return None
    sr, audio_np = audio_data
    if audio_np.ndim > 1:
        audio_np = audio_np[:, 0]
    for old in _temp_files:
        try:
            Path(old).unlink(missing_ok=True)
        except OSError:
            pass
    _temp_files.clear()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_np = audio_np.astype(np.float32)
    if audio_np.max() > 1.0 or audio_np.min() < -1.0:
        audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()))
    sf.write(tmp.name, audio_np, sr)
    _temp_files.append(tmp.name)
    return tmp.name


def run_inference(
    pipe: RaonMLXPipeline,
    task: str,
    text: str,
    audio: tuple[int, np.ndarray] | None,
    ref_audio: tuple[int, np.ndarray] | None,
    voice_seed: int = -1,
):
    try:
        if task == "STT":
            if audio is None:
                return "STT requires audio input.", None
            path = audio_to_tempfile(audio)
            transcript = pipe.stt(path)
            return transcript, None

        elif task == "TTS":
            if not text.strip():
                return "TTS requires text input.", None
            speaker_path = audio_to_tempfile(ref_audio)
            seed = int(voice_seed) if voice_seed >= 0 else None
            audio_out, sr = pipe.tts(text, speaker_audio=speaker_path, seed=seed)
            return "", (sr, audio_out)

        elif task == "SpeechChat":
            if audio is None:
                return "SpeechChat requires audio input.", None
            path = audio_to_tempfile(audio)
            answer = pipe.speech_chat(path)
            return answer, None

        elif task == "TextQA":
            if not text.strip():
                return "TextQA requires text input.", None
            audio_path = audio_to_tempfile(audio) if audio is not None else None
            response = pipe.textqa(text, audio=audio_path)
            return response, None

        else:
            return f"Unknown task: {task}", None

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return f"Error: {exc}", None


def build_interface(pipe: RaonMLXPipeline) -> gr.Blocks:
    with gr.Blocks(title="Raon-Speech MLX", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Raon-Speech MLX\n"
            "**9B speech model running on Apple Silicon via MLX** "
            "(hybrid quantized, 2.6x real-time TTS)"
        )

        with gr.Row():
            task = gr.Dropdown(
                choices=["TTS", "STT", "SpeechChat", "TextQA"],
                value="TTS",
                label="Task",
            )

        with gr.Row():
            with gr.Column():
                text_in = gr.Textbox(label="Text Input", lines=3, placeholder="Enter text for TTS or TextQA...")
                audio_in = gr.Audio(label="Audio Input", type="numpy", visible=False)
                ref_audio = gr.Audio(label="Speaker Reference (optional)", type="numpy", visible=True)
                voice_seed = gr.Number(label="Voice Seed (-1 = random)", value=-1, precision=0, visible=True)
                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                text_out = gr.Textbox(label="Text Output", lines=5, visible=False)
                audio_out = gr.Audio(label="Audio Output", type="numpy")

        def on_task_change(t):
            show_text_in = t in ("TTS", "TextQA")
            show_audio_in = t in ("STT", "TextQA", "SpeechChat")
            show_ref = t == "TTS"
            show_seed = t == "TTS"
            show_text_out = t != "TTS"
            show_audio_out = t == "TTS"
            return (
                gr.update(visible=show_text_in),
                gr.update(visible=show_audio_in),
                gr.update(visible=show_ref),
                gr.update(visible=show_seed),
                gr.update(visible=show_text_out),
                gr.update(visible=show_audio_out),
            )

        task.change(
            on_task_change, [task],
            [text_in, audio_in, ref_audio, voice_seed, text_out, audio_out],
        )

        generate_btn.click(
            lambda t, txt, aud, ref, seed: run_inference(pipe, t, txt, aud, ref, seed),
            inputs=[task, text_in, audio_in, ref_audio, voice_seed],
            outputs=[text_out, audio_out],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Raon-Speech MLX Gradio Demo")
    parser.add_argument("--model", default="models/Raon-Speech-9B", help="HF model path")
    parser.add_argument("--quant", default="hybrid", choices=["none", "4bit", "8bit", "hybrid"])
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print(f"Loading Raon-Speech MLX pipeline (quant={args.quant})...")
    pipe = RaonMLXPipeline(args.model, quant=args.quant)
    print("Ready!")

    demo = build_interface(pipe)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
