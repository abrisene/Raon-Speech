# Gradio full-duplex demo for Raon-Speech MLX.
# Adapted from demo/gradio_duplex_demo.py with MLX backend.

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import gradio as gr
import uvicorn

from raon_mlx.realtime.api.app import create_fastapi_app, mount_gradio_app

DEFAULT_MODEL_PATH = "KRAFTON/Raon-SpeechChat-9B"
DEFAULT_WS_PATH = "/realtime/ws"
DEFAULT_RESULT_ROOT = "output/mlx_duplex_demo"

START_ENDPOINT_CANDIDATES = ["/realtime/session/start"]
FINISH_ENDPOINT_CANDIDATES = ["/realtime/session/finish"]
START_SESSION_TIMEOUT_SEC = 180.0
FINISH_SESSION_TIMEOUT_SEC = 20.0


def _normalize_api_base(host: str, port: int, api_base: str | None) -> str:
    if api_base and api_base.strip():
        return api_base.rstrip("/")
    local_host = "127.0.0.1" if host == "0.0.0.0" else host
    return f"http://{local_host}:{port}"


def _post_json(url: str, payload: dict[str, Any], timeout: float = 20.0) -> tuple[int, dict[str, Any]]:
    raw = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=raw, method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = int(getattr(resp, "status", 200))
        body = resp.read().decode("utf-8", errors="replace")
        if not body.strip():
            return status, {}
        try:
            return status, json.loads(body)
        except json.JSONDecodeError:
            return status, {"raw": body}


def _call_first(base: str, endpoints: list[str], payload: dict[str, Any], *, timeout: float = 20.0) -> dict[str, Any]:
    errors: list[str] = []
    for endpoint in endpoints:
        url = f"{base}{endpoint}"
        try:
            status, body = _post_json(url, payload, timeout=timeout)
            if 200 <= status < 300:
                return body
            errors.append(f"{endpoint}: HTTP {status} ({body})")
        except urllib.error.HTTPError as exc:
            content = exc.read().decode("utf-8", errors="replace")
            errors.append(f"{endpoint}: HTTPError {exc.code} ({content})")
        except Exception as exc:
            errors.append(f"{endpoint}: {exc}")
    raise RuntimeError("; ".join(errors))


def _to_file_path(value: Any) -> str | None:
    if not value:
        return None
    p = Path(str(value))
    return str(p) if p.exists() else None


def finish_session(
    api_base: str, session_id: str,
) -> tuple[str, str, str | None, str | None, str | None, str | None]:
    session_id = session_id.strip()
    if not session_id:
        return "No active session.", "", None, None, None, None

    try:
        data = _call_first(
            api_base, FINISH_ENDPOINT_CANDIDATES,
            {"session_id": session_id}, timeout=FINISH_SESSION_TIMEOUT_SEC,
        )
    except Exception as exc:
        return f"Finish failed: {exc}", session_id, None, None, None, None

    close_reason = str(data.get("close_reason", "finished"))
    user_wav = _to_file_path(data.get("user_wav"))
    assistant_wav = _to_file_path(data.get("assistant_wav"))
    transcript_txt = _to_file_path(data.get("transcript_txt"))
    metadata_json = _to_file_path(data.get("metadata_json"))

    status = f"Session finished: {close_reason}"
    return status, "", user_wav, assistant_wav, transcript_txt, metadata_json


def start_session(
    api_base: str,
    model_path: str,
    result_root: str,
    speaker_audio: str,
    system_prompt: str,
    speak_first: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    eos_penalty: float,
    sil_penalty: float,
    bc_penalty: float,
    mic_gain: float,
    noise_gate: float,
) -> tuple[str, str, str]:
    payload = {
        "model_path": model_path,
        "result_root": result_root,
        "session": {
            "prompt": system_prompt,
            "speak_first": bool(speak_first),
            "sampling": {
                "temperature": float(temperature),
                "top_k": int(top_k),
                "top_p": float(top_p),
                "eos_penalty": float(eos_penalty),
                "sil_penalty": float(sil_penalty),
                "bc_penalty": float(bc_penalty),
            },
            "audio": {
                "input_gain": float(mic_gain),
                "silence_rms_threshold": float(noise_gate),
            },
        },
        "force_restart": True,
    }
    speaker_audio_str = speaker_audio.strip() if speaker_audio else ""
    if speaker_audio_str:
        payload["session"]["speaker_audio"] = speaker_audio_str

    try:
        data = _call_first(
            api_base, START_ENDPOINT_CANDIDATES, payload,
            timeout=START_SESSION_TIMEOUT_SEC,
        )
    except Exception as exc:
        return f"Start failed: {exc}", "", ""

    session_id = str(data.get("session_id") or "").strip()
    if not session_id:
        return "Backend did not return session_id.", "", ""
    return f"Session reserved: {session_id}", "", session_id


# Load JS from the original demo (reuse existing files)
_JS_DIR = Path(__file__).parent / "realtime" / "web"
START_STREAM_JS = (_JS_DIR / "gradio_stream.js").read_text(encoding="utf-8") if (_JS_DIR / "gradio_stream.js").exists() else ""
STOP_JS = (_JS_DIR / "gradio_stop.js").read_text(encoding="utf-8") if (_JS_DIR / "gradio_stop.js").exists() else ""


def download_all(
    user_wav: str | None, assistant_wav: str | None,
    transcript_txt: str | None, metadata_json: str | None,
) -> str | None:
    paths = [p for p in [user_wav, assistant_wav, transcript_txt, metadata_json] if p and Path(p).exists()]
    if not paths:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False, prefix="raon_mlx_all_")
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, Path(p).name)
    return tmp.name


def build_interface(
    default_api_base: str,
    ws_path: str,
    model_path: str,
    result_root: str,
    speaker_audio: str,
) -> gr.Blocks:
    with gr.Blocks(title="RAON MLX Full-Duplex Demo") as demo:
        gr.Markdown(
            "# RAON MLX Full-Duplex Demo\n"
            f"**Backend:** MLX (Apple Silicon) | **Model:** `{model_path}`\n\n"
            "- Browser mic uses echo cancel / noise suppression / auto gain.\n"
            "- `Start` begins realtime duplex streaming.\n"
            "- `Finish` closes the session and fetches downloadable artifacts."
        )

        with gr.Row():
            api_base = gr.Textbox(label="Realtime API Base", value=default_api_base)
            ws_path_box = gr.Textbox(label="WebSocket Path", value=ws_path)
            model_path_box = gr.Textbox(label="Model Path", value=model_path)

        result_root_box = gr.Textbox(label="Result Root", value=result_root)
        speaker_audio_box = gr.Textbox(label="Speaker Ref Audio", value=speaker_audio)

        with gr.Row():
            mode_radio = gr.Radio(
                label="Mode", choices=["listen-first", "speak-first"],
                value="listen-first",
            )
            system_prompt = gr.Textbox(
                label="System Prompt",
                value="You are engaging in real-time conversation.",
                visible=True,
            )
            speak_first = gr.Checkbox(label="Speak First", value=False, visible=False)

            def _sync_mode(mode: str) -> tuple[bool]:
                return (mode == "speak-first",)

            mode_radio.change(fn=_sync_mode, inputs=[mode_radio], outputs=[speak_first])

        with gr.Row():
            temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.2, step=0.01, value=0.9)
            top_k = gr.Slider(label="Top-k", minimum=1, maximum=200, step=1, value=66)
            top_p = gr.Slider(label="Top-p", minimum=0.0, maximum=1.0, step=0.01, value=0.99)

        with gr.Row():
            eos_penalty = gr.Slider(label="EOS Penalty", minimum=0.0, maximum=2.0, step=0.01, value=0.0)
            sil_penalty = gr.Slider(label="Sil Penalty", minimum=0.0, maximum=2.0, step=0.01, value=0.0)
            bc_penalty = gr.Slider(label="BC Penalty", minimum=-2.0, maximum=2.0, step=0.01, value=0.0)
            mic_gain = gr.Slider(label="Mic Gain", minimum=0.0, maximum=2.0, step=0.05, value=1.0)
            noise_gate = gr.Slider(label="Noise Gate (RMS)", minimum=0.0, maximum=0.1, step=0.001, value=0.03)

        with gr.Row():
            start_btn = gr.Button("Start", variant="primary", size="lg")
            finish_btn = gr.Button("Finish", variant="stop", size="lg")

        status = gr.Textbox(label="Status", value="Idle", interactive=False, elem_id="fd-status")
        transcript = gr.Textbox(
            label="Live Transcript", value="", lines=14,
            interactive=False, autoscroll=True, elem_id="fd-transcript",
        )
        session_id = gr.Textbox(label="Session ID", value="", interactive=False)

        gr.Markdown("### Downloads")
        with gr.Row():
            user_wav_file = gr.File(label="User Audio (WAV)", interactive=False)
            assistant_wav_file = gr.File(label="Assistant Audio (WAV)", interactive=False)
        with gr.Row():
            transcript_file = gr.File(label="Transcript (TXT)", interactive=False)
            metadata_file = gr.File(label="Metadata (JSON)", interactive=False)
        with gr.Row():
            download_all_btn = gr.Button("Download All", variant="secondary", size="sm")
            download_all_file = gr.File(label="All Files (ZIP)", interactive=False)

        start_event = start_btn.click(
            fn=start_session,
            inputs=[
                api_base, model_path_box, result_root_box, speaker_audio_box,
                system_prompt, speak_first,
                temperature, top_k, top_p,
                eos_penalty, sil_penalty, bc_penalty,
                mic_gain, noise_gate,
            ],
            outputs=[status, transcript, session_id],
            queue=False,
        )
        if START_STREAM_JS:
            start_event.then(
                fn=None, js=START_STREAM_JS,
                inputs=[ws_path_box, session_id, mic_gain, noise_gate, transcript, status],
                outputs=[status, transcript, session_id],
            )

        finish_event = finish_btn.click(
            fn=finish_session,
            inputs=[api_base, session_id],
            outputs=[status, session_id, user_wav_file, assistant_wav_file, transcript_file, metadata_file],
            queue=False,
        )
        if STOP_JS:
            finish_event.then(fn=None, js=STOP_JS, inputs=[session_id], outputs=[session_id])

        download_all_btn.click(
            fn=download_all,
            inputs=[user_wav_file, assistant_wav_file, transcript_file, metadata_file],
            outputs=[download_all_file],
            queue=False,
        )

        demo.queue()
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAON MLX Full-Duplex Gradio demo")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--api-base", type=str, default="")
    parser.add_argument("--ws-path", type=str, default=DEFAULT_WS_PATH)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--result-root", type=str, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--speaker-audio", type=str, default="")
    parser.add_argument("--quantize", type=str, default="hybrid",
                        choices=["hybrid", "8bit", "4bit", "none"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_base = _normalize_api_base(args.host, args.port, args.api_base)
    demo = build_interface(
        default_api_base=api_base,
        ws_path=args.ws_path,
        model_path=args.model_path,
        result_root=args.result_root,
        speaker_audio=args.speaker_audio,
    )
    fastapi_app = create_fastapi_app(
        model_path=args.model_path,
        session_kwargs={
            "result_root": args.result_root,
            "quantize": args.quantize,
        },
        ws_path=args.ws_path,
    )
    app = mount_gradio_app(fastapi_app, demo, path="/")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
