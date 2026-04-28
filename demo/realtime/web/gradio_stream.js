async (
  wsPath,
  sessionId,
  micGain,
  noiseGate,
  oldTranscript,
  oldStatus,
) => {
  console.log("[raon-duplex] JS handler called. sessionId:", JSON.stringify(sessionId), "wsPath:", wsPath);
  const state = window.__raon_fd_demo || {};
  if (state.ws && state.ws.readyState <= 1) {
    return ["Session already active.", oldTranscript || "", sessionId || ""];
  }

  // Fallback: read session_id from DOM if the Gradio parameter is empty
  // (can happen if Gradio's internal WebSocket reconnected during slow session creation)
  let effectiveSessionId = sessionId;
  if (!effectiveSessionId) {
    const sessionElem = document.querySelector('#component-session_id textarea, #component-session_id input');
    if (sessionElem) effectiveSessionId = sessionElem.value;
  }
  if (!effectiveSessionId) {
    // Also try the general session id element
    const el = document.getElementById('fd-session-id');
    if (el) effectiveSessionId = (el.value || el.textContent || '').trim();
  }

  if (!effectiveSessionId) {
    return [oldStatus || "Missing session id.", oldTranscript || "", sessionId || ""];
  }
  sessionId = effectiveSessionId;

  const SERVER_SAMPLE_RATE = 24000;
  const FRAME_SAMPLES = 1920;
  const CAPTURE_BUFFER = 1024;
  const PLAYBACK_TARGET_LEAD_SEC = 0.03;
  const PLAYBACK_FADE_SEC = 0.005;
  const PLAYBACK_LATE_THRESHOLD_SEC = 0.005;

  const encodeAudioFrame = (samples) => {
    const out = new ArrayBuffer(1 + samples.length * 4);
    const view = new DataView(out);
    view.setUint8(0, 0x01);
    for (let i = 0; i < samples.length; i += 1) {
      view.setFloat32(1 + i * 4, samples[i], true);
    }
    return out;
  };

  const resampleLinear = (input, inputSampleRate, outputSampleRate) => {
    if (inputSampleRate === outputSampleRate) {
      return input;
    }
    const ratio = inputSampleRate / outputSampleRate;
    const outLength = Math.max(1, Math.round(input.length / ratio));
    const out = new Float32Array(outLength);
    for (let i = 0; i < outLength; i += 1) {
      const srcIndex = i * ratio;
      const lo = Math.floor(srcIndex);
      const hi = Math.min(lo + 1, input.length - 1);
      const frac = srcIndex - lo;
      out[i] = input[lo] * (1 - frac) + input[hi] * frac;
    }
    return out;
  };

  const teardownCapture = async (runtime) => {
    if (runtime.processor) {
      try { runtime.processor.disconnect(); } catch (_) {}
    }
    if (runtime.micSource) {
      try { runtime.micSource.disconnect(); } catch (_) {}
    }
    if (runtime.stream) {
      try { runtime.stream.getTracks().forEach((track) => track.stop()); } catch (_) {}
    }
    if (runtime.captureContext) {
      try { await runtime.captureContext.close(); } catch (_) {}
    }
    runtime.processor = null;
    runtime.micSource = null;
    runtime.stream = null;
    runtime.captureContext = null;
    runtime.pending = new Float32Array(0);
  };

  const teardownPlayback = async (runtime) => {
    const sources = runtime.sources || new Set();
    for (const src of sources) {
      try { src.stop(0); } catch (_) {}
      try { src.disconnect(); } catch (_) {}
    }
    sources.clear();
    runtime.sources = sources;
    if (runtime.audioContext) {
      try { await runtime.audioContext.close(); } catch (_) {}
    }
    runtime.audioContext = null;
    runtime.nextPlayTime = 0;
  };

  if (state && typeof state === "object") {
    await teardownCapture(state);
    await teardownPlayback(state);
    if (state.ws && state.ws.readyState <= 1) {
      try { state.ws.close(1000, "restart"); } catch (_) {}
    }
  }

  const startCapture = async (runtime) => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("getUserMedia is unavailable in this browser.");
    }

    runtime.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    runtime.captureContext = new (window.AudioContext || window.webkitAudioContext)();
    await runtime.captureContext.resume();
    const inputSampleRate = runtime.captureContext.sampleRate;
    runtime.micSource = runtime.captureContext.createMediaStreamSource(runtime.stream);
    runtime.processor = runtime.captureContext.createScriptProcessor(CAPTURE_BUFFER, 1, 1);
    runtime.pending = new Float32Array(0);

    runtime.processor.onaudioprocess = (event) => {
      try {
        event.outputBuffer.getChannelData(0).fill(0);
      } catch (_) {}
      if (!runtime.ws || runtime.ws.readyState !== WebSocket.OPEN) {
        return;
      }

      let chunk = new Float32Array(event.inputBuffer.getChannelData(0));
      if (inputSampleRate !== SERVER_SAMPLE_RATE) {
        chunk = resampleLinear(chunk, inputSampleRate, SERVER_SAMPLE_RATE);
      }

      const gain = Number(micGain);
      if (Number.isFinite(gain) && gain !== 1.0) {
        for (let i = 0; i < chunk.length; i += 1) {
          chunk[i] *= gain;
        }
      }

      const gate = Number(noiseGate);
      if (Number.isFinite(gate) && gate > 0) {
        let sum = 0;
        for (let i = 0; i < chunk.length; i += 1) {
          sum += chunk[i] * chunk[i];
        }
        const rms = Math.sqrt(sum / Math.max(1, chunk.length));
        if (rms < gate) {
          chunk = new Float32Array(chunk.length);
        }
      }

      const merged = new Float32Array(runtime.pending.length + chunk.length);
      merged.set(runtime.pending, 0);
      merged.set(chunk, runtime.pending.length);
      runtime.pending = merged;

      while (runtime.pending.length >= FRAME_SAMPLES) {
        const frame = runtime.pending.slice(0, FRAME_SAMPLES);
        runtime.pending = runtime.pending.slice(FRAME_SAMPLES);
        runtime.sentAudioFrames = (runtime.sentAudioFrames || 0) + 1;
        runtime.ws.send(encodeAudioFrame(frame));
      }
    };

    runtime.micSource.connect(runtime.processor);
    runtime.processor.connect(runtime.captureContext.destination);
  };

  const wsProto = window.location.protocol === "https:" ? "wss://" : "ws://";
  const wsHost = window.location.host;
  const socketPath = (wsPath || "/realtime/ws").startsWith("/")
    ? (wsPath || "/realtime/ws")
    : ("/" + wsPath);
  const wsUrl = `${wsProto}${wsHost}${socketPath}?session_id=${encodeURIComponent(sessionId)}`;
  const ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";

  const resolveTextNode = (id) => {
    const root = document.getElementById(id);
    if (!root) return null;
    if ("value" in root) return root;
    return root.querySelector("textarea, input, [contenteditable='true']") || root;
  };

  const transcriptElem = resolveTextNode("fd-transcript");
  const statusElem = resolveTextNode("fd-status");
  const setTranscript = (text) => {
    if (!transcriptElem) return;
    if ("value" in transcriptElem) {
      transcriptElem.value = text;
      transcriptElem.dispatchEvent(new Event("input", { bubbles: true }));
      transcriptElem.dispatchEvent(new Event("change", { bubbles: true }));
    } else {
      transcriptElem.textContent = text;
    }
  };
  const setStatus = (text) => {
    if (!statusElem) return;
    if ("value" in statusElem) {
      statusElem.value = text;
      statusElem.dispatchEvent(new Event("input", { bubbles: true }));
      statusElem.dispatchEvent(new Event("change", { bubbles: true }));
    } else {
      statusElem.textContent = text;
    }
  };

  let transcript = oldTranscript || "";
  let audioContext = null;
  const playPcm = (pcmFloat32) => {
    if (!pcmFloat32 || pcmFloat32.length === 0) return;
    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
      runtime.audioContext = audioContext;
      runtime.nextPlayTime = 0;
      try { audioContext.resume(); } catch (_) {}
    }
    const ctx = audioContext;
    const now = ctx.currentTime;
    const scheduledTail = (runtime.nextPlayTime || 0) > now + 0.001
      ? (runtime.nextPlayTime || 0)
      : now + PLAYBACK_TARGET_LEAD_SEC;
    const ahead = scheduledTail - now;
    runtime.maxPlaybackAheadSec = Math.max(runtime.maxPlaybackAheadSec || 0, ahead);
    runtime.minPlaybackAheadSec = Math.min(
      Number.isFinite(runtime.minPlaybackAheadSec) ? runtime.minPlaybackAheadSec : Infinity,
      ahead,
    );
    if (ahead < PLAYBACK_LATE_THRESHOLD_SEC) {
      runtime.latePlaybackChunks = (runtime.latePlaybackChunks || 0) + 1;
    }
    const buffer = ctx.createBuffer(1, pcmFloat32.length, 24000);
    buffer.getChannelData(0).set(pcmFloat32);
    const src = ctx.createBufferSource();
    const gainNode = ctx.createGain();
    src.buffer = buffer;
    src.connect(gainNode);
    gainNode.connect(ctx.destination);
    const startAt = scheduledTail;
    runtime.sources.add(src);
    src.onended = () => {
      runtime.sources.delete(src);
      try { src.disconnect(); } catch (_) {}
      try { gainNode.disconnect(); } catch (_) {}
    };
    const fade = Math.min(PLAYBACK_FADE_SEC, buffer.duration / 4);
    gainNode.gain.setValueAtTime(0.0, startAt);
    gainNode.gain.linearRampToValueAtTime(1.0, startAt + fade);
    gainNode.gain.setValueAtTime(1.0, Math.max(startAt + fade, startAt + buffer.duration - fade));
    gainNode.gain.linearRampToValueAtTime(0.0, startAt + buffer.duration);
    src.start(startAt);
    runtime.nextPlayTime = startAt + buffer.duration;
  };

  const runtime = {
    ws,
    audioContext,
    captureContext: null,
    stream: null,
    micSource: null,
    processor: null,
    sources: new Set(),
    pending: new Float32Array(0),
    nextPlayTime: 0,
    droppedPlaybackChunks: 0,
    latePlaybackChunks: 0,
    sentAudioFrames: 0,
    receivedAudioFrames: 0,
    maxPlaybackAheadSec: 0,
    minPlaybackAheadSec: Infinity,
    sessionId,
    transcript,
  };

  console.log("[raon-duplex] Opening WebSocket:", wsUrl);
  ws.onopen = () => {
    console.log("[raon-duplex] WebSocket connected, waiting for READY...");
    setStatus(`WebSocket connected: ${sessionId}`);
  };
  ws.onclose = async (ev) => {
    console.log("[raon-duplex] WebSocket closed:", ev.code, ev.reason);
    await teardownCapture(runtime);
    await teardownPlayback(runtime);
    const stats = `sent=${runtime.sentAudioFrames || 0} recv=${runtime.receivedAudioFrames || 0} late=${runtime.latePlaybackChunks || 0} maxAhead=${(runtime.maxPlaybackAheadSec || 0).toFixed(3)} minAhead=${Number.isFinite(runtime.minPlaybackAheadSec) ? runtime.minPlaybackAheadSec.toFixed(3) : 'n/a'}`;
    console.log("[raon-duplex] Playback stats:", stats);
    setStatus(`Streaming closed (code=${ev.code} reason=${ev.reason || "none"} ${stats}).`);
  };
  ws.onerror = (ev) => {
    console.error("[raon-duplex] WebSocket error:", ev);
    setStatus(`WebSocket error: ${String(ev?.type || "unknown")}`);
  };
  ws.onmessage = async (event) => {
    if (typeof event.data === "string") {
      try {
        const msg = JSON.parse(event.data);
        const kind = String(msg.kind || msg.type || "").toLowerCase();
        if (kind === "text" || kind === "transcript") {
          const delta = String(msg.text || msg.content || "");
          if (delta) {
            transcript += delta;
            setTranscript(transcript);
          }
        } else if (kind === "error") {
          setStatus(String(msg.message || "Server error"));
        }
      } catch (_) {
        transcript += String(event.data);
        setTranscript(transcript);
      }
      return;
    }

    if (event.data instanceof ArrayBuffer) {
      const bytes = new Uint8Array(event.data);
      if (bytes.length < 1) return;
      const kind = bytes[0];
      const payload = bytes.slice(1);
      if (kind === 0x00) {
        console.log("[raon-duplex] READY received, starting mic capture...");
        try {
          await startCapture(runtime);
          console.log("[raon-duplex] Mic capture started OK");
          setStatus(`Streaming started: ${sessionId}`);
        } catch (err) {
          const message = String(err && err.message ? err.message : err);
          console.error("[raon-duplex] Mic capture failed:", err);
          setStatus(`Microphone start failed: ${message}`);
          try {
            const closePayload = new TextEncoder().encode("capture-start-failed");
            const frame = new Uint8Array(1 + closePayload.length);
            frame[0] = 0x06;
            frame.set(closePayload, 1);
            ws.send(frame);
          } catch (_) {}
          try { ws.close(1011, "capture-start-failed"); } catch (_) {}
        }
        return;
      }
      if (kind === 0x02) {
        const text = new TextDecoder("utf-8").decode(payload);
        if (text) {
          transcript += text;
          setTranscript(transcript);
        }
        return;
      }
      if (kind === 0x01) {
        if (!runtime._audioCount) { runtime._audioCount = 0; }
        runtime._audioCount++;
        runtime.receivedAudioFrames = (runtime.receivedAudioFrames || 0) + 1;
        if (runtime._audioCount <= 3) {
          console.log(`[raon-duplex] Audio frame #${runtime._audioCount} received (${payload.byteLength} bytes)`);
        }
        const pcm = new Float32Array(payload.buffer, payload.byteOffset, Math.floor(payload.byteLength / 4));
        playPcm(pcm);
        return;
      }
      if (kind === 0x05) {
        const errMsg = new TextDecoder("utf-8").decode(payload);
        setStatus(`Server error: ${errMsg}`);
        return;
      }
      if (kind === 0x06) {
        const closeMsg = new TextDecoder("utf-8").decode(payload);
        setStatus(closeMsg ? `Server closed: ${closeMsg}` : "Server closed.");
      }
    }
  };

  window.__raon_fd_demo = runtime;
  return [`Session started: ${sessionId}`, transcript, sessionId];
}
