import json
import queue
import threading
import time

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from app.audio_utils import float32_bytes_to_numpy, rms_energy
from app.config import settings
from app.diarization import DiarizationManager, load_diarization_pipeline
from app.state import AppState
from app.transcriber import WhisperTranscriber
from app.worker import asr_worker_thread


device = "cuda" if torch.cuda.is_available() else "cpu"

state = AppState(queue_maxsize=settings.queue_maxsize)

transcriber = WhisperTranscriber(
    model_name=settings.model_name,
    device=device,
    force_language=settings.force_language,
)

diarization_pipeline = load_diarization_pipeline(settings, device=device)
diarizer = DiarizationManager(
    pipeline=diarization_pipeline,
    sample_rate=settings.target_sr,
    every_n_segments=settings.diarization_every_n_segments,
    min_audio_sec=settings.diarization_min_audio_sec,
)

app = FastAPI(title="Whisper WebSocket ASR")

worker_thread = threading.Thread(
    target=asr_worker_thread,
    args=(state, settings, transcriber, diarizer),
    daemon=True,
)
worker_thread.start()


def load_index_html() -> str:
    with open("app/static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/")
async def index():
    return HTMLResponse(load_index_html())


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "app_version": settings.app_version})


@app.get("/metrics")
async def get_metrics():
    return JSONResponse(state.metrics_payload(settings, diarizer))


@app.post("/reset")
async def reset_session():
    state.reset(diarizer)
    return JSONResponse({"status": "reset_ok", "message": "Session reset complete."})


@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    state.server_state["connected"] = True
    state.server_state["last_error"] = None
    print("WebSocket connected.")

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                print("WebSocket disconnect message received.")
                break

            if "text" in message and message["text"] is not None:
                data = json.loads(message["text"])

                if data.get("type") == "start":
                    state.server_state["browser_sample_rate"] = int(data["sample_rate"])
                    print("Browser sample rate:", state.server_state["browser_sample_rate"])

                elif data.get("type") == "stop":
                    print("Stop message received.")

            elif "bytes" in message and message["bytes"] is not None:
                if state.server_state["browser_sample_rate"] is None:
                    continue

                audio = float32_bytes_to_numpy(message["bytes"])

                state.server_state["raw_chunks_received"] += 1
                state.server_state["raw_audio_seconds_received"] += (
                    len(audio) / state.server_state["browser_sample_rate"]
                )
                state.server_state["last_chunk_rms"] = rms_energy(audio)

                if state.server_state["raw_chunks_received"] % 50 == 0:
                    print(
                        "Received chunks:",
                        state.server_state["raw_chunks_received"],
                        "| audio seconds:",
                        round(state.server_state["raw_audio_seconds_received"], 2),
                        "| RMS:",
                        round(state.server_state["last_chunk_rms"], 6),
                    )

                try:
                    state.audio_queue.put_nowait(
                        {
                            "audio": audio,
                            "source_sr": state.server_state["browser_sample_rate"],
                            "received_at": time.perf_counter(),
                        }
                    )
                except queue.Full:
                    state.server_state["last_error"] = "Audio queue full; dropping chunk."

    except WebSocketDisconnect:
        print("WebSocket disconnected.")

    except Exception as e:
        state.server_state["last_error"] = str(e)
        print("WebSocket error:", e)

    finally:
        state.server_state["connected"] = False
