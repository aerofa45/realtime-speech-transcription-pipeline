# Whisper WebSocket ASR with WebRTC VAD, Reset, Metrics Dashboard, and Optional Diarization

A near-real-time browser microphone transcription system using OpenAI Whisper, FastAPI WebSockets, WebRTC VAD, and a live latency/RTF dashboard.

The model can be tested here: [Whisper_LIVE_Transcription](https://huggingface.co/spaces/aerofa24/Whisper_Real_Time_ASR)

This repository was converted from the working Colab notebook into a clean GitHub/Hugging Face Spaces project structure.

## What this project does

The app opens a browser dashboard. When the user clicks **Start Microphone**, browser audio is streamed to a FastAPI backend over WebSocket. The backend receives raw Float32 PCM chunks, resamples the audio to 16 kHz, applies WebRTC VAD, buffers speech segments, runs Whisper transcription, filters common hallucinations, and updates a live dashboard with latency and real-time-factor metrics.

Core features:

-   Browser microphone capture
-   WebSocket audio streaming
-   Thread-safe backend audio queue
-   WebRTC VAD speech segmentation
-   Torchaudio resampling instead of SciPy
-   Whisper transcription
-   English-only mode by default to reduce random-language hallucinations
-   Reset / New Recording endpoint
-   Live backend debug counters
-   Separate Latency and RTF charts
-   Optional pyannote Community-1 diarization support

## Why the final architecture uses a thread queue

During notebook testing, the browser and backend were receiving audio correctly, but the async queue was not being consumed inside the Colab/Uvicorn thread setup. The final fix uses a normal Python `queue.Queue` and a dedicated background worker thread.

This was the key evidence from debugging:

``` text
raw_chunks_received > 0
raw_audio_seconds_received > 0
queue_size = raw_chunks_received
vad_segments_created = 0
```

That meant:

``` text
Browser mic worked.
WebSocket worked.
Backend received audio.
Queue put worked.
Worker did not consume queue.
```

The production version therefore uses:

``` text
WebSocket route -> queue.Queue -> worker thread -> VAD -> Whisper
```

## Architecture

``` mermaid
flowchart TD
    A[Browser Microphone] --> B[Web Audio API]
    B --> C[Float32 PCM Chunks]
    C --> D[WebSocket /ws/audio]
    D --> E[FastAPI Backend]
    E --> F[Thread-safe queue.Queue]
    F --> G[ASR Worker Thread]
    G --> H[Torchaudio Resampling 48kHz/44.1kHz -> 16kHz]
    H --> I[WebRTC VAD]
    I --> J[Speech Segment Buffer]
    J --> K[Whisper Mel Spectrogram + Decode]
    K --> L[Hallucination Filter]
    L --> M[Transcript Records]
    M --> N[Metrics: Latency, RTF, RMS]
    N --> O[Dashboard Charts]
    M --> P[transcription_log.txt]

    J -. optional .-> Q[pyannote Community-1 Diarization]
    Q -. speaker labels .-> M
```

## Runtime sequence

``` mermaid
sequenceDiagram
    participant Browser
    participant WS as FastAPI WebSocket
    participant Q as Audio Queue
    participant Worker as ASR Worker Thread
    participant VAD as WebRTC VAD
    participant Whisper
    participant UI as Dashboard

    Browser->>WS: send {"type":"start","sample_rate":48000}
    Browser->>WS: send Float32 audio bytes
    WS->>Q: put audio chunk
    Worker->>Q: get audio chunk
    Worker->>Worker: resample to 16 kHz
    Worker->>VAD: classify 30ms frames
    VAD-->>Worker: completed speech segment
    Worker->>Whisper: transcribe segment
    Whisper-->>Worker: text
    Worker->>UI: metrics available through /metrics
    Browser->>UI: refresh /metrics every second
```

## Repository structure

``` text
whisper-websocket-asr/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, WebSocket, reset, metrics
│   ├── config.py            # environment-based configuration
│   ├── state.py             # shared app state and transcript dataclass
│   ├── audio_utils.py       # Float32 conversion, resampling, RMS, WAV temp
│   ├── vad_segmenter.py     # WebRTC VAD segmenter
│   ├── transcriber.py       # Whisper model + hallucination filter
│   ├── diarization.py       # optional pyannote Community-1 manager
│   ├── worker.py            # queue-consuming ASR worker thread
│   └── static/
│       └── index.html       # browser dashboard
├── requirements.txt
├── requirements-diarization.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

## Local setup

Create a virtual environment:

``` bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Run the app:

``` bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Open:

``` text
http://127.0.0.1:7860
```

For public browser microphone access, use HTTPS deployment, such as Hugging Face Spaces, Cloudflare tunnel, or another HTTPS host. Browser microphone APIs generally require HTTPS except on localhost.

## Environment variables

| Variable | Default | Meaning |
|----------------------|----------------------------:|----------------------|
| `MODEL_NAME` | `base` | Whisper model name: `tiny`, `base`, `small`, etc. |
| `TARGET_SR` | `16000` | Backend target sample rate |
| `FRAME_MS` | `30` | WebRTC VAD frame size |
| `VAD_AGGRESSIVENESS` | `1` | 0 least strict, 3 most strict |
| `MIN_SPEECH_SEC` | `0.5` | Minimum accepted speech segment length |
| `MAX_SEGMENT_SEC` | `8.0` | Force transcribe after this duration |
| `END_SILENCE_SEC` | `0.5` | End segment after this much silence |
| `PRE_ROLL_SEC` | `0.3` | Keep audio before VAD trigger |
| `FORCE_LANGUAGE` | `en` | Whisper language; use empty string for auto |
| `ENABLE_DIARIZATION` | `false` | Enable pyannote diarization |
| `HF_TOKEN` | unset | Hugging Face token for pyannote gated model |
| `PORT` | `7860` | Server port |

## Optional diarization

The core app does not require pyannote. To enable speaker diarization:

1.  Accept access to `pyannote/speaker-diarization-community-1` on Hugging Face.
2.  Create a Hugging Face token with read access to public gated repos.
3.  Install optional dependencies:

``` bash
pip install -r requirements-diarization.txt
```

4.  Set environment variables:

``` bash
export ENABLE_DIARIZATION=true
export HF_TOKEN=hf_your_token_here
```

Then run the app.

Diarization is delayed by design. It updates after enough accumulated audio exists:

``` text
DIARIZATION_MIN_AUDIO_SEC = 10.0
DIARIZATION_EVERY_N_SEGMENTS = 3
```

## Docker

Build:

``` bash
docker build -t whisper-websocket-asr .
```

Run:

``` bash
docker run --rm -p 7860:7860 --env-file .env whisper-websocket-asr
```

Open:

``` text
http://127.0.0.1:7860
```

## Hugging Face Spaces deployment

Use a Docker Space.

Recommended Space SDK metadata is included at the top of this README when pushed to Hugging Face if desired. For Docker Spaces, Hugging Face expects the container to listen on port `7860`.

Steps:

1.  Create a new Hugging Face Space.
2.  Choose **Docker** as the SDK.
3.  Push this repo.
4.  Set Space secrets if needed:
    -   `HF_TOKEN`
    -   `ENABLE_DIARIZATION=true`
5.  Keep `ENABLE_DIARIZATION=false` for the first deployment test.

## Debug interpretation

The dashboard shows browser and backend counters.

Important fields:

``` text
Browser chunks sent > 0
```

The browser is recording microphone audio.

``` text
raw_chunks_received > 0
```

FastAPI is receiving WebSocket binary audio.

``` text
worker_items_processed > 0
```

The background worker is consuming the queue.

``` text
queue_size = 0 or small
```

The worker keeps up with input.

``` text
vad_segments_created > 0
```

WebRTC VAD created speech segments.

``` text
segments_transcribed > 0
```

Whisper transcription is working.

``` text
RTF < 1.0
```

Faster than real time.

## Common issues

### Browser says connected but no transcript

Check dashboard:

``` text
raw_chunks_received > 0
queue_size increasing
worker_items_processed = 0
```

The worker is not consuming the queue. This repo uses `queue.Queue` + thread to avoid that issue.

### Random Korean/Arabic/Russian hallucinations

Keep:

``` bash
FORCE_LANGUAGE=en
```

And keep the hallucination filter enabled.

### pyannote dependency problems

Keep:

``` bash
ENABLE_DIARIZATION=false
```

Deploy the core ASR first. Add diarization only after the app works.

### Browser microphone blocked

Use HTTPS. Browser microphone access may be blocked on public HTTP.

## Limitations

-   Diarization is optional and adds latency. However ONNX optimization makes the latency decrease by 40 percent.
