# Real-Time Speech Transcription Pipeline

A lightweight Python project that simulates a real-time speech-to-text workflow using audio chunking,
preprocessing, MFCC feature extraction, and a small PyTorch inference model.

## Features

- Load WAV audio and normalize amplitude
- Chunk long recordings into short streaming windows
- Extract MFCC features with `librosa`
- Run a small PyTorch model on each chunk
- Generate timestamped pseudo-transcription output
- Benchmark end-to-end latency and throughput

## Installation

```bash
pip install -r requirements.txt
```

## Run transcription

```bash
python transcribe_stream.py sample_audio.wav --chunk-ms 1000
```

## Run benchmark

```bash
python benchmark.py sample_audio.wav
```

## Notes

This repo is designed as a portfolio project for real-time ASR-style system design.
The model is intentionally lightweight and produces simulated token output rather than a full production ASR decode.
