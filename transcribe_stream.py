from __future__ import annotations

import argparse
import time
from typing import List

import numpy as np
import torch
from torch import nn

from audio_utils import load_audio, normalize_audio, stream_chunks, extract_mfcc, summarize_energy


VOCAB = [
    "hello", "world", "audio", "signal", "model",
    "stream", "python", "speech", "caption", "test"
]


class TinyASRHead(nn.Module):
    def __init__(self, input_dim: int = 13, hidden_dim: int = 32, vocab_size: int = len(VOCAB)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def decode_logits(logits: torch.Tensor) -> str:
    token_id = int(torch.argmax(logits).item())
    return VOCAB[token_id]


def transcribe_audio(path: str, chunk_ms: int = 1000) -> List[str]:
    torch.manual_seed(42)
    model = TinyASRHead()
    model.eval()

    samples, sr = load_audio(path)
    samples = normalize_audio(samples)

    outputs: List[str] = []
    start_time = time.perf_counter()

    for chunk in stream_chunks(samples, sr, chunk_ms=chunk_ms):
        mfcc = extract_mfcc(chunk.samples, chunk.sample_rate)
        features = torch.tensor(np.mean(mfcc, axis=0), dtype=torch.float32)
        with torch.no_grad():
            logits = model(features)
        token = decode_logits(logits)
        energy = summarize_energy(chunk.samples)
        outputs.append(
            f"[{chunk.start_sec:05.2f}-{chunk.end_sec:05.2f}s] token={token} energy={energy:.4f}"
        )

    elapsed = time.perf_counter() - start_time
    print(f"Processed {len(samples)/sr:.2f}s of audio in {elapsed:.3f}s")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulated real-time speech transcription pipeline")
    parser.add_argument("audio_path", help="Path to WAV or other librosa-supported audio file")
    parser.add_argument("--chunk-ms", type=int, default=1000, help="Chunk size in milliseconds")
    args = parser.parse_args()

    lines = transcribe_audio(args.audio_path, chunk_ms=args.chunk_ms)
    print("Transcription output:")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
