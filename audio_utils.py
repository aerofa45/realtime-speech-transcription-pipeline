from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Tuple
import numpy as np
import librosa


@dataclass
class AudioChunk:
    start_sec: float
    end_sec: float
    samples: np.ndarray
    sample_rate: int


def load_audio(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    samples, sample_rate = librosa.load(path, sr=sr, mono=True)
    return samples.astype(np.float32), sample_rate


def normalize_audio(samples: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak == 0.0:
        return samples
    return (samples / peak).astype(np.float32)


def stream_chunks(samples: np.ndarray, sample_rate: int, chunk_ms: int = 1000) -> Generator[AudioChunk, None, None]:
    chunk_size = max(1, int(sample_rate * chunk_ms / 1000))
    total = len(samples)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        yield AudioChunk(
            start_sec=start / sample_rate,
            end_sec=end / sample_rate,
            samples=samples[start:end],
            sample_rate=sample_rate,
        )


def extract_mfcc(samples: np.ndarray, sample_rate: int, n_mfcc: int = 13) -> np.ndarray:
    if samples.size == 0:
        return np.zeros((1, n_mfcc), dtype=np.float32)
    mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=n_mfcc)
    return mfcc.T.astype(np.float32)


def summarize_energy(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))
