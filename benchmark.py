from __future__ import annotations

import argparse
import statistics
import time

from transcribe_stream import transcribe_audio


def benchmark(audio_path: str, runs: int = 5, chunk_ms: int = 1000) -> None:
    timings = []
    for idx in range(runs):
        start = time.perf_counter()
        _ = transcribe_audio(audio_path, chunk_ms=chunk_ms)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        print(f"Run {idx + 1}: {elapsed:.4f}s")

    print("\nBenchmark summary")
    print(f"runs: {runs}")
    print(f"min: {min(timings):.4f}s")
    print(f"max: {max(timings):.4f}s")
    print(f"mean: {statistics.mean(timings):.4f}s")
    if len(timings) > 1:
        print(f"stdev: {statistics.stdev(timings):.4f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the transcription pipeline")
    parser.add_argument("audio_path", help="Path to input audio")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--chunk-ms", type=int, default=1000)
    args = parser.parse_args()
    benchmark(args.audio_path, runs=args.runs, chunk_ms=args.chunk_ms)
