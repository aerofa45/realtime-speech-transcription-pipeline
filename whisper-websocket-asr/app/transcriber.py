import time
from typing import Any
import numpy as np
import torch
import whisper

from app.audio_utils import rms_energy
from app.state import TranscriptRecord


class WhisperTranscriber:
    def __init__(self, model_name: str, device: str, force_language: str | None):
        self.model_name = model_name
        self.device = device
        self.force_language = force_language
        self.model = whisper.load_model(model_name, device=device)
        self.model.eval()
        print(f"Loaded Whisper model: {model_name} on {device}")

    def is_probable_hallucination(self, text: str) -> bool:
        clean = text.strip()
        lower = clean.lower()

        if lower == "":
            return True

        bad_exact = {
            "thank you.",
            "thank you",
            "thanks for watching.",
            "thanks for watching",
            "like and subscribe",
            "subscribe",
            "bye!",
            "bye bye.",
            "music",
            "[music]",
            "(music)",
        }

        if lower in bad_exact:
            return True

        if len(clean) > 50:
            most_common_char_ratio = max(clean.count(ch) for ch in set(clean)) / len(clean)
            if most_common_char_ratio > 0.45:
                return True

        words = lower.split()
        generic_short = {"oh", "yeah", "let's go.", "let's go", "come on", "alright", "oh yeah", "oh, yeah."}

        if lower in generic_short:
            return True

        if len(words) >= 8 and len(set(words)) / len(words) < 0.40:
            return True

        if self.force_language == "en":
            ascii_ratio = sum(1 for c in clean if ord(c) < 128) / max(len(clean), 1)
            if ascii_ratio < 0.75:
                return True

        return False

    def transcribe_speech_segment(self, segment: dict[str, Any], diarizer) -> TranscriptRecord | None:
        audio_16k = segment["audio"]
        start_sec = segment["start_sec"]
        end_sec = segment["end_sec"]
        duration_sec = segment["duration_sec"]
        energy = rms_energy(audio_16k)

        if self.device == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        audio_30s = whisper.pad_or_trim(audio_16k)
        mel = whisper.log_mel_spectrogram(audio_30s).to(self.model.device)

        options = whisper.DecodingOptions(
            language=self.force_language,
            fp16=(self.device == "cuda"),
            without_timestamps=False,
        )

        with torch.inference_mode():
            decoded = whisper.decode(self.model, mel, options)

        if self.device == "cuda":
            torch.cuda.synchronize()

        latency_sec = time.perf_counter() - start_time
        rtf = latency_sec / max(duration_sec, 1e-6)

        text = decoded.text.strip()

        if self.is_probable_hallucination(text):
            return None

        speaker = diarizer.assign_speaker(start_sec, end_sec)

        return TranscriptRecord(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            speaker=speaker,
            text=text,
            start_sec=start_sec,
            end_sec=end_sec,
            audio_duration_sec=duration_sec,
            latency_sec=latency_sec,
            rtf=rtf,
            rms=energy,
            model=self.model_name,
            device=self.device,
        )
