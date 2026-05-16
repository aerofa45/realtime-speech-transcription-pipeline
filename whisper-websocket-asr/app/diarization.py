import os
import numpy as np

from app.audio_utils import save_wav_temp


class DiarizationManager:
    def __init__(self, pipeline=None, sample_rate=16000, every_n_segments=3, min_audio_sec=10.0):
        self.pipeline = pipeline
        self.sample_rate = sample_rate
        self.every_n_segments = every_n_segments
        self.min_audio_sec = min_audio_sec
        self.session_audio_chunks = []
        self.speaker_turns = []
        self.segment_counter = 0
        self.enabled = pipeline is not None

    def append_audio(self, audio_16k: np.ndarray):
        if self.enabled:
            self.session_audio_chunks.append(audio_16k.astype(np.float32))

    def get_session_audio(self) -> np.ndarray:
        if not self.session_audio_chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.session_audio_chunks).astype(np.float32)

    def reset(self):
        self.session_audio_chunks.clear()
        self.speaker_turns.clear()
        self.segment_counter = 0

    def maybe_update_diarization(self):
        if not self.enabled:
            return

        self.segment_counter += 1
        session_audio = self.get_session_audio()
        session_duration = len(session_audio) / self.sample_rate

        if session_duration < self.min_audio_sec:
            return

        if self.segment_counter % self.every_n_segments != 0:
            return

        wav_path = save_wav_temp(session_audio, self.sample_rate)

        try:
            output = self.pipeline(wav_path)
            turns = []

            if hasattr(output, "exclusive_speaker_diarization"):
                diarization = output.exclusive_speaker_diarization
            elif hasattr(output, "speaker_diarization"):
                diarization = output.speaker_diarization
            else:
                diarization = output

            try:
                for turn, speaker in diarization:
                    turns.append(
                        {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
                    )
            except Exception:
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    turns.append(
                        {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
                    )

            self.speaker_turns = turns
            print(f"Diarization updated: {len(turns)} speaker turns found.")

        except Exception as e:
            print("Diarization update failed:", e)

        finally:
            try:
                os.remove(wav_path)
            except Exception:
                pass

    def assign_speaker(self, start_sec: float, end_sec: float) -> str:
        if not self.enabled or not self.speaker_turns:
            return "Speaker ?"

        overlap_by_speaker = {}

        for turn in self.speaker_turns:
            overlap_start = max(start_sec, turn["start"])
            overlap_end = min(end_sec, turn["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > 0:
                speaker = turn["speaker"]
                overlap_by_speaker[speaker] = overlap_by_speaker.get(speaker, 0.0) + overlap

        if not overlap_by_speaker:
            return "Speaker ?"

        return max(overlap_by_speaker, key=overlap_by_speaker.get)


def load_diarization_pipeline(settings, device: str):
    if not settings.enable_diarization:
        print("Diarization disabled.")
        return None

    if not settings.hf_token:
        print("ENABLE_DIARIZATION=true but HF_TOKEN is missing. Continuing without diarization.")
        return None

    try:
        from pyannote.audio import Pipeline
        import torch

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=settings.hf_token,
        )

        if device == "cuda":
            pipeline.to(torch.device("cuda"))

        print("Community-1 diarization pipeline loaded.")
        return pipeline

    except Exception as e:
        print("Could not load diarization pipeline.")
        print("Reason:", e)
        print("Continuing without diarization.")
        return None
