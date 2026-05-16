from collections import deque
from typing import Any
import numpy as np
import webrtcvad

from app.audio_utils import float32_to_pcm16_bytes


class VADSegmenter:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        aggressiveness: int = 1,
        min_speech_sec: float = 0.5,
        max_segment_sec: float = 8.0,
        end_silence_sec: float = 0.5,
        pre_roll_sec: float = 0.3,
    ):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.frame_bytes = self.frame_samples * 2

        self.vad = webrtcvad.Vad(aggressiveness)

        self.min_speech_samples = int(min_speech_sec * sample_rate)
        self.max_segment_samples = int(max_segment_sec * sample_rate)
        self.end_silence_frames = max(1, int(end_silence_sec / (frame_ms / 1000)))
        self.pre_roll_frames = max(1, int(pre_roll_sec / (frame_ms / 1000)))

        self.byte_buffer = b""
        self.global_sample_cursor = 0

        self.in_speech = False
        self.speech_frames: list[np.ndarray] = []
        self.speech_start_sample = None

        self.pre_roll = deque(maxlen=self.pre_roll_frames)
        self.pre_roll_start_samples = deque(maxlen=self.pre_roll_frames)

        self.silence_count = 0

    def process_audio(self, audio_16k: np.ndarray) -> list[dict[str, Any]]:
        completed = []
        self.byte_buffer += float32_to_pcm16_bytes(audio_16k)

        while len(self.byte_buffer) >= self.frame_bytes:
            frame_bytes = self.byte_buffer[:self.frame_bytes]
            self.byte_buffer = self.byte_buffer[self.frame_bytes:]

            frame_start_sample = self.global_sample_cursor
            self.global_sample_cursor += self.frame_samples

            frame_np = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            try:
                is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
            except Exception:
                is_speech = False

            if not self.in_speech:
                self.pre_roll.append(frame_np)
                self.pre_roll_start_samples.append(frame_start_sample)

                if is_speech:
                    self.in_speech = True
                    self.speech_frames = list(self.pre_roll)
                    self.speech_start_sample = (
                        self.pre_roll_start_samples[0]
                        if self.pre_roll_start_samples
                        else frame_start_sample
                    )
                    self.silence_count = 0

            else:
                self.speech_frames.append(frame_np)

                if is_speech:
                    self.silence_count = 0
                else:
                    self.silence_count += 1

                speech_audio = np.concatenate(self.speech_frames)

                enough_silence = self.silence_count >= self.end_silence_frames
                too_long = len(speech_audio) >= self.max_segment_samples

                if enough_silence or too_long:
                    speech_end_sample = self.speech_start_sample + len(speech_audio)

                    if len(speech_audio) >= self.min_speech_samples:
                        completed.append(
                            {
                                "audio": speech_audio.astype(np.float32),
                                "start_sec": self.speech_start_sample / self.sample_rate,
                                "end_sec": speech_end_sample / self.sample_rate,
                                "duration_sec": len(speech_audio) / self.sample_rate,
                            }
                        )

                    self.in_speech = False
                    self.speech_frames = []
                    self.speech_start_sample = None
                    self.silence_count = 0
                    self.pre_roll.clear()
                    self.pre_roll_start_samples.clear()

        return completed
