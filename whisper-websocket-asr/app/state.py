from dataclasses import dataclass, asdict
from typing import Any
import queue


@dataclass
class TranscriptRecord:
    timestamp: str
    speaker: str
    text: str
    start_sec: float
    end_sec: float
    audio_duration_sec: float
    latency_sec: float
    rtf: float
    rms: float
    model: str
    device: str


class AppState:
    def __init__(self, queue_maxsize: int = 2000):
        self.audio_queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self.reset_requested: bool = False
        self.worker_running: bool = False

        self.transcripts: list[TranscriptRecord] = []
        self.metrics_rows: list[dict[str, Any]] = []

        self.server_state: dict[str, Any] = {
            "connected": False,
            "browser_sample_rate": None,
            "last_error": None,
            "raw_chunks_received": 0,
            "raw_audio_seconds_received": 0.0,
            "last_chunk_rms": 0.0,
            "worker_items_processed": 0,
            "vad_segments_created": 0,
            "segments_transcribed": 0,
            "segments_filtered": 0,
        }

    def reset(self, diarizer) -> None:
        self.transcripts.clear()
        self.metrics_rows.clear()
        diarizer.reset()

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except Exception:
                break

        self.reset_requested = True

        self.server_state["last_error"] = None
        self.server_state["browser_sample_rate"] = None
        self.server_state["raw_chunks_received"] = 0
        self.server_state["raw_audio_seconds_received"] = 0.0
        self.server_state["last_chunk_rms"] = 0.0
        self.server_state["worker_items_processed"] = 0
        self.server_state["vad_segments_created"] = 0
        self.server_state["segments_transcribed"] = 0
        self.server_state["segments_filtered"] = 0

        with open("transcription_log.txt", "w", encoding="utf-8") as f:
            f.write("")

    def metrics_payload(self, settings, diarizer) -> dict[str, Any]:
        return {
            "app_version": settings.app_version,
            "port": settings.port,
            "connected": self.server_state["connected"],
            "browser_sample_rate": self.server_state["browser_sample_rate"],
            "last_error": self.server_state["last_error"],
            "raw_chunks_received": self.server_state["raw_chunks_received"],
            "raw_audio_seconds_received": self.server_state["raw_audio_seconds_received"],
            "last_chunk_rms": self.server_state["last_chunk_rms"],
            "worker_running": self.worker_running,
            "worker_items_processed": self.server_state["worker_items_processed"],
            "vad_segments_created": self.server_state["vad_segments_created"],
            "segments_transcribed": self.server_state["segments_transcribed"],
            "segments_filtered": self.server_state["segments_filtered"],
            "queue_size": self.audio_queue.qsize(),
            "diarization_enabled": diarizer.enabled,
            "num_speaker_turns": len(diarizer.speaker_turns),
            "num_transcripts": len(self.transcripts),
            "transcripts": [asdict(t) for t in self.transcripts[-30:]],
            "metrics": self.metrics_rows[-100:],
            "speaker_turns": diarizer.speaker_turns[-50:],
        }
