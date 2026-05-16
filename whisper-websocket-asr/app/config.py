import os
import time
from dataclasses import dataclass


def _get_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


@dataclass
class Settings:
    app_version: str = os.getenv("APP_VERSION", time.strftime("asr_%Y%m%d_%H%M%S"))
    model_name: str = os.getenv("MODEL_NAME", "base")
    target_sr: int = _get_int("TARGET_SR", 16000)
    frame_ms: int = _get_int("FRAME_MS", 30)
    vad_aggressiveness: int = _get_int("VAD_AGGRESSIVENESS", 1)

    min_speech_sec: float = _get_float("MIN_SPEECH_SEC", 0.5)
    max_segment_sec: float = _get_float("MAX_SEGMENT_SEC", 8.0)
    end_silence_sec: float = _get_float("END_SILENCE_SEC", 0.5)
    pre_roll_sec: float = _get_float("PRE_ROLL_SEC", 0.3)

    force_language: str | None = os.getenv("FORCE_LANGUAGE", "en") or None

    enable_diarization: bool = _get_bool("ENABLE_DIARIZATION", False)
    hf_token: str | None = os.getenv("HF_TOKEN")

    diarization_every_n_segments: int = _get_int("DIARIZATION_EVERY_N_SEGMENTS", 3)
    diarization_min_audio_sec: float = _get_float("DIARIZATION_MIN_AUDIO_SEC", 10.0)

    port: int = _get_int("PORT", 7860)
    queue_maxsize: int = _get_int("QUEUE_MAXSIZE", 2000)


settings = Settings()
