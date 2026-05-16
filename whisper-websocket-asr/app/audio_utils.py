import tempfile
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF


def float32_bytes_to_numpy(data: bytes) -> np.ndarray:
    audio = np.frombuffer(data, dtype=np.float32)
    audio = np.nan_to_num(audio)
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32)


def resample_audio(audio: np.ndarray, source_sr: int, target_sr: int = 16000) -> np.ndarray:
    """
    Resample browser/device sample rate to Whisper/WebRTC VAD sample rate.
    Uses torchaudio instead of scipy to avoid NumPy/SciPy binary conflicts.
    """
    if source_sr == target_sr:
        return audio.astype(np.float32)

    audio_tensor = torch.from_numpy(audio.astype(np.float32))
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    with torch.inference_mode():
        resampled = AF.resample(audio_tensor, orig_freq=source_sr, new_freq=target_sr)

    return resampled.squeeze(0).cpu().numpy().astype(np.float32)


def float32_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767).astype(np.int16)
    return pcm16.tobytes()


def rms_energy(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))


def save_wav_temp(audio: np.ndarray, sr: int = 16000) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, audio, sr)
    return tmp.name
