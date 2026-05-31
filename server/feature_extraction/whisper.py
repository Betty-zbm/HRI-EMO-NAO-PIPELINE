"""
Whisper speech-to-text for the online feature-extraction pipeline.

Transcription runs before text feature encoding (BERT for IEMOCAP, word vectors
for MOSEI). Expects mono float32 WAV at 16 kHz — use ``decode_wav_bytes`` on
NAO upload bytes before calling ``transcribe``.
"""
from __future__ import annotations

import io
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

# NAO posts mono PCM WAV (16 kHz after client-side conversion in the test page).
TARGET_SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 10.0

# Fixed Whisper settings (openai-whisper model size)
# Options: tiny | base | small | medium | large
WHISPER_MODEL_SIZE = "base"


def decode_wav_bytes(
    data: bytes,
    *,
    filename_hint: str = "upload.wav",
    target_sr: int = TARGET_SAMPLE_RATE,
    max_seconds: float = MAX_AUDIO_SECONDS,
) -> tuple[np.ndarray, int]:
    """
    Decode NAO-uploaded WAV bytes into a mono float32 waveform for ASR / MOSEI.

    Applies general preprocessing shared across non-WavLM paths:
      - mono mixdown
      - resample to ``target_sr`` when needed
      - peak normalize to [-1, 1]
      - truncate to first ``max_seconds`` (no zero-padding)

    Args:
        data: Raw PCM WAV bytes from the /predict multipart upload.
        filename_hint: Original filename (used in error messages only).
        target_sr: Target sample rate (default 16 kHz).
        max_seconds: Maximum clip duration to keep from the start.

    Returns:
        Tuple of (waveform, sample_rate).

    Raises:
        ValueError: When ``data`` is empty or not a valid WAV payload.
    """
    if not data:
        raise ValueError("Empty audio upload")

    try:
        wav_data, sr = sf.read(io.BytesIO(data), always_2d=True, dtype="float32")
    except Exception as exc:
        raise ValueError(
            f"Expected PCM WAV from NAO upload (file: {filename_hint}): {exc}"
        ) from exc

    wav = wav_data.mean(axis=1)

    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    peak = float(np.abs(wav).max())
    if peak > 0:
        wav = wav / peak

    max_samples = int(target_sr * max_seconds)
    if wav.shape[0] > max_samples:
        wav = wav[:max_samples]

    return wav.astype(np.float32), sr


class WhisperTranscriber:
    """
    Lazy-loaded Whisper ASR wrapper (``openai-whisper``).

    The model is loaded on first ``transcribe`` call so server startup stays
    fast when no prediction requests have been made yet.
    """

    def __init__(
        self,
        *,
        model_size: str = WHISPER_MODEL_SIZE,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_size: Whisper checkpoint size (see ``WHISPER_MODEL_SIZE``).
            device: Torch device string. Auto-detected on first use when omitted.
        """
        self.model_size = model_size
        self._device = device
        self._model = None

    @property
    def device(self) -> str:
        """Resolve and cache the compute device (CUDA when available)."""
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _ensure_loaded(self) -> None:
        """Load the Whisper checkpoint into memory if not already present."""
        if self._model is not None:
            return
        import whisper

        self._model = whisper.load_model(self.model_size, device=self.device)

    def transcribe(self, wav_np: np.ndarray, sample_rate: int = 16_000) -> str:
        """
        Transcribe a mono waveform to plain text.

        Processing:
          1. Lazy-load Whisper on first call.
          2. Run ``model.transcribe`` with fp16 on GPU, language auto-detect.
          3. Return stripped transcript (empty string when no speech detected).

        Args:
            wav_np: Mono float32 waveform. Must already be at 16 kHz
                (``sample_rate`` is accepted for API compatibility but Whisper
                assumes 16 kHz input).
            sample_rate: Sample rate of ``wav_np`` (informational; resample
                upstream if not 16 kHz).

        Returns:
            Transcript string, or ``""`` when Whisper returns no text.
        """
        self._ensure_loaded()
        result = self._model.transcribe(
            wav_np,
            fp16=(self.device == "cuda"),
            language=None,  # auto-detect
        )
        return (result.get("text") or "").strip()
