"""
Whisper speech-to-text for the online feature-extraction pipeline.

Transcription runs before text feature encoding (BERT for IEMOCAP, GloVe for
MOSEI). Expects mono float32 WAV at 16 kHz — use ``decode_wav_bytes`` on NAO
upload bytes before calling ``transcribe``.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional, Union

import librosa
import numpy as np
import soundfile as sf

from server.config import MAX_AUDIO_SECONDS, TARGET_SAMPLE_RATE, WHISPER_MODEL_SIZE


@dataclass
class TimedWord:
    """Single Whisper word token with second-level timestamps."""

    word: str
    start: float
    end: float


@dataclass
class WhisperTranscript:
    """Whisper output with optional word-level alignment."""

    text: str
    words: list[TimedWord] = field(default_factory=list)


def decode_wav_bytes(
    data: bytes,
    *,
    filename_hint: str = "upload.wav",
    target_sr: int = TARGET_SAMPLE_RATE,
    max_seconds: float | None = MAX_AUDIO_SECONDS,
) -> tuple[np.ndarray, int]:
    """
    Decode NAO-uploaded WAV bytes into a mono float32 waveform for ASR / MOSEI.

    Applies general preprocessing shared across non-WavLM paths:
      - mono mixdown
      - resample to ``target_sr`` when needed
      - peak normalize to [-1, 1]
      - truncate to first ``max_seconds`` when set (no zero-padding)

    Args:
        data: Raw PCM WAV bytes from the /predict multipart upload.
        filename_hint: Original filename (used in error messages only).
        target_sr: Target sample rate (default 16 kHz).
        max_seconds: Maximum clip duration to keep from the start. ``None`` keeps all.

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

    if max_seconds is not None:
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
        model_size: str | None = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_size: Whisper checkpoint size (see ``WHISPER_MODEL_SIZE``).
            device: Torch device string. Auto-detected on first use when omitted.
        """
        self.model_size = model_size or WHISPER_MODEL_SIZE
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

    def transcribe(
        self,
        wav_np: np.ndarray,
        sample_rate: int = TARGET_SAMPLE_RATE,
        *,
        word_timestamps: bool = False,
    ) -> Union[str, WhisperTranscript]:
        """
        Transcribe a mono waveform.

        When ``word_timestamps=False`` (IEMOCAP default), returns plain text.
        When ``word_timestamps=True`` (MOSEI), returns ``WhisperTranscript``
        with per-word ``start`` / ``end`` in seconds.

        Args:
            wav_np: Mono float32 waveform at 16 kHz.
            sample_rate: Informational; resample upstream if not 16 kHz.
            word_timestamps: Enable Whisper word-level timestamps.

        Returns:
            Transcript string, or ``WhisperTranscript`` when timestamps requested.
        """
        result = self._transcribe_raw(wav_np, word_timestamps=word_timestamps)
        if word_timestamps:
            return result
        return result.text

    def _transcribe_raw(
        self,
        wav_np: np.ndarray,
        *,
        word_timestamps: bool,
    ) -> WhisperTranscript:
        self._ensure_loaded()

        kwargs: dict = {
            "fp16": self.device == "cuda",
            "language": None,  # auto-detect
        }
        if word_timestamps:
            kwargs["word_timestamps"] = True

        raw = self._model.transcribe(wav_np, **kwargs)
        text = (raw.get("text") or "").strip()

        words: list[TimedWord] = []
        for segment in raw.get("segments") or []:
            for item in segment.get("words") or []:
                token = (item.get("word") or "").strip()
                if not token:
                    continue
                words.append(
                    TimedWord(
                        word=token,
                        start=float(item.get("start", 0.0)),
                        end=float(item.get("end", 0.0)),
                    )
                )

        words.sort(key=lambda w: (w.start, w.end))
        return WhisperTranscript(text=text, words=words)
