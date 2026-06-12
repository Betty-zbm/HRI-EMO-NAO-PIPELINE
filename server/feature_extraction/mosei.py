"""
Online MOSEI feature extraction (COVAREP + GloVe).

Training/offline MOSEI uses precomputed CSD features:
  - Audio: CMU_MOSEI_COVAREP (74-dim, ~100 Hz)
  - Text:  CMU_MOSEI_TimestampedWordVectors (glove.840B.300d word vectors)

Online deployment uses:
  - COVAREP via MATLAB/Octave subprocess (``CovarepRunner``)
  - Whisper word timestamps + glove.840B.300d (``GloveEncoder``)

Online features may differ from MOSEI CSD — quantify with
``scripts/validate_mosei_online_features.py`` before production use.
"""
from __future__ import annotations

import torch

from server.config import MAX_AUDIO_SECONDS, TARGET_SAMPLE_RATE
from server.feature_extraction.covarep_runner import CovarepRunner
from server.feature_extraction.glove_encoder import GloveEncoder
from server.feature_extraction.iemocap import SeqFeatures, pack_seq_features
from server.feature_extraction.whisper import WhisperTranscript


class MoseiFeatureExtractor:
    """
    Online MOSEI feature extractor: COVAREP audio + GloVe text.

    Expects mono float32 waveform at 16 kHz and a Whisper transcript
    (plain string or ``WhisperTranscript`` with word timestamps).
    """

    PLACEHOLDER = False

    def __init__(self) -> None:
        self._covarep = CovarepRunner()
        self._glove = GloveEncoder()

    def extract_audio(self, wav_np) -> SeqFeatures:
        """Run COVAREP on waveform and return ``[1, L, 74]`` sequence features."""
        frames = self._covarep.extract_frames(wav_np, sample_rate=TARGET_SAMPLE_RATE)
        hidden = torch.from_numpy(frames)
        attention_mask = torch.ones(hidden.size(0), dtype=torch.long)
        return pack_seq_features(hidden, attention_mask)

    def extract_text(
        self,
        transcript: str | WhisperTranscript,
        *,
        max_duration: float = MAX_AUDIO_SECONDS,
    ) -> SeqFeatures:
        """
        Map transcript to GloVe word sequence ``[1, L, 300]``.

        When ``WhisperTranscript.words`` is present, words are ordered by
        Whisper timestamps; otherwise falls back to regex tokenization.
        """
        if isinstance(transcript, WhisperTranscript) and transcript.words:
            vectors = self._glove.encode_timed_words(
                transcript.words,
                max_duration=max_duration,
            )
        else:
            text = transcript.text if isinstance(transcript, WhisperTranscript) else transcript
            vectors = self._glove.encode_text(text)

        hidden = torch.from_numpy(vectors)
        attention_mask = torch.ones(hidden.size(0), dtype=torch.long)
        return pack_seq_features(hidden, attention_mask)

    def extract(
        self,
        wav_np,
        transcript: str | WhisperTranscript,
    ) -> tuple[SeqFeatures, SeqFeatures]:
        duration = min(float(len(wav_np) / TARGET_SAMPLE_RATE), MAX_AUDIO_SECONDS)
        return (
            self.extract_audio(wav_np),
            self.extract_text(transcript, max_duration=duration),
        )

    def status(self) -> dict[str, object]:
        return {
            "mode": "covarep+glove+whisper_word_timestamps",
            "covarep_root": str(self._covarep.covarep_root) if self._covarep.covarep_root else None,
            "covarep_runner": self._covarep.runner_bin,
            "glove_configured": self._glove.is_configured,
        }
