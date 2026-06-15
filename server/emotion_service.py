"""End-to-end emotion recognition pipeline for the NAO server API."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

from server.config import DEFAULT_BENCHMARK, WHISPER_WORD_TIMESTAMPS
from server.feature_extraction import (
    IemocapFeatureExtractor,
    MoseiFeatureExtractor,
    WhisperTranscriber,
)
from server.feature_extraction.whisper import (
    MAX_AUDIO_SECONDS,
    WHISPER_MODEL_SIZE,
    WhisperTranscript,
    decode_wav_bytes,
)
from server.emotion_inference import EmotionDecoderWrapper, EmotionPrediction

Benchmark = Literal["iemocap", "mosei"]


class EmotionService:
    """
    Orchestrates the full predict use case:

      NAO WAV bytes -> decode once -> Whisper ASR -> online features -> fusion model
    """

    def __init__(self):
        self.transcriber = WhisperTranscriber()
        self.iemocap_features = IemocapFeatureExtractor()
        self.mosei_features = MoseiFeatureExtractor()
        self.decoder = EmotionDecoderWrapper()

    def run(
        self,
        audio_bytes: bytes,
        *,
        benchmark: Benchmark = DEFAULT_BENCHMARK,  # type: ignore[assignment]
        filename_hint: str = "upload.wav",
    ) -> dict[str, Any]:
        benchmark = benchmark.lower()  # type: ignore[assignment]
        if benchmark not in ("iemocap", "mosei"):
            raise ValueError(f"Unsupported benchmark: {benchmark}. Use 'iemocap' or 'mosei'.")

        wav_np, sr = decode_wav_bytes(audio_bytes, filename_hint=filename_hint)

        if benchmark == "iemocap":
            transcript = self.transcriber.transcribe(wav_np, sample_rate=sr)
            if not transcript:
                transcript = "<empty>"
            audio_feats, text_feats = self.iemocap_features.extract(wav_np, transcript)
            prediction = self.decoder.predict_iemocap(
                audio_feats,
                text_feats,
                transcript=transcript,
            )
        else:
            asr = self.transcriber.transcribe(
                wav_np,
                sample_rate=sr,
                word_timestamps=WHISPER_WORD_TIMESTAMPS,
            )
            if isinstance(asr, WhisperTranscript):
                transcript = asr.text or "<empty>"
                if not asr.text:
                    asr = WhisperTranscript(text=transcript, words=[])
            else:
                transcript = asr or "<empty>"
                asr = WhisperTranscript(text=transcript, words=[])

            audio_feats, text_feats = self.mosei_features.extract(wav_np, asr)
            prediction = self.decoder.predict_mosei(
                audio_feats,
                text_feats,
                transcript=transcript,
                placeholder_features=False,
            )

        return self._format_response(
            prediction,
            audio_duration_sec=float(len(wav_np) / sr),
            sample_rate=sr,
        )

    @staticmethod
    def _format_response(
        prediction: EmotionPrediction,
        *,
        audio_duration_sec: float,
        sample_rate: int,
    ) -> dict[str, Any]:
        payload = asdict(prediction)
        payload["audio"] = {
            "duration_sec": round(audio_duration_sec, 3),
            "sample_rate": sample_rate,
            "max_seconds_used": MAX_AUDIO_SECONDS,
        }

        if not prediction.weights_loaded:
            payload["warning"] = (
                "Model checkpoint not loaded — predictions use randomly initialized "
                "weights and are NOT meaningful. Set checkpoint path in server/config.py."
            )

        return payload

    def health(self) -> dict[str, Any]:
        mosei_status = self.mosei_features.status()
        return {
            "status": "ok",
            "benchmarks": ["iemocap", "mosei"],
            "default_benchmark": DEFAULT_BENCHMARK,
            "emotion_decoder": self.decoder.status(),
            "whisper_model": WHISPER_MODEL_SIZE,
            "whisper_word_timestamps": WHISPER_WORD_TIMESTAMPS,
            "mosei_feature_mode": mosei_status.get("mode", "unknown"),
            "mosei_features": mosei_status,
            "iemocap_feature_mode": "wavlm+bert (matches offline scripts)",
        }
