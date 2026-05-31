"""Application service: orchestrates emotion recognition for the NAO API."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

from server.config import DEFAULT_BENCHMARK
from server.feature_extraction import (
    IemocapFeatureExtractor,
    MoseiFeatureExtractor,
    WhisperTranscriber,
)
from server.feature_extraction.whisper import (
    MAX_AUDIO_SECONDS,
    WHISPER_MODEL_SIZE,
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

        transcript = self.transcriber.transcribe(wav_np, sample_rate=sr)
        if not transcript:
            transcript = "<empty>"

        if benchmark == "iemocap":
            audio_feats, text_feats = self.iemocap_features.extract(wav_np, transcript)
            prediction = self.decoder.predict_iemocap(
                audio_feats,
                text_feats,
                transcript=transcript,
            )
        else:
            audio_feats, text_feats = self.mosei_features.extract(wav_np, transcript)
            prediction = self.decoder.predict_mosei(
                audio_feats,
                text_feats,
                transcript=transcript,
                placeholder_features=MoseiFeatureExtractor.PLACEHOLDER,
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

        if prediction.placeholder_features:
            payload["warning_features"] = (
                "MOSEI features use a placeholder online extractor (not COVAREP / "
                "TimestampedWordVectors). Replace before production inference."
            )

        return payload

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "benchmarks": ["iemocap", "mosei"],
            "default_benchmark": DEFAULT_BENCHMARK,
            "emotion_decoder": self.decoder.status(),
            "whisper_model": WHISPER_MODEL_SIZE,
            "mosei_feature_mode": "placeholder",
            "iemocap_feature_mode": "wavlm+bert (matches offline scripts)",
        }
