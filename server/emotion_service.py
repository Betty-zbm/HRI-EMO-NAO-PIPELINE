"""End-to-end emotion recognition pipeline for the NAO server API."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from server import checkpoint_registry, runtime_settings
from server.checkpoint_registry import CheckpointInfo
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
from server.emotion_inference import CheckpointRunner, EmotionPrediction

# Legacy ``benchmark`` form values -> registry checkpoint ids
_BENCHMARK_TO_CHECKPOINT = {
    "iemocap": "iemocap_4cls",
    "mosei": "mosei_6cls",
}


class EmotionService:
    """
    Orchestrates the full predict use case:

      NAO WAV bytes -> decode once -> Whisper ASR -> online features -> fusion model

    Which checkpoint runs (and how predictions are formatted) is governed by
    the platform settings in ``server/runtime_settings.py`` unless the request
    explicitly overrides them.
    """

    def __init__(self):
        self.transcriber = WhisperTranscriber()
        self.iemocap_features = IemocapFeatureExtractor()
        self.mosei_features = MoseiFeatureExtractor()
        self.runner = CheckpointRunner()

    @staticmethod
    def resolve_checkpoint(
        checkpoint_id: Optional[str] = None,
        benchmark: Optional[str] = None,
    ) -> tuple[CheckpointInfo, dict[str, Any]]:
        """
        Resolve which checkpoint + output preference to use for a request.

        Priority: explicit ``checkpoint_id`` > legacy ``benchmark`` field >
        platform-configured settings.
        """
        settings = runtime_settings.get_settings()
        output = settings["output"]

        if checkpoint_id:
            info = checkpoint_registry.get(checkpoint_id)
        elif benchmark:
            key = benchmark.strip().lower()
            if key not in _BENCHMARK_TO_CHECKPOINT:
                raise ValueError("benchmark must be 'iemocap' or 'mosei'")
            info = checkpoint_registry.get(_BENCHMARK_TO_CHECKPOINT[key])
            # Legacy MOSEI clients expect calibrated multi-label behaviour
            if info.has_calibrated_thresholds:
                output = {**output, "mode": "calibrated"}
        else:
            info = checkpoint_registry.get(settings["checkpoint_id"])

        ok, reason = checkpoint_registry.availability(info)
        if not ok:
            raise RuntimeError(f"Checkpoint '{info.id}' is not usable: {reason}")
        return info, output

    def run(
        self,
        audio_bytes: bytes,
        *,
        checkpoint_id: Optional[str] = None,
        benchmark: Optional[str] = None,
        filename_hint: str = "upload.wav",
    ) -> dict[str, Any]:
        info, output = self.resolve_checkpoint(checkpoint_id, benchmark)

        wav_np, sr = decode_wav_bytes(audio_bytes, filename_hint=filename_hint)

        if info.feature_family == "wavlm_bert":
            transcript = self.transcriber.transcribe(wav_np, sample_rate=sr)
            if not transcript:
                transcript = "<empty>"
            audio_feats, text_feats = self.iemocap_features.extract(wav_np, transcript)
        else:  # covarep_glove
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

        prediction = self.runner.predict(
            info,
            audio_feats,
            text_feats,
            transcript=transcript,
            output=output,
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
                "weights and are NOT meaningful. Check the checkpoint path."
            )

        return payload

    def health(self) -> dict[str, Any]:
        settings = runtime_settings.get_settings()
        mosei_status = self.mosei_features.status()
        return {
            "status": "ok",
            "active_checkpoint": settings["checkpoint_id"],
            "output_preference": settings["output"],
            "default_benchmark": DEFAULT_BENCHMARK,
            "runner": self.runner.status(),
            "whisper_model": WHISPER_MODEL_SIZE,
            "whisper_word_timestamps": WHISPER_WORD_TIMESTAMPS,
            "mosei_feature_mode": mosei_status.get("mode", "unknown"),
            "mosei_features": mosei_status,
            "iemocap_feature_mode": "wavlm+bert (matches offline scripts)",
        }
