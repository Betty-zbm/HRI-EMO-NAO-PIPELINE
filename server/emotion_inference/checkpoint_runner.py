"""
Generic, registry-driven inference runner.

Unlike ``EmotionDecoderWrapper`` (fixed IEMOCAP + MOSEI checkpoints from
``server/config.py``), this runner can load *any* checkpoint registered in
``server/checkpoint_registry.py`` and applies the experimenter-selected output
preference (single / top-N / threshold / calibrated) from the platform.

Loaded models are cached per checkpoint id, so switching settings in the
platform only pays the load cost once per checkpoint.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from models.mosei_fusion_with_emotion_decoder import MoseiFusionWithEmotionDecoder
from server.checkpoint_registry import CheckpointInfo
from server.emotion_inference.decoder_wrapper import EmotionPrediction
from server.feature_extraction import SeqFeatures

# Input dims per online feature pipeline
_FAMILY_DIMS = {
    "wavlm_bert": (768, 768),      # WavLM audio, BERT text
    "covarep_glove": (74, 300),    # COVAREP audio, GloVe text
}

# emo_cols names used by MOSEI training scripts -> serving label names
_EMO_COL_TO_LABEL = {
    "emo_happy": "happy",
    "emo_sad": "sad",
    "emo_anger": "angry",
    "emo_fear": "fear",
    "emo_disgust": "disgust",
    "emo_surprise": "surprise",
}


class _LoadedCheckpoint:
    """A built model plus everything resolved from its checkpoint file."""

    def __init__(self, info: CheckpointInfo, device: torch.device):
        self.info = info
        ckpt = torch.load(info.checkpoint_path, map_location=device, weights_only=False)
        if not isinstance(ckpt, dict):
            ckpt = {"model_state_dict": ckpt}
        args: dict[str, Any] = ckpt.get("args", {}) or {}

        self.labels = self._resolve_labels(info, ckpt)
        d_audio, d_text = _FAMILY_DIMS[info.feature_family]
        self.model = MoseiFusionWithEmotionDecoder(
            d_audio=d_audio,
            d_text=d_text,
            d_model=args.get("d_model", 256),
            num_emotions=len(self.labels),
            n_heads=args.get("n_heads", 4),
            num_layers_fusion=args.get("num_layers_fusion", 1),
            num_layers_decoder=args.get("num_layers_decoder", 2),
            beta_hidden=args.get("beta_hidden", 128),
            dropout=args.get("dropout", 0.1),
        ).to(device)
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state)
        self.model.eval()

        self.max_len_audio: Optional[int] = args.get("max_len_audio")
        self.max_len_text: Optional[int] = args.get("max_len_text")

        # Per-class calibrated thresholds (MOSEI multi-label), aligned to labels
        raw_ths = ckpt.get("val_calibrated_thresholds")
        if raw_ths is not None and len(raw_ths) == len(self.labels):
            self.calibrated_thresholds = {
                label: float(t) for label, t in zip(self.labels, raw_ths)
            }
        else:
            self.calibrated_thresholds = None

    @staticmethod
    def _resolve_labels(info: CheckpointInfo, ckpt: dict) -> list[str]:
        label2id = ckpt.get("label2id")
        if label2id:
            id2label = {v: k for k, v in label2id.items()}
            return [id2label[i] for i in sorted(id2label)]
        emo_cols = ckpt.get("emo_cols")
        if emo_cols:
            return [_EMO_COL_TO_LABEL.get(c, c.removeprefix("emo_")) for c in emo_cols]
        return list(info.labels)


class CheckpointRunner:
    """Load registry checkpoints on demand and run configured predictions."""

    def __init__(self, device: Optional[str] = None):
        self._device = device
        self._cache: dict[str, _LoadedCheckpoint] = {}

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(self._device)

    def _ensure(self, info: CheckpointInfo) -> _LoadedCheckpoint:
        if info.id not in self._cache:
            self._cache[info.id] = _LoadedCheckpoint(info, self.device)
        return self._cache[info.id]

    @torch.no_grad()
    def predict(
        self,
        info: CheckpointInfo,
        audio_feats: SeqFeatures,
        text_feats: SeqFeatures,
        *,
        transcript: str,
        output: dict[str, Any],
    ) -> EmotionPrediction:
        """
        Run one forward pass and format the prediction per output preference.

        ``output``: validated dict from ``runtime_settings`` —
        ``{"mode": "single"|"top_n"|"threshold"|"calibrated", "top_n": int, "threshold": float}``.
        """
        loaded = self._ensure(info)

        h_a = audio_feats.hidden.to(self.device)
        m_a = audio_feats.key_padding_mask.to(self.device)
        h_t = text_feats.hidden.to(self.device)
        m_t = text_feats.key_padding_mask.to(self.device)

        # Truncate to training sequence caps so inference matches training
        if loaded.max_len_audio and h_a.size(1) > loaded.max_len_audio:
            h_a = h_a[:, : loaded.max_len_audio, :]
            m_a = m_a[:, : loaded.max_len_audio]
        if loaded.max_len_text and h_t.size(1) > loaded.max_len_text:
            h_t = h_t[:, : loaded.max_len_text, :]
            m_t = m_t[:, : loaded.max_len_text]

        logits, beta, _z = loaded.model(h_a, h_t, m_a, m_t)

        if info.task == "multi_label":
            probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
        else:
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

        scores = {label: float(p) for label, p in zip(loaded.labels, probs)}
        predicted, applied = self._apply_output_preference(
            scores, output, loaded.calibrated_thresholds
        )

        beta_mean = float(beta.detach().float().mean().cpu()) if beta is not None else None

        return EmotionPrediction(
            benchmark=info.dataset.lower(),
            transcript=transcript,
            labels=list(loaded.labels),
            scores=scores,
            predicted=predicted,
            weights_loaded=True,
            placeholder_features=False,
            meta={
                "checkpoint_id": info.id,
                "checkpoint_name": info.display_name,
                "task": (
                    "multi_label_classification"
                    if info.task == "multi_label"
                    else "single_label_classification"
                ),
                "feature_family": info.feature_family,
                "output_preference": applied,
                "beta_mean": beta_mean,
                "audio_seq_len": int(audio_feats.hidden.size(1)),
                "text_seq_len": int(text_feats.hidden.size(1)),
            },
        )

    @staticmethod
    def _apply_output_preference(
        scores: dict[str, float],
        output: dict[str, Any],
        calibrated_thresholds: Optional[dict[str, float]],
    ) -> tuple[list[str], dict[str, Any]]:
        """Return (predicted labels, description of what was applied)."""
        ranked = sorted(scores, key=scores.get, reverse=True)
        mode = output.get("mode", "single")

        if mode == "calibrated" and calibrated_thresholds:
            predicted = [l for l in ranked if scores[l] >= calibrated_thresholds[l]]
            return predicted, {
                "mode": "calibrated",
                "thresholds": dict(calibrated_thresholds),
            }

        if mode == "top_n":
            n = max(1, int(output.get("top_n", 2)))
            return ranked[:n], {"mode": "top_n", "top_n": n}

        if mode == "threshold":
            t = float(output.get("threshold", 0.5))
            predicted = [l for l in ranked if scores[l] >= t]
            if not predicted:  # always return at least the top emotion
                predicted = ranked[:1]
            return predicted, {"mode": "threshold", "threshold": t}

        # default: single top emotion
        return ranked[:1], {"mode": "single"}

    def status(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "loaded_checkpoints": sorted(self._cache),
        }
