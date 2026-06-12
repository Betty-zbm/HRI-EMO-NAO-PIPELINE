"""
Wrapper around pretrained fusion + emotion-decoder checkpoints.

Loads ``FusionWithEmotionDecoder`` (IEMOCAP) and ``MoseiFusionWithEmotionDecoder``
(MOSEI), runs a single forward pass on sequence features, and returns structured
prediction results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from models.fusion_with_emotion_decoder import FusionWithEmotionDecoder
from models.mosei_fusion_with_emotion_decoder import MoseiFusionWithEmotionDecoder
from server.config import (
    IEMOCAP_BETA_HIDDEN,
    IEMOCAP_CHECKPOINT,
    IEMOCAP_D_MODEL,
    IEMOCAP_DROPOUT,
    IEMOCAP_LABELS,
    IEMOCAP_N_HEADS,
    IEMOCAP_NUM_LAYERS_DECODER,
    IEMOCAP_NUM_LAYERS_FUSION,
    MOSEI_BETA_HIDDEN,
    MOSEI_CHECKPOINT,
    MOSEI_D_MODEL,
    MOSEI_DROPOUT,
    MOSEI_LABELS,
    MOSEI_N_HEADS,
    MOSEI_NUM_LAYERS_DECODER,
    MOSEI_NUM_LAYERS_FUSION,
    MOSEI_PREDICTION_THRESHOLD,
)
from server.feature_extraction import SeqFeatures
from server.config import MOSEI_AUDIO_DIM, MOSEI_TEXT_DIM


@dataclass
class EmotionPrediction:
    """Structured output from a single fusion-decoder forward pass."""

    benchmark: str
    transcript: str
    labels: list[str]
    scores: dict[str, float]
    predicted: list[str]
    weights_loaded: bool
    placeholder_features: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


def _checkpoint_exists(path: str) -> bool:
    return bool(path) and Path(path).is_file()


def _load_checkpoint(path: str, device: torch.device) -> Optional[dict]:
    if not _checkpoint_exists(path):
        return None
    ckpt = torch.load(path, map_location=device)
    return ckpt if isinstance(ckpt, dict) else {"model_state_dict": ckpt}


def _resolve_mosei_thresholds(ckpt: Optional[dict]) -> tuple[dict[str, float], str]:
    """
    Per-class sigmoid cutoffs for MOSEI multi-label predictions.

    Prefer ``val_calibrated_thresholds`` saved with ``--save_calibrated_ths``;
    otherwise fall back to ``MOSEI_PREDICTION_THRESHOLD`` for every class.
    """
    fallback = {label: float(MOSEI_PREDICTION_THRESHOLD) for label in MOSEI_LABELS}
    if ckpt is None:
        return fallback, "default"

    raw = ckpt.get("val_calibrated_thresholds")
    if raw is None or len(raw) != len(MOSEI_LABELS):
        return fallback, "default"

    return (
        {label: float(t) for label, t in zip(MOSEI_LABELS, raw)},
        "calibrated",
    )


class EmotionDecoderWrapper:
    """
    Lazy-loaded wrapper for IEMOCAP and MOSEI fusion emotion-decoder models.

    Handles checkpoint loading, device placement, forward pass, and conversion
    of raw logits into ``EmotionPrediction`` (softmax for IEMOCAP, sigmoid
    for MOSEI).
    """

    def __init__(self, device: Optional[str] = None):
        self._device = device
        self._iemocap_model: Optional[FusionWithEmotionDecoder] = None
        self._mosei_model: Optional[MoseiFusionWithEmotionDecoder] = None
        self._iemocap_weights_loaded = False
        self._mosei_weights_loaded = False
        self._mosei_thresholds: dict[str, float] = {
            label: float(MOSEI_PREDICTION_THRESHOLD) for label in MOSEI_LABELS
        }
        self._mosei_threshold_source: str = "default"

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(self._device)

    def _build_iemocap_model(self, ckpt: Optional[dict]) -> FusionWithEmotionDecoder:
        cfg = (ckpt or {}).get("args", {})
        model = FusionWithEmotionDecoder(
            d_model=cfg.get("d_model", IEMOCAP_D_MODEL),
            num_emotions=len(IEMOCAP_LABELS),
            n_heads=cfg.get("n_heads", IEMOCAP_N_HEADS),
            num_layers_fusion=cfg.get("num_layers_fusion", IEMOCAP_NUM_LAYERS_FUSION),
            num_layers_decoder=cfg.get("num_layers_decoder", IEMOCAP_NUM_LAYERS_DECODER),
            beta_hidden=cfg.get("beta_hidden", IEMOCAP_BETA_HIDDEN),
            dropout=cfg.get("dropout", IEMOCAP_DROPOUT),
        )
        return model

    def _build_mosei_model(self, ckpt: Optional[dict]) -> MoseiFusionWithEmotionDecoder:
        cfg = (ckpt or {}).get("args", {})
        model = MoseiFusionWithEmotionDecoder(
            d_audio=MOSEI_AUDIO_DIM,
            d_text=MOSEI_TEXT_DIM,
            d_model=cfg.get("d_model", MOSEI_D_MODEL),
            num_emotions=len(MOSEI_LABELS),
            n_heads=cfg.get("n_heads", MOSEI_N_HEADS),
            num_layers_fusion=cfg.get("num_layers_fusion", MOSEI_NUM_LAYERS_FUSION),
            num_layers_decoder=cfg.get("num_layers_decoder", MOSEI_NUM_LAYERS_DECODER),
            beta_hidden=cfg.get("beta_hidden", MOSEI_BETA_HIDDEN),
            dropout=cfg.get("dropout", MOSEI_DROPOUT),
        )
        return model

    def _ensure_iemocap(self) -> FusionWithEmotionDecoder:
        if self._iemocap_model is not None:
            return self._iemocap_model

        ckpt = _load_checkpoint(IEMOCAP_CHECKPOINT, self.device)
        model = self._build_iemocap_model(ckpt).to(self.device)

        if ckpt is not None:
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state)
            self._iemocap_weights_loaded = True

        model.eval()
        self._iemocap_model = model
        return model

    def _ensure_mosei(self) -> MoseiFusionWithEmotionDecoder:
        if self._mosei_model is not None:
            return self._mosei_model

        ckpt = _load_checkpoint(MOSEI_CHECKPOINT, self.device)
        model = self._build_mosei_model(ckpt).to(self.device)

        if ckpt is not None:
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state)
            self._mosei_weights_loaded = True

        self._mosei_thresholds, self._mosei_threshold_source = _resolve_mosei_thresholds(
            ckpt
        )

        model.eval()
        self._mosei_model = model
        return model

    @torch.no_grad()
    def predict_iemocap(
        self,
        audio_feats: SeqFeatures,
        text_feats: SeqFeatures,
        *,
        transcript: str,
    ) -> EmotionPrediction:
        model = self._ensure_iemocap()

        h_a = audio_feats.hidden.to(self.device)
        m_a = audio_feats.key_padding_mask.to(self.device)
        h_t = text_feats.hidden.to(self.device)
        m_t = text_feats.key_padding_mask.to(self.device)

        logits, beta, _z = model(h_a, h_t, m_a, m_t)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

        scores = {label: float(prob) for label, prob in zip(IEMOCAP_LABELS, probs)}
        best_label = IEMOCAP_LABELS[int(logits.argmax(dim=-1).item())]

        beta_mean = float(beta.detach().float().mean().cpu()) if beta is not None else None

        return EmotionPrediction(
            benchmark="iemocap",
            transcript=transcript,
            labels=IEMOCAP_LABELS,
            scores=scores,
            predicted=[best_label],
            weights_loaded=self._iemocap_weights_loaded,
            placeholder_features=False,
            meta={
                "task": "single_label_classification",
                "beta_mean": beta_mean,
                "audio_seq_len": int(audio_feats.hidden.size(1)),
                "text_seq_len": int(text_feats.hidden.size(1)),
            },
        )

    @torch.no_grad()
    def predict_mosei(
        self,
        audio_feats: SeqFeatures,
        text_feats: SeqFeatures,
        *,
        transcript: str,
        placeholder_features: bool = True,
    ) -> EmotionPrediction:
        model = self._ensure_mosei()

        h_a = audio_feats.hidden.to(self.device)
        m_a = audio_feats.key_padding_mask.to(self.device)
        h_t = text_feats.hidden.to(self.device)
        m_t = text_feats.key_padding_mask.to(self.device)

        logits, beta, _z = model(h_a, h_t, m_a, m_t)
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

        scores = {label: float(prob) for label, prob in zip(MOSEI_LABELS, probs)}
        predicted = [
            label
            for label, prob in scores.items()
            if prob >= self._mosei_thresholds[label]
        ]

        beta_mean = float(beta.detach().float().mean().cpu()) if beta is not None else None

        return EmotionPrediction(
            benchmark="mosei",
            transcript=transcript,
            labels=MOSEI_LABELS,
            scores=scores,
            predicted=predicted,
            weights_loaded=self._mosei_weights_loaded,
            placeholder_features=placeholder_features,
            meta={
                "task": "multi_label_classification",
                "threshold_source": self._mosei_threshold_source,
                "thresholds": dict(self._mosei_thresholds),
                "beta_mean": beta_mean,
                "audio_seq_len": int(audio_feats.hidden.size(1)),
                "text_seq_len": int(text_feats.hidden.size(1)),
            },
        )

    def status(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "iemocap_checkpoint": IEMOCAP_CHECKPOINT or None,
            "iemocap_weights_loaded": self._iemocap_weights_loaded,
            "mosei_checkpoint": MOSEI_CHECKPOINT or None,
            "mosei_weights_loaded": self._mosei_weights_loaded,
            "mosei_threshold_source": self._mosei_threshold_source,
            "mosei_thresholds": dict(self._mosei_thresholds),
        }
