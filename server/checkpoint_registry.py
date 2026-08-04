"""
Registry of trained checkpoints selectable from the configuration platform.

Each entry describes one deployable checkpoint: where the weights live, which
online feature pipeline it needs (WavLM+BERT vs COVAREP+GloVe), what task it
solves (single-label vs multi-label), and user-facing guidance text shown in
the platform UI.

Only checkpoints that have been validated through the *online* pipeline
(raw WAV -> Whisper -> live features -> model) are exposed here. Legacy runs
(e.g. ``iemocap_fusion_seq_decoder_v*``, ``mosei_fusion_decoder_v2``) are
intentionally not registered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from server.config import COVAREP_ROOT, COVAREP_RUNNER_BIN, GLOVE_MODEL_PATH, ROOT_DIR

FeatureFamily = Literal["wavlm_bert", "covarep_glove"]
TaskType = Literal["single_label", "multi_label"]


@dataclass(frozen=True)
class CheckpointInfo:
    """Static description of one selectable checkpoint."""

    id: str
    display_name: str
    dataset: str                      # IEMOCAP / MELD / MOSEI
    task: TaskType
    feature_family: FeatureFamily
    checkpoint_path: str              # absolute path to .pt file
    labels: list[str]                 # fallback order; ckpt label2id/emo_cols wins
    description: str                  # one-line summary for the UI
    best_for: str                     # usage guidance for experimenters
    metrics: dict[str, float] = field(default_factory=dict)
    recommended: bool = False
    variant_of: Optional[str] = None  # id of the primary entry this is a seed/variant of
    has_calibrated_thresholds: bool = False
    latency: str = "~2-3 s per utterance"  # measured end-to-end on the serving machine
    startup_note: str = ""                 # one-time first-request cost, if notable
    scenario: str = ""                     # short usage-scenario tag shown on the card

    @property
    def num_classes(self) -> int:
        return len(self.labels)


def _run(path: str) -> str:
    return str(ROOT_DIR / "runs" / path)


_CHECKPOINTS: list[CheckpointInfo] = [
    # ── IEMOCAP · WavLM+BERT ──────────────────────────────────────────────────
    CheckpointInfo(
        id="iemocap_4cls",
        display_name="IEMOCAP · 4-class emotion",
        dataset="IEMOCAP",
        task="single_label",
        feature_family="wavlm_bert",
        checkpoint_path=_run("iemocap_4cls_seed7777/best_fusion_seq_decoder.pt"),
        labels=["angry", "happy", "neutral", "sad"],
        description=(
            "Single-label classifier over angry / happy / neutral / sad, trained on "
            "acted dyadic conversations (IEMOCAP, seed 7777 — best of 4 runs)."
        ),
        best_for=(
            "Controlled experiments: one participant, quiet room, close to the "
            "microphone, clearly expressed emotions. Trained on acted speech, so "
            "it rewards deliberate emotional expression — brief it to participants."
        ),
        metrics={"val_weighted_acc": 0.6891, "val_macro_f1": 0.6905},
        scenario="Quiet lab · clear expressive speech",
    ),
    # Alternate IEMOCAP seeds (2024/2025/expC) exist under runs/ but only the
    # best run (seed 7777) is exposed on the platform.
    # ── MELD · WavLM+BERT ─────────────────────────────────────────────────────
    CheckpointInfo(
        id="meld_4cls",
        display_name="MELD · 4-class emotion",
        dataset="MELD",
        task="single_label",
        feature_family="wavlm_bert",
        checkpoint_path=_run("meld_4cls_wavlm_bert/best_fusion_seq_decoder.pt"),
        labels=["angry", "happy", "neutral", "sad"],
        description=(
            "Single-label classifier over angry / happy / neutral / sad, trained on "
            "multi-party TV-series dialogue (MELD)."
        ),
        best_for=(
            "Natural human-robot interaction: short, casual, spontaneous "
            "utterances like everyday conversation, and tolerant of background "
            "noise. Closest match to how people actually talk to a robot — "
            "recommended starting point."
        ),
        metrics={"val_macro_f1": 0.6049, "offline_test_wa": 0.6612},
        recommended=True,
        scenario="Everyday conversation · noise-tolerant",
    ),
    CheckpointInfo(
        id="meld_sentiment",
        display_name="MELD · binary sentiment",
        dataset="MELD",
        task="single_label",
        feature_family="wavlm_bert",
        checkpoint_path=_run("meld_sentiment_wavlm_bert/best_fusion_seq_decoder.pt"),
        labels=["negative", "positive"],
        description=(
            "Binary sentiment classifier trained on MELD: negative vs non-negative "
            "(the 'positive' label includes neutral speech; API returns it as "
            "'positive')."
        ),
        best_for=(
            "Studies that only need to detect negative affect (frustration, "
            "distress) vs everything else. The 2-class decision is the most "
            "reliable option in this registry — prefer it when the robot's "
            "response is binary anyway."
        ),
        metrics={"val_macro_f1": 0.7108, "offline_test_wa": 0.7433},
        scenario="Everyday conversation · negative-affect detection",
    ),
    # ── MOSEI · COVAREP+GloVe ─────────────────────────────────────────────────
    CheckpointInfo(
        id="mosei_6cls",
        display_name="MOSEI · 6-emotion multi-label",
        dataset="MOSEI",
        task="multi_label",
        feature_family="covarep_glove",
        checkpoint_path=_run("mosei_6cls_v3/best_mosei_6cls.pt"),
        # Order matches training emo_cols; overridden by ckpt["emo_cols"] at load
        labels=["happy", "sad", "angry", "fear", "disgust", "surprise"],
        description=(
            "Multi-label model over happy / sad / angry / fear / disgust / surprise "
            "with per-class calibrated thresholds (CMU-MOSEI, in-the-wild YouTube)."
        ),
        best_for=(
            "Richer emotional analysis: captures intensity and co-occurring "
            "emotions (e.g. sad + angry at once). NOT recommended for live "
            "robot interaction — COVAREP runs in a MATLAB subprocess, so each "
            "utterance takes noticeably longer than the WavLM+BERT models. "
            "Ideal for post-experiment analysis of recorded participant audio, "
            "where per-utterance latency does not matter."
        ),
        metrics={"val_calibrated_macro_f1": 0.4246, "val_macro_auc": 0.6880},
        has_calibrated_thresholds=True,
        latency="~7 s per utterance",
        startup_note="first request: ~7 min one-time (MATLAB + GloVe load)",
        scenario="Recorded audio · post-session analysis",
    ),
    # MOSEI 4-class (runs/mosei_4cls) exists but is intentionally not exposed:
    # val macro-F1 0.30 — too weak for experiments.
    CheckpointInfo(
        id="mosei_sentiment",
        display_name="MOSEI · binary sentiment",
        dataset="MOSEI",
        task="single_label",
        feature_family="covarep_glove",
        checkpoint_path=_run("mosei_sentiment/best_mosei_sentiment.pt"),
        labels=["negative", "positive"],
        description=(
            "Binary sentiment classifier trained on CMU-MOSEI: negative "
            "(sentiment score < 0) vs non-negative (score ≥ 0, includes neutral; "
            "API returns it as 'positive')."
        ),
        best_for=(
            "Valence-only analysis of spontaneous, in-the-wild speech. Same "
            "COVAREP latency caveat as the 6-emotion model: better for offline "
            "analysis of recorded audio than for live interaction (use MELD "
            "sentiment there)."
        ),
        metrics={"val_acc2": 0.7937, "val_macro_f1": 0.7488},
        latency="~7 s per utterance",
        startup_note="first request: ~7 min one-time (MATLAB + GloVe load)",
        scenario="Recorded audio · negative-affect detection",
    ),
]

REGISTRY: dict[str, CheckpointInfo] = {c.id: c for c in _CHECKPOINTS}

FEATURE_FAMILY_LABELS = {
    "wavlm_bert": "WavLM (audio) + BERT (text)",
    "covarep_glove": "COVAREP (audio) + GloVe (text)",
}


def get(checkpoint_id: str) -> CheckpointInfo:
    try:
        return REGISTRY[checkpoint_id]
    except KeyError:
        raise KeyError(
            f"Unknown checkpoint id '{checkpoint_id}'. "
            f"Valid ids: {', '.join(REGISTRY)}"
        ) from None


def covarep_glove_availability() -> tuple[bool, Optional[str]]:
    """COVAREP+GloVe checkpoints additionally need MATLAB/Octave + GloVe vectors."""
    problems = []
    if not Path(COVAREP_ROOT).is_dir():
        problems.append("COVAREP repo not found (server/config.py::COVAREP_ROOT)")
    runner = Path(COVAREP_RUNNER_BIN)
    if not (runner.is_file() or COVAREP_RUNNER_BIN in ("matlab", "octave")):
        problems.append("MATLAB/Octave runner not found (COVAREP_RUNNER_BIN)")
    if not Path(GLOVE_MODEL_PATH).exists():
        problems.append("GloVe vectors not found (GLOVE_MODEL_PATH)")
    if problems:
        return False, "; ".join(problems)
    return True, None


def availability(info: CheckpointInfo) -> tuple[bool, Optional[str]]:
    """Whether this checkpoint can actually serve predictions on this machine."""
    if not Path(info.checkpoint_path).is_file():
        return False, f"Checkpoint file missing: {info.checkpoint_path}"
    if info.feature_family == "covarep_glove":
        return covarep_glove_availability()
    return True, None


def describe_all() -> list[dict[str, Any]]:
    """JSON-friendly registry dump for the platform UI."""
    out = []
    for info in _CHECKPOINTS:
        ok, reason = availability(info)
        out.append(
            {
                "id": info.id,
                "display_name": info.display_name,
                "dataset": info.dataset,
                "task": info.task,
                "feature_family": info.feature_family,
                "feature_family_label": FEATURE_FAMILY_LABELS[info.feature_family],
                "labels": list(info.labels),
                "num_classes": info.num_classes,
                "description": info.description,
                "best_for": info.best_for,
                "metrics": dict(info.metrics),
                "recommended": info.recommended,
                "variant_of": info.variant_of,
                "has_calibrated_thresholds": info.has_calibrated_thresholds,
                "latency": info.latency,
                "startup_note": info.startup_note,
                "scenario": info.scenario,
                "available": ok,
                "unavailable_reason": reason,
            }
        )
    return out
