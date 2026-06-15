#!/usr/bin/env python3
"""
End-to-end MOSEI evaluation through the online server pipeline.

For each WAV clip:
  Whisper -> COVAREP + GloVe -> fusion checkpoint -> multilabel prediction

Compares against labels in ``data/mosei_index_splits.csv`` (emo_* > 0).

Example (10 validation clips):
  PYTHONPATH=. python scripts/evaluate_mosei_online_e2e.py

Optional offline baseline on the same uids (CSD .pt features, same checkpoint):
  PYTHONPATH=. python scripts/evaluate_mosei_online_e2e.py --offline-baseline
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.config import MOSEI_LABELS, ROOT_DIR
from server.emotion_inference.decoder_wrapper import EmotionDecoderWrapper
from server.emotion_service import EmotionService
from server.feature_extraction.iemocap import SeqFeatures, pack_seq_features

EMO_COLS = [
    "emo_happy",
    "emo_sad",
    "emo_anger",
    "emo_fear",
    "emo_disgust",
    "emo_surprise",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MOSEI online pipeline e2e evaluation")
    ap.add_argument(
        "--manifest",
        type=str,
        default="data/online_mosei_validation/manifest.csv",
        help="CSV with uid and clip_path columns",
    )
    ap.add_argument(
        "--index-csv",
        type=str,
        default="data/mosei_index_splits.csv",
        help="MOSEI labels and splits",
    )
    ap.add_argument(
        "--clips-dir",
        type=str,
        default="",
        help="Optional directory of {uid}.wav (overrides manifest clip_path)",
    )
    ap.add_argument(
        "--features-root",
        type=str,
        default="features/mosei/seq_level",
        help="Offline CSD features for --offline-baseline",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default="data/online_mosei_validation/online_e2e_results.csv",
    )
    ap.add_argument(
        "--offline-baseline",
        action="store_true",
        help="Also run fusion on offline .pt features for the same uids",
    )
    ap.add_argument(
        "--max-clips",
        type=int,
        default=0,
        help="Limit number of clips (0 = all in manifest/clips-dir)",
    )
    return ap.parse_args()


def _resolve_repo_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    return p if p.is_absolute() else ROOT_DIR / p


def _load_label_row(index_df: pd.DataFrame, uid: str) -> pd.Series | None:
    rows = index_df[index_df["uid"] == uid]
    if rows.empty:
        return None
    return rows.iloc[0]


def _binary_labels(row: pd.Series) -> np.ndarray:
    return np.array([(row[c] > 0) for c in EMO_COLS], dtype=int)


def _prediction_to_vectors(result: dict) -> tuple[np.ndarray, np.ndarray]:
    scores = result.get("scores") or {}
    probs = np.array([float(scores.get(label, 0.0)) for label in MOSEI_LABELS], dtype=float)
    predicted = result.get("predicted") or []
    pred_bin = np.array([1 if label in predicted else 0 for label in MOSEI_LABELS], dtype=int)
    return probs, pred_bin


def _load_offline_features(uid: str, features_root: Path) -> tuple[SeqFeatures, SeqFeatures] | None:
    audio_pt = features_root / "audio" / f"{uid}.pt"
    text_pt = features_root / "text" / f"{uid}.pt"
    if not audio_pt.is_file() or not text_pt.is_file():
        return None

    def _to_seq(obj) -> SeqFeatures:
        if isinstance(obj, dict):
            hidden = obj["hidden"]
            mask = obj.get("attention_mask")
        else:
            hidden = obj
            mask = torch.ones(hidden.size(0), dtype=torch.long)
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.float()
        else:
            hidden = torch.from_numpy(np.asarray(hidden)).float()
        if mask is None:
            mask = torch.ones(hidden.size(0), dtype=torch.long)
        elif not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.asarray(mask)).long()
        return pack_seq_features(hidden, mask)

    audio_obj = torch.load(audio_pt, map_location="cpu")
    text_obj = torch.load(text_pt, map_location="cpu")
    return _to_seq(audio_obj), _to_seq(text_obj)


def _aggregate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    n, k = y_true.shape
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    subset_acc = float(np.mean(np.all(y_true == y_pred, axis=1)))

    aucs = []
    for i in range(k):
        col = y_true[:, i]
        if col.max() > 0 and col.min() < 1:
            aucs.append(roc_auc_score(col, y_prob[:, i]))
    macro_auc = float(np.mean(aucs)) if aucs else float("nan")

    return {
        "n_clips": n,
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "macro_auc": macro_auc,
        "subset_accuracy": subset_acc,
    }


def _print_metrics(title: str, metrics: dict) -> None:
    print(f"\n=== {title} ({metrics['n_clips']} clips) ===")
    print(f"  micro-F1:  {metrics['micro_f1']:.4f}")
    print(f"  macro-F1:  {metrics['macro_f1']:.4f}")
    print(f"  macro-AUC: {metrics['macro_auc']:.4f}")
    print(f"  subset accuracy (exact label set match): {metrics['subset_accuracy']:.4f}")


def _iter_clips(args: argparse.Namespace) -> list[tuple[str, Path]]:
    clips: list[tuple[str, Path]] = []
    clips_dir = _resolve_repo_path(args.clips_dir) if args.clips_dir else None

    if clips_dir and clips_dir.is_dir():
        for wav in sorted(clips_dir.glob("*.wav")):
            clips.append((wav.stem, wav))
    else:
        manifest = _resolve_repo_path(args.manifest)
        if not manifest.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        df = pd.read_csv(manifest)
        base = manifest.parent
        for _, row in df.iterrows():
            uid = str(row["uid"])
            clip_rel = row.get("clip_path") or f"test_clips/{uid}.wav"
            wav = base / clip_rel
            if wav.is_file():
                clips.append((uid, wav))

    if args.max_clips and args.max_clips > 0:
        clips = clips[: args.max_clips]
    return clips


def main() -> None:
    args = parse_args()
    index_df = pd.read_csv(_resolve_repo_path(args.index_csv))
    clips = _iter_clips(args)
    if not clips:
        raise SystemExit("No WAV clips found.")

    out_path = _resolve_repo_path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features_root = _resolve_repo_path(args.features_root)

    service = EmotionService()
    decoder = service.decoder
    decoder._ensure_mosei()

    rows: list[dict] = []
    y_true_all: list[np.ndarray] = []
    y_pred_online: list[np.ndarray] = []
    y_prob_online: list[np.ndarray] = []
    y_pred_offline: list[np.ndarray] = []
    y_prob_offline: list[np.ndarray] = []

    print(f"[info] evaluating {len(clips)} clip(s) through online MOSEI pipeline")
    print(f"[info] checkpoint loaded: {decoder._mosei_weights_loaded}")
    print(f"[info] thresholds: {decoder._mosei_threshold_source} {decoder._mosei_thresholds}")

    for idx, (uid, wav_path) in enumerate(clips, start=1):
        label_row = _load_label_row(index_df, uid)
        if label_row is None:
            print(f"[skip] {uid}: no labels in index csv")
            continue

        y_true = _binary_labels(label_row)
        gt_labels = [MOSEI_LABELS[i] for i, v in enumerate(y_true) if v]

        t0 = time.time()
        try:
            online = service.run(wav_path.read_bytes(), benchmark="mosei", filename_hint=wav_path.name)
        except Exception as exc:
            print(f"[error] {uid}: {exc}", file=sys.stderr)
            continue
        elapsed = time.time() - t0

        prob_on, pred_on = _prediction_to_vectors(online)
        y_true_all.append(y_true)
        y_pred_online.append(pred_on)
        y_prob_online.append(prob_on)

        row: dict = {
            "uid": uid,
            "wav": str(wav_path),
            "elapsed_sec": round(elapsed, 2),
            "transcript": online.get("transcript", ""),
            "ground_truth": "|".join(gt_labels) if gt_labels else "",
            "online_predicted": "|".join(online.get("predicted") or []),
            "online_exact_match": int(np.array_equal(y_true, pred_on)),
        }
        for label in MOSEI_LABELS:
            row[f"gt_{label}"] = int(y_true[MOSEI_LABELS.index(label)])
            row[f"prob_{label}"] = round(float(online["scores"][label]), 4)
            row[f"pred_{label}"] = int(pred_on[MOSEI_LABELS.index(label)])

        if args.offline_baseline:
            feats = _load_offline_features(uid, features_root)
            if feats is None:
                row["offline_predicted"] = ""
                row["offline_exact_match"] = ""
            else:
                audio_feats, text_feats = feats
                off_pred = decoder.predict_mosei(
                    audio_feats,
                    text_feats,
                    transcript=str(label_row.get("text", "")),
                    placeholder_features=False,
                )
                prob_off, pred_off = _prediction_to_vectors(
                    {
                        "scores": off_pred.scores,
                        "predicted": off_pred.predicted,
                    }
                )
                y_pred_offline.append(pred_off)
                y_prob_offline.append(prob_off)
                row["offline_predicted"] = "|".join(off_pred.predicted)
                row["offline_exact_match"] = int(np.array_equal(y_true, pred_off))

        rows.append(row)
        print(
            f"[{idx}/{len(clips)}] {uid} ({elapsed:.1f}s) "
            f"gt={row['ground_truth'] or '-'} "
            f"online={row['online_predicted'] or '-'}"
            + (f" offline={row.get('offline_predicted', '')}" if args.offline_baseline else "")
        )

    if not rows:
        raise SystemExit("No successful evaluations.")

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n[saved] per-clip results -> {out_path}")

    y_true_mat = np.stack(y_true_all)
    y_pred_on_mat = np.stack(y_pred_online)
    y_prob_on_mat = np.stack(y_prob_online)
    _print_metrics("Online pipeline (Whisper + COVAREP + GloVe)", _aggregate_metrics(
        y_true_mat, y_pred_on_mat, y_prob_on_mat
    ))

    if args.offline_baseline and y_pred_offline:
        y_pred_off_mat = np.stack(y_pred_offline)
        y_prob_off_mat = np.stack(y_prob_offline)
        _print_metrics(
            "Offline features baseline (CSD .pt, same checkpoint)",
            _aggregate_metrics(y_true_mat[: len(y_pred_offline)], y_pred_off_mat, y_prob_off_mat),
        )

    summary = {
        "n_clips": len(rows),
        "online": _aggregate_metrics(y_true_mat, y_pred_on_mat, y_prob_on_mat),
    }
    if args.offline_baseline and y_pred_offline:
        summary["offline_baseline"] = _aggregate_metrics(
            y_true_mat[: len(y_pred_offline)], y_pred_off_mat, y_prob_off_mat
        )
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[saved] summary -> {summary_path}")


if __name__ == "__main__":
    main()
