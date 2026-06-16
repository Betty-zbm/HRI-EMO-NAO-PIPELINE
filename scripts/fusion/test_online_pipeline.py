#!/usr/bin/env python3
"""
Online pipeline validation on IEMOCAP test set.

Each utterance goes through the FULL online pipeline:
  raw WAV → Whisper ASR → WavLM (audio) + BERT (text) → 4-model ensemble

Reports per-class Precision / Recall / F1 / Support so the gap between
offline (pre-extracted features) and online (real-time extraction) is visible.

Usage:
  python scripts/fusion/test_online_pipeline.py [--whisper_size base] [--limit N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.mosei_fusion_with_emotion_decoder import MoseiFusionWithEmotionDecoder
from server.feature_extraction.iemocap import IemocapFeatureExtractor, preprocess_audio_for_wavlm
from server.feature_extraction.whisper import WhisperTranscriber

# ── 4-model selective ensemble (same as offline test) ─────────────────────────
ENSEMBLE_CKPTS = [
    ("seed=42",   "runs/iemocap_4cls_expC/best_fusion_seq_decoder.pt"),
    ("seed=2024", "runs/iemocap_4cls_seed2024/best_fusion_seq_decoder.pt"),
    ("seed=7777", "runs/iemocap_4cls_seed7777/best_fusion_seq_decoder.pt"),
    ("seed=2025", "runs/iemocap_4cls_seed2025/best_fusion_seq_decoder.pt"),
]

IEMOCAP_RAW = Path("data/iemocap/IEMOCAP_full_release")
TARGET_SR = 16_000


def find_wav(utter_id: str) -> Path | None:
    """
    Map an utter_id (e.g. Ses05F_impro01_F000) to its per-utterance WAV file.
    Path pattern: Session{N}/sentences/wav/{dialog}/{utter_id}.wav
    """
    # Extract session number from the first 5 chars: "Ses05"
    session_num = int(utter_id[3:5])
    # Dialog name = everything up to the last underscore + speaker tag
    # e.g. Ses05F_impro01_F000 → dialog = Ses05F_impro01
    parts = utter_id.rsplit("_", 1)
    dialog = parts[0]
    wav = IEMOCAP_RAW / f"Session{session_num}" / "sentences" / "wav" / dialog / f"{utter_id}.wav"
    return wav if wav.is_file() else None


def load_wav(path: Path) -> np.ndarray:
    """Load WAV, convert to mono float32 at 16 kHz, peak-normalize."""
    import librosa
    wav_data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = wav_data.mean(axis=1)
    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    peak = float(np.abs(wav).max())
    if peak > 0:
        wav = wav / peak
    return wav.astype(np.float32)


def load_model(path: str, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    label2id = ckpt["label2id"]
    m = MoseiFusionWithEmotionDecoder(
        d_audio=768, d_text=768, d_model=args["d_model"],
        num_emotions=len(label2id), n_heads=args["n_heads"],
        num_layers_fusion=args["num_layers_fusion"],
        num_layers_decoder=args["num_layers_decoder"],
        beta_hidden=args["beta_hidden"], dropout=args["dropout"],
    ).to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m, args, label2id, ckpt.get("epoch", "?")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper_size", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size for ASR")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N test utterances (for quick testing)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load ensemble models ─────────────────────────────────────────────────
    print("\nLoading ensemble models...")
    models_data, label2id_ref = [], None
    for tag, path in ENSEMBLE_CKPTS:
        m, m_args, label2id, ep = load_model(path, device)
        if label2id_ref is None:
            label2id_ref = label2id
        print(f"  {tag}  epoch={ep}  d_model={m_args['d_model']}  "
              f"max_a={m_args.get('max_len_audio',300)}  max_t={m_args.get('max_len_text',128)}")
        models_data.append((m, m_args, label2id))

    id2label = {v: k for k, v in label2id_ref.items()}
    names = [id2label[i] for i in sorted(id2label)]

    # ── Load online pipeline components ─────────────────────────────────────
    print(f"\nLoading Whisper ({args.whisper_size})...")
    transcriber = WhisperTranscriber(model_size=args.whisper_size)

    print("Loading WavLM + BERT feature extractor...")
    feat_extractor = IemocapFeatureExtractor(device=str(device))

    # ── Load test utterances ─────────────────────────────────────────────────
    df = pd.read_csv("data/iemocap_index_splits.csv")
    df["label"] = df["label"].map(lambda x: "happy" if x == "excited" else x)
    test_df = df[
        (df["split"] == "test") & (~df["label"].isin(["frustration"]))
    ].reset_index(drop=True)

    if args.limit:
        test_df = test_df.head(args.limit)

    print(f"\nTest set: {len(test_df)} utterances")
    print(test_df["label"].value_counts().to_string())
    print()

    # ── Run online pipeline ──────────────────────────────────────────────────
    all_preds, all_labels = [], []
    skipped, no_wav = 0, 0

    for idx, row in test_df.iterrows():
        uid = str(row["utter_id"])
        wav_path = find_wav(uid)
        if wav_path is None:
            no_wav += 1
            continue

        # 1. Load raw audio
        wav_np = load_wav(wav_path)

        # 2. Whisper ASR
        transcript = transcriber.transcribe(wav_np, sample_rate=TARGET_SR)
        if not isinstance(transcript, str):
            transcript = transcript.text  # WhisperTranscript object

        # 3. Online feature extraction (WavLM + BERT)
        audio_feats, text_feats = feat_extractor.extract(wav_np, transcript)

        # 4. Ensemble inference with truncation matching training
        probs_sum = None
        skip = False
        for m, m_args, l2id in models_data:
            max_a = m_args.get("max_len_audio", 300)
            max_t = m_args.get("max_len_text", 128)
            h_a = audio_feats.hidden.clone()
            m_a = audio_feats.key_padding_mask.clone()
            h_t = text_feats.hidden.clone()
            m_t = text_feats.key_padding_mask.clone()
            if h_a.size(1) > max_a:
                h_a, m_a = h_a[:, :max_a, :], m_a[:, :max_a]
            if h_t.size(1) > max_t:
                h_t, m_t = h_t[:, :max_t, :], m_t[:, :max_t]
            h_a = h_a.to(device); m_a = m_a.to(device)
            h_t = h_t.to(device); m_t = m_t.to(device)
            with torch.no_grad(), autocast("cuda"):
                logits, _, _ = m(h_a, h_t, m_a, m_t)
            probs = F.softmax(logits, dim=-1)
            probs_sum = probs if probs_sum is None else probs_sum + probs

        if skip:
            skipped += 1
            continue

        pred_idx = probs_sum.argmax(-1).item()
        all_preds.append(pred_idx)
        all_labels.append(label2id_ref[row["label"]])

        if (len(all_preds)) % 50 == 0:
            print(f"  [{len(all_preds)}/{len(test_df)}] done  "
                  f"(last transcript: \"{transcript[:60]}\")")

    print(f"\nProcessed: {len(all_preds)}  |  No WAV: {no_wav}  |  Skipped: {skipped}")

    # ── Metrics ──────────────────────────────────────────────────────────────
    wa  = accuracy_score(all_labels, all_preds)
    ua  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    mf1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"\n{'='*60}")
    print(f"ONLINE PIPELINE — 4-seed ensemble (Whisper-{args.whisper_size})")
    print(f"{'='*60}")
    print(f"WA={wa:.4f}  UA={ua:.4f}  macro-F1={mf1:.4f}  wt-F1={wf1:.4f}")
    print()
    print(classification_report(all_labels, all_preds, target_names=names, zero_division=0))

    print("\nComparison:")
    print(f"  Offline (pre-extracted)  WA=0.7004  UA=0.6375  macro-F1=0.6324")
    print(f"  Online  (this run)       WA={wa:.4f}  UA={ua:.4f}  macro-F1={mf1:.4f}")
    delta = wa - 0.7004
    print(f"  Gap (online - offline):  WA={delta:+.4f}")


if __name__ == "__main__":
    main()
