#!/usr/bin/env python3
"""
Cross-corpus online pipeline validation on MELD test set.

MELD audio (Friends TV show) → Whisper ASR → WavLM + BERT → IEMOCAP 4-model ensemble.

Label mapping (MELD → IEMOCAP 4-class):
  anger   → angry
  joy     → happy
  neutral → neutral
  sadness → sad
  surprise / disgust / fear → excluded

Usage:
  python scripts/fusion/test_meld_online_pipeline.py [--whisper_size base] [--limit N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import librosa
import numpy as np
import pandas as pd
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

# ── Paths ─────────────────────────────────────────────────────────────────────
MELD_CSV  = Path("data/MELD.Raw/test_sent_emo.csv")
MELD_AUDIO = Path("data/MELD.Raw/output_repeated_splits_test")

# ── Label mapping ──────────────────────────────────────────────────────────────
MELD_TO_4CLS = {
    "anger":   "angry",
    "joy":     "happy",
    "neutral": "neutral",
    "sadness": "sad",
}

# ── Ensemble checkpoints (same as IEMOCAP online test) ────────────────────────
ENSEMBLE_CKPTS = [
    ("seed=42",   "runs/iemocap_4cls_expC/best_fusion_seq_decoder.pt"),
    ("seed=2024", "runs/iemocap_4cls_seed2024/best_fusion_seq_decoder.pt"),
    ("seed=7777", "runs/iemocap_4cls_seed7777/best_fusion_seq_decoder.pt"),
    ("seed=2025", "runs/iemocap_4cls_seed2025/best_fusion_seq_decoder.pt"),
]

TARGET_SR = 16_000


def load_mp4_audio(path: Path) -> np.ndarray:
    """Extract mono float32 waveform at 16 kHz from an mp4 file via PyAV."""
    container = av.open(str(path))
    sr_native = container.streams.audio[0].rate
    frames = [f.to_ndarray() for f in container.decode(audio=0)]
    container.close()
    wav = np.concatenate(frames, axis=1).mean(axis=0).astype(np.float32)
    if sr_native != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr_native, target_sr=TARGET_SR)
    peak = float(np.abs(wav).max())
    if peak > 0:
        wav = wav / peak
    return wav.astype(np.float32)


def find_mp4(dialogue_id: int, utterance_id: int) -> Path | None:
    p = MELD_AUDIO / f"dia{dialogue_id}_utt{utterance_id}.mp4"
    return p if p.is_file() else None


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
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N 4-class utterances (quick test)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load ensemble ────────────────────────────────────────────────────────
    print("\nLoading ensemble models...")
    models_data, label2id_ref = [], None
    for tag, path in ENSEMBLE_CKPTS:
        m, m_args, label2id, ep = load_model(path, device)
        if label2id_ref is None:
            label2id_ref = label2id
        print(f"  {tag}  epoch={ep}  d_model={m_args['d_model']}")
        models_data.append((m, m_args, label2id))

    id2label = {v: k for k, v in label2id_ref.items()}
    names = [id2label[i] for i in sorted(id2label)]

    # ── Online pipeline components ───────────────────────────────────────────
    print(f"\nLoading Whisper ({args.whisper_size})...")
    transcriber = WhisperTranscriber(model_size=args.whisper_size)

    print("Loading WavLM + BERT feature extractor...")
    feat_extractor = IemocapFeatureExtractor(device=str(device))

    # ── Load and filter MELD test CSV ───────────────────────────────────────
    df = pd.read_csv(MELD_CSV, encoding="utf-8", encoding_errors="replace")
    df4 = df[df["Emotion"].isin(MELD_TO_4CLS)].copy().reset_index(drop=True)
    df4["label_4cls"] = df4["Emotion"].map(MELD_TO_4CLS)

    if args.limit:
        df4 = df4.head(args.limit)

    print(f"\nMELD test 4-class: {len(df4)} utterances")
    print(df4["label_4cls"].value_counts().to_string())
    print(f"\nDropped (not in 4-class): {len(df) - len(df4)} utterances "
          f"({df[~df['Emotion'].isin(MELD_TO_4CLS)]['Emotion'].value_counts().to_dict()})")
    print()

    # ── Run online pipeline ──────────────────────────────────────────────────
    all_preds, all_labels = [], []
    no_mp4, errors = 0, 0

    for idx, row in df4.iterrows():
        dia_id = int(row["Dialogue_ID"])
        utt_id = int(row["Utterance_ID"])
        mp4_path = find_mp4(dia_id, utt_id)

        if mp4_path is None:
            no_mp4 += 1
            continue

        try:
            # 1. Load mp4 audio
            wav_np = load_mp4_audio(mp4_path)

            # 2. Whisper ASR
            transcript = transcriber.transcribe(wav_np, sample_rate=TARGET_SR)
            if not isinstance(transcript, str):
                transcript = transcript.text

            # 3. Online feature extraction (WavLM + BERT)
            audio_feats, text_feats = feat_extractor.extract(wav_np, transcript)

            # 4. Ensemble inference with training-matched truncation
            probs_sum = None
            for model, m_args, _ in models_data:
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
                    logits, _, _ = model(h_a, h_t, m_a, m_t)
                probs = F.softmax(logits, dim=-1)
                probs_sum = probs if probs_sum is None else probs_sum + probs

            all_preds.append(probs_sum.argmax(-1).item())
            all_labels.append(label2id_ref[row["label_4cls"]])

        except Exception as e:
            errors += 1
            continue

        n = len(all_preds)
        if n % 100 == 0:
            print(f"  [{n}/{len(df4)}] done  "
                  f"(last: \"{str(transcript)[:55]}\")")

    print(f"\nProcessed: {len(all_preds)}  |  No MP4: {no_mp4}  |  Errors: {errors}")

    # ── Metrics ──────────────────────────────────────────────────────────────
    wa  = accuracy_score(all_labels, all_preds)
    ua  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    mf1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"\n{'='*60}")
    print(f"MELD cross-corpus — 4-seed ensemble (Whisper-{args.whisper_size})")
    print(f"{'='*60}")
    print(f"WA={wa:.4f}  UA={ua:.4f}  macro-F1={mf1:.4f}  wt-F1={wf1:.4f}")
    print()
    print(classification_report(all_labels, all_preds, target_names=names, zero_division=0))

    print("\nContext (same model, same pipeline):")
    print(f"  IEMOCAP test  (in-corpus,  pre-extracted)  WA=0.7004  macro-F1=0.6324")
    print(f"  IEMOCAP test  (in-corpus,  online pipeline) WA=0.6949  macro-F1=0.6195")
    print(f"  MELD test     (cross-corpus, online pipeline) WA={wa:.4f}  macro-F1={mf1:.4f}")


if __name__ == "__main__":
    main()
