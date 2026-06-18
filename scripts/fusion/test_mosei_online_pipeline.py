#!/usr/bin/env python3
"""
Online validation for MOSEI: run the production live-extraction pipeline
(COVAREP via MATLAB + Whisper ASR + GloVe) over downloaded test-split
segments, then evaluate BOTH the 6-class emotion model and the 2-class
sentiment model on those live features.

No retraining — reuses the already-trained offline checkpoints:
  6-class : runs/mosei_fusion_decoder_v2/best_mosei_fusion_decoder.pt
            (max_len_audio=300, max_len_text=128, calibrated thresholds)
  sentiment: runs/mosei_sentiment/best_mosei_sentiment.pt
            (max_len_audio=600, max_len_text=128)

Audio is extracted once per segment at full resolution (10s-truncated,
uncropped COVAREP output), then center-cropped separately per model to
match each checkpoint's own training max_len_audio. Per-segment live
features are cached to disk so the run is resumable.

Run from repo root:
  python scripts/fusion/test_mosei_online_pipeline.py --limit 20   # smoke test
  python scripts/fusion/test_mosei_online_pipeline.py              # full run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.mosei_fusion_with_emotion_decoder import MoseiFusionWithEmotionDecoder
from server.feature_extraction.covarep_runner import CovarepRunner
from server.feature_extraction.glove_encoder import GloveEncoder
from server.feature_extraction.whisper import WhisperTranscriber, decode_wav_bytes

CLS6_CKPT = "runs/mosei_fusion_decoder_v2/best_mosei_fusion_decoder.pt"
SENTIMENT_CKPT = "runs/mosei_sentiment/best_mosei_sentiment.pt"
OFFLINE_6CLS_TEST_MACRO_F1 = 0.4206
OFFLINE_SENTIMENT_TEST_ACC2 = 0.8069
OFFLINE_SENTIMENT_TEST_WF1 = 0.8092


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", default="data/mosei_index_splits.csv")
    ap.add_argument("--seg_dir",   default="data/mosei_raw/segments")
    ap.add_argument("--cache_dir", default="runs/mosei_online_validation/features_cache")
    ap.add_argument("--out_dir",   default="runs/mosei_online_validation")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of segments (smoke test)")
    return ap.parse_args()


def crop_center(x: np.ndarray, max_len: int) -> np.ndarray:
    if max_len is None or x.shape[0] <= max_len:
        return x
    start = (x.shape[0] - max_len) // 2
    return x[start:start + max_len]


def to_tensor_batch(arr: np.ndarray, device: torch.device):
    h = torch.from_numpy(arr).float().unsqueeze(0).to(device)        # [1, L, D]
    m = torch.zeros(1, arr.shape[0], dtype=torch.bool, device=device)  # no padding
    return h, m


def load_6cls_model(device: torch.device):
    ckpt = torch.load(CLS6_CKPT, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = MoseiFusionWithEmotionDecoder(
        d_audio=74, d_text=300, d_model=a["d_model"], num_emotions=6,
        n_heads=a["n_heads"], num_layers_fusion=a["num_layers_fusion"],
        num_layers_decoder=a["num_layers_decoder"], beta_hidden=a["beta_hidden"],
        dropout=a["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    thresholds = np.array(ckpt["val_calibrated_thresholds"], dtype=np.float32)
    return model, thresholds, ckpt["emo_cols"], int(a["max_len_audio"]), int(a["max_len_text"])


def load_sentiment_model(device: torch.device):
    ckpt = torch.load(SENTIMENT_CKPT, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = MoseiFusionWithEmotionDecoder(
        d_audio=74, d_text=300, d_model=a["d_model"], num_emotions=2,
        n_heads=a["n_heads"], num_layers_fusion=a["num_layers_fusion"],
        num_layers_decoder=a["num_layers_decoder"], beta_hidden=a["beta_hidden"],
        dropout=a["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, int(a["max_len_audio"]), int(a["max_len_text"])


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "online_validation_log.txt", "a", encoding="utf-8")

    def log(msg: object) -> None:
        safe = str(msg).encode("utf-8", errors="replace").decode("utf-8")
        print(safe, flush=True)
        log_f.write(safe + "\n")
        log_f.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("=" * 70)
    log(f"Device: {device}")

    df = pd.read_csv(args.index_csv)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    seg_dir = Path(args.seg_dir)
    available = {p.stem for p in seg_dir.glob("*.wav")}
    eval_df = test_df[test_df["uid"].isin(available)].reset_index(drop=True)
    if args.limit:
        eval_df = eval_df.iloc[:args.limit].reset_index(drop=True)
    log(f"Test segments with downloaded audio: {len(eval_df)} / {len(test_df)}")

    log("Loading 6-class model...")
    model6, thresholds6, emo_cols, maxlen_a6, maxlen_t6 = load_6cls_model(device)
    log(f"  checkpoint: {CLS6_CKPT}")
    log(f"  max_len_audio={maxlen_a6}  max_len_text={maxlen_t6}")
    log(f"  calibrated thresholds: {dict(zip([c.replace('emo_','') for c in emo_cols], thresholds6.round(3).tolist()))}")

    log("Loading sentiment model...")
    model2, maxlen_a2, maxlen_t2 = load_sentiment_model(device)
    log(f"  checkpoint: {SENTIMENT_CKPT}")
    log(f"  max_len_audio={maxlen_a2}  max_len_text={maxlen_t2}")

    covarep = CovarepRunner()
    glove   = GloveEncoder()
    whisper = WhisperTranscriber()

    rows = []
    n_failed = 0

    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Online extraction+infer"):
        uid = str(row["uid"])
        wav_path = seg_dir / f"{uid}.wav"
        cache_path = cache_dir / f"{uid}.pt"

        try:
            if cache_path.is_file():
                cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                audio_full, text_full, transcript = cached["audio"], cached["text"], cached["transcript"]
            else:
                data = wav_path.read_bytes()
                wav_np, sr = decode_wav_bytes(data, filename_hint=wav_path.name)
                audio_full = covarep.extract_frames(wav_np, sample_rate=sr, max_len=None)
                transcript = whisper.transcribe(wav_np, sample_rate=sr, word_timestamps=False)
                text_full = (glove.encode_text(transcript) if transcript.strip()
                            else np.zeros((1, 300), dtype=np.float32))
                torch.save({"audio": audio_full, "text": text_full, "transcript": transcript}, cache_path)
        except Exception as e:
            n_failed += 1
            log(f"  [SKIP] {uid}: {e}")
            continue

        a6 = crop_center(audio_full, maxlen_a6)
        t6 = crop_center(text_full,  maxlen_t6)
        h_a, m_a = to_tensor_batch(a6, device)
        h_t, m_t = to_tensor_batch(t6, device)
        with torch.no_grad():
            logits6, _, _ = model6(h_a, h_t, m_a, m_t)
            probs6 = torch.sigmoid(logits6).squeeze(0).cpu().numpy()
        pred6 = (probs6 >= thresholds6).astype(int)

        a2 = crop_center(audio_full, maxlen_a2)
        t2 = crop_center(text_full,  maxlen_t2)
        h_a2, m_a2 = to_tensor_batch(a2, device)
        h_t2, m_t2 = to_tensor_batch(t2, device)
        with torch.no_grad():
            logits2, _, _ = model2(h_a2, h_t2, m_a2, m_t2)
            pred2 = int(logits2.argmax(dim=-1).item())

        y6 = np.array([float(row[c]) for c in emo_cols], dtype=np.float32)
        y6_bin = (y6 > 0.0).astype(int)
        y2 = 1 if float(row["sentiment"]) >= 0.0 else 0

        rows.append({
            "uid": uid, "transcript": transcript,
            "pred6": pred6.tolist(), "true6": y6_bin.tolist(),
            "pred2": pred2, "true2": y2,
        })

    log(f"\nExtraction done: {len(rows)} succeeded, {n_failed} failed")

    if not rows:
        log("No segments succeeded — aborting metrics.")
        log_f.close()
        return

    # ---- 6-class metrics ----
    pred6_arr = np.array([r["pred6"] for r in rows])
    true6_arr = np.array([r["true6"] for r in rows])
    macro_f1_6 = f1_score(true6_arr, pred6_arr, average="macro", zero_division=0)
    micro_f1_6 = f1_score(true6_arr, pred6_arr, average="micro", zero_division=0)
    pc_f1_6 = f1_score(true6_arr, pred6_arr, average=None, zero_division=0)

    log("\n" + "=" * 60)
    log("6-CLASS EMOTION — ONLINE (live extraction) RESULTS")
    log(f"N={len(rows)}")
    log(f"macro-F1={macro_f1_6:.4f}  micro-F1={micro_f1_6:.4f}")
    for name, f1v in zip(emo_cols, pc_f1_6):
        log(f"  {name.replace('emo_',''):<9} F1={f1v:.4f}")
    log(f"\nOffline (pre-extracted CSD, full test set, calibrated): macro-F1={OFFLINE_6CLS_TEST_MACRO_F1:.4f}")
    log(f"Online vs offline gap: {macro_f1_6 - OFFLINE_6CLS_TEST_MACRO_F1:+.4f}")

    # ---- sentiment metrics ----
    pred2_arr = np.array([r["pred2"] for r in rows])
    true2_arr = np.array([r["true2"] for r in rows])
    acc2 = accuracy_score(true2_arr, pred2_arr)
    wf1  = f1_score(true2_arr, pred2_arr, average="weighted", zero_division=0)
    mf1  = f1_score(true2_arr, pred2_arr, average="macro",    zero_division=0)

    log("\n" + "=" * 60)
    log("2-CLASS SENTIMENT — ONLINE (live extraction) RESULTS")
    log(f"N={len(rows)}")
    log(f"Acc-2={acc2:.4f}  Weighted-F1={wf1:.4f}  Macro-F1={mf1:.4f}")
    log(f"\nOffline (pre-extracted CSD, full test set): Acc-2={OFFLINE_SENTIMENT_TEST_ACC2:.4f}  Weighted-F1={OFFLINE_SENTIMENT_TEST_WF1:.4f}")
    log(f"Online vs offline Acc-2 gap: {acc2 - OFFLINE_SENTIMENT_TEST_ACC2:+.4f}")

    with open(out_dir / "online_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    log(f"\nRaw per-segment results: {out_dir / 'online_results.json'}")

    log_f.close()


if __name__ == "__main__":
    main()
