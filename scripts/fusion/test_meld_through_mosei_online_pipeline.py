#!/usr/bin/env python3
"""
Cross-corpus test: run MELD test-set clips through the MOSEI online
(COVAREP + Whisper + GloVe) pipeline and evaluate with the MOSEI
checkpoints (6-class emotion + 2-class sentiment) — no retraining.

Label mapping:
  6-class emotion (MELD 7-class -> MOSEI 6-class multi-label):
    joy->happy, sadness->sad, anger->anger, fear->fear,
    disgust->disgust, surprise->surprise, neutral->all six = 0

  2-class sentiment (MELD 3-class -> MOSEI 2-class):
    MOSEI training used score>=0 -> positive, so "positive" really means
    "non-negative". MELD positive + neutral -> label 1, MELD negative -> 0.
    All rows are used (none dropped).

Run from repo root:
  python scripts/fusion/test_meld_through_mosei_online_pipeline.py --limit 800
"""
from __future__ import annotations

import argparse
import json
import subprocess
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
FFMPEG = r"C:\Users\HuRoN Lab\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"

MELD_TO_MOSEI_EMO = {
    "joy": "happy", "sadness": "sad", "anger": "anger",
    "fear": "fear", "disgust": "disgust", "surprise": "surprise",
}
MOSEI_EMO_COLS = ["emo_happy", "emo_sad", "emo_anger", "emo_fear", "emo_disgust", "emo_surprise"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", default="data/MELD.Raw/test_sent_emo.csv")
    ap.add_argument("--video_dir",  default="data/MELD.Raw/output_repeated_splits_test")
    ap.add_argument("--wav_dir",    default="data/meld_raw/segments_test")
    ap.add_argument("--cache_dir",  default="runs/meld_through_mosei/features_cache")
    ap.add_argument("--out_dir",    default="runs/meld_through_mosei")
    ap.add_argument("--limit", type=int, default=800, help="Subsample size (0 = all)")
    ap.add_argument("--seed", type=int, default=1234)
    return ap.parse_args()


def crop_center(x: np.ndarray, max_len: int) -> np.ndarray:
    if max_len is None or x.shape[0] <= max_len:
        return x
    start = (x.shape[0] - max_len) // 2
    return x[start:start + max_len]


def to_tensor_batch(arr: np.ndarray, device: torch.device):
    h = torch.from_numpy(arr).float().unsqueeze(0).to(device)
    m = torch.zeros(1, arr.shape[0], dtype=torch.bool, device=device)
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
    return model, thresholds, int(a["max_len_audio"]), int(a["max_len_text"])


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


def extract_wav(video_path: Path, wav_path: Path) -> bool:
    if wav_path.is_file():
        return True
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(video_path), "-ac", "1", "-ar", "16000", str(wav_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0 and wav_path.is_file()


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["uid"] = df.apply(lambda r: f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}", axis=1)

    for mosei_name in ["happy", "sad", "anger", "fear", "disgust", "surprise"]:
        col = f"emo_{mosei_name}"
        df[col] = 0.0
    for meld_emo, mosei_name in MELD_TO_MOSEI_EMO.items():
        mask = df["Emotion"] == meld_emo
        df.loc[mask, f"emo_{mosei_name}"] = 1.0

    # sentiment: positive + neutral -> 1 ("non-negative", matches MOSEI's score>=0 boundary)
    df["sentiment_label"] = (df["Sentiment"] != "negative").astype(int)
    return df


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = Path(args.wav_dir)
    log_f = open(out_dir / "meld_online_log.txt", "a", encoding="utf-8")

    def log(msg: object) -> None:
        safe = str(msg).encode("utf-8", errors="replace").decode("utf-8")
        print(safe, flush=True)
        log_f.write(safe + "\n")
        log_f.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("=" * 70)
    log(f"Device: {device}")

    raw_df = pd.read_csv(args.labels_csv)
    df = build_labels(raw_df)
    log(f"MELD test labeled rows: {len(df)}")
    log(f"  Emotion dist: {raw_df['Emotion'].value_counts().to_dict()}")
    log(f"  Sentiment dist: {raw_df['Sentiment'].value_counts().to_dict()}  "
        f"-> mapped non-negative(1)={int(df['sentiment_label'].sum())}  negative(0)={int((1-df['sentiment_label']).sum())}")

    if args.limit and args.limit > 0 and len(df) > args.limit:
        df = df.sample(n=args.limit, random_state=args.seed).reset_index(drop=True)
    log(f"Evaluating subsample: {len(df)}")

    log("Loading 6-class model...")
    model6, thresholds6, maxlen_a6, maxlen_t6 = load_6cls_model(device)
    log(f"  max_len_audio={maxlen_a6}  max_len_text={maxlen_t6}")

    log("Loading sentiment model...")
    model2, maxlen_a2, maxlen_t2 = load_sentiment_model(device)
    log(f"  max_len_audio={maxlen_a2}  max_len_text={maxlen_t2}")

    covarep = CovarepRunner()
    glove   = GloveEncoder()
    whisper = WhisperTranscriber()

    video_dir = Path(args.video_dir)
    rows = []
    n_failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="MELD online extraction+infer"):
        uid = row["uid"]
        video_path = video_dir / f"{uid}.mp4"
        wav_path = wav_dir / f"{uid}.wav"
        cache_path = cache_dir / f"{uid}.pt"

        try:
            if not extract_wav(video_path, wav_path):
                raise RuntimeError("ffmpeg extraction failed")

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

        y6 = np.array([float(row[c]) for c in MOSEI_EMO_COLS], dtype=np.float32)
        y2 = int(row["sentiment_label"])

        rows.append({
            "uid": uid, "transcript": transcript, "meld_emotion": row["Emotion"], "meld_sentiment": row["Sentiment"],
            "pred6": pred6.tolist(), "true6": y6.astype(int).tolist(),
            "pred2": pred2, "true2": y2,
        })

    log(f"\nExtraction done: {len(rows)} succeeded, {n_failed} failed")

    if not rows:
        log("No segments succeeded - aborting metrics.")
        log_f.close()
        return

    pred6_arr = np.array([r["pred6"] for r in rows])
    true6_arr = np.array([r["true6"] for r in rows])
    macro_f1_6 = f1_score(true6_arr, pred6_arr, average="macro", zero_division=0)
    micro_f1_6 = f1_score(true6_arr, pred6_arr, average="micro", zero_division=0)
    pc_f1_6 = f1_score(true6_arr, pred6_arr, average=None, zero_division=0)

    log("\n" + "=" * 60)
    log("MELD -> MOSEI 6-CLASS EMOTION (cross-corpus, online pipeline)")
    log(f"N={len(rows)}")
    log(f"macro-F1={macro_f1_6:.4f}  micro-F1={micro_f1_6:.4f}")
    for name, f1v in zip(MOSEI_EMO_COLS, pc_f1_6):
        log(f"  {name.replace('emo_',''):<9} F1={f1v:.4f}")

    pred2_arr = np.array([r["pred2"] for r in rows])
    true2_arr = np.array([r["true2"] for r in rows])
    acc2 = accuracy_score(true2_arr, pred2_arr)
    wf1  = f1_score(true2_arr, pred2_arr, average="weighted", zero_division=0)
    mf1  = f1_score(true2_arr, pred2_arr, average="macro",    zero_division=0)

    log("\n" + "=" * 60)
    log("MELD -> MOSEI 2-CLASS SENTIMENT (cross-corpus, online pipeline)")
    log(f"N={len(rows)}")
    log(f"Acc-2={acc2:.4f}  Weighted-F1={wf1:.4f}  Macro-F1={mf1:.4f}")

    with open(out_dir / "meld_online_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    log(f"\nRaw per-segment results: {out_dir / 'meld_online_results.json'}")

    log_f.close()


if __name__ == "__main__":
    main()
