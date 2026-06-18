#!/usr/bin/env python3
"""
Overnight pipeline: MOSEI WavLM+BERT feature extraction + training.

Steps:
  1. Download CMU_MOSEI_Labels.csd (segment timestamps)
  2. Download ~250 YouTube videos + cut 22k segments to WAV
  3. Extract WavLM audio features (768-dim)
  4. Transcribe with Whisper + encode with BERT (768-dim text)
  5. Train 6-class multi-label model

All steps are resumable — already-completed files are skipped.
Log: runs/mosei_wavlm_bert_pipeline.log
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO   = Path(__file__).parent
PYTHON = r"C:\Users\HuRoN Lab\Desktop\HRI-EMO-main\.venv\Scripts\python.exe"
YTDLP  = r"C:\Users\HuRoN Lab\Desktop\HRI-EMO-main\.venv\Scripts\yt-dlp.exe"
FFMPEG = None  # resolved from PATH below

DATA_ROOT    = REPO / "data"
MOSEI_DIR    = DATA_ROOT / "mosei_raw"           # YouTube videos + cut segments
FEAT_ROOT    = DATA_ROOT / "mosei_wavlm_bert"
AUDIO_FEAT   = FEAT_ROOT / "features" / "audio"
TEXT_FEAT    = FEAT_ROOT / "features" / "text"
CSD_DIR      = DATA_ROOT / "MOSEI"
CSD_LABELS   = CSD_DIR / "CMU_MOSEI_Labels.csd"
INDEX_CSV    = DATA_ROOT / "mosei_index_splits.csv"
OUT_DIR      = REPO / "runs" / "mosei_6cls_wavlm_bert"
LOG_PATH     = REPO / "runs" / "mosei_wavlm_bert_pipeline.log"

# ── Helpers ───────────────────────────────────────────────────────────────────
log_f = None

def log(msg: str, *, also_print: bool = True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if also_print:
        print(line, flush=True)
    if log_f:
        log_f.write(line + "\n")
        log_f.flush()


def run(cmd: list, step: str, check: bool = True) -> int:
    log(f"CMD: {' '.join(str(c) for c in cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        cwd=str(REPO),
    )
    for line in proc.stdout:
        line = line.rstrip()
        safe = line.encode("utf-8", errors="replace").decode("utf-8")
        print(safe, flush=True)
        if log_f:
            log_f.write(safe + "\n")
            log_f.flush()
    proc.wait()
    if check and proc.returncode != 0:
        log(f"STEP FAILED: {step}  exit={proc.returncode}")
        raise RuntimeError(f"{step} failed with exit code {proc.returncode}")
    return proc.returncode


def find_ffmpeg() -> str:
    import shutil
    # Try PATH first (winget installs here after shell restart, but may need full path)
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    # Winget default install location
    candidates = list(Path(r"C:\Users\HuRoN Lab\AppData\Local\Microsoft\WinGet\Packages").glob(
        "Gyan.FFmpeg*/**/ffmpeg.exe"))
    if candidates:
        return str(candidates[0])
    raise SystemExit("ffmpeg not found. Run: winget install Gyan.FFmpeg")


# ── Step 1: Download CMU_MOSEI_Labels.csd ────────────────────────────────────
CSD_URLS = [
    "https://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/labels/CMU_MOSEI_Labels.csd",
    "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/labels/CMU_MOSEI_Labels.csd",
]
CSD_RETRY_INTERVALS = [0, 15*60, 30*60, 60*60, 90*60]  # seconds between retries


def _try_download_csd() -> bool:
    """Try each URL once. Returns True if downloaded successfully."""
    import urllib.request, shutil, ssl
    CSD_DIR.mkdir(parents=True, exist_ok=True)
    dest = str(CSD_LABELS)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    for url in CSD_URLS:
        log(f"  Trying {url} ...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120, context=ctx) as r, \
                 open(dest + ".tmp", "wb") as f:
                shutil.copyfileobj(r, f)
            import os; os.replace(dest + ".tmp", dest)
            log(f"  Downloaded OK from {url}")
            return True
        except Exception as e:
            log(f"  Failed ({type(e).__name__}: {e})")
    return False


def step1_download_csd():
    if CSD_LABELS.exists():
        log(f"STEP 1 SKIP — {CSD_LABELS} already exists")
        return
    log("STEP 1: Downloading CMU_MOSEI_Labels.csd ...")
    for attempt, wait in enumerate(CSD_RETRY_INTERVALS):
        if wait > 0:
            log(f"  Retrying in {wait//60} min (attempt {attempt+1}/{len(CSD_RETRY_INTERVALS)}) ...")
            time.sleep(wait)
        if _try_download_csd():
            log(f"STEP 1 DONE — {CSD_LABELS} ({CSD_LABELS.stat().st_size/1e6:.1f} MB)")
            return
    log("STEP 1 FAILED — CMU server unreachable after all retries.")
    log("  Manual fix: download CMU_MOSEI_Labels.csd from the CMU-MultimodalSDK")
    log("  dataset page and place it at:  data/MOSEI/CMU_MOSEI_Labels.csd")
    log("  Then re-run this script — it will resume from Step 2.")
    raise RuntimeError("CMU_MOSEI_Labels.csd could not be downloaded — see log for instructions.")


# ── Step 2: Download YouTube videos + cut segments ────────────────────────────
STEP2_MARKER = MOSEI_DIR / ".step2_done"

def step2_download_and_cut():
    seg_dir = MOSEI_DIR / "segments"
    if STEP2_MARKER.exists():
        existing = len(list(seg_dir.glob("*.wav"))) if seg_dir.exists() else 0
        log(f"STEP 2 SKIP — already completed ({existing} segments on disk)")
        return
    existing = len(list(seg_dir.glob("*.wav"))) if seg_dir.exists() else 0
    log(f"STEP 2: Download & cut segments ({existing} on disk so far) ...")
    run([
        PYTHON,
        "scripts/mosei_feature_extraction_seq_level/download_and_cut_mosei_audio.py",
        "--csd",        str(CSD_LABELS),
        "--index_csv",  str(INDEX_CSV),
        "--out_dir",    str(MOSEI_DIR),
        "--yt_dlp",     YTDLP,
        "--ffmpeg",     FFMPEG,
        "--download_delay", "1.5",
    ], "download_and_cut", check=False)  # non-fatal: some videos may be unavailable
    existing = len(list(seg_dir.glob("*.wav"))) if seg_dir.exists() else 0
    STEP2_MARKER.touch()  # mark complete so restart skips this step
    log(f"STEP 2 DONE — {existing} segment WAV files")


# ── Step 3: WavLM audio features ─────────────────────────────────────────────
STEP3_MARKER = AUDIO_FEAT / ".step3_done"

def step3_wavlm():
    if STEP3_MARKER.exists():
        existing = len(list(AUDIO_FEAT.glob("*.pt"))) if AUDIO_FEAT.exists() else 0
        log(f"STEP 3 SKIP — already completed ({existing} WavLM .pt files)")
        return
    existing = len(list(AUDIO_FEAT.glob("*.pt"))) if AUDIO_FEAT.exists() else 0
    log(f"STEP 3: WavLM feature extraction ({existing} done so far) ...")
    run([
        PYTHON,
        "scripts/mosei_feature_extraction_seq_level/extract_audio_feats_wavlm.py",
        "--index_csv",   str(INDEX_CSV),
        "--seg_dir",     str(MOSEI_DIR / "segments"),
        "--out_dir",     str(AUDIO_FEAT),
        "--model_name",  "microsoft/wavlm-base-plus",
        "--max_seconds", "15.0",
    ], "wavlm_extraction")
    STEP3_MARKER.touch()
    existing = len(list(AUDIO_FEAT.glob("*.pt"))) if AUDIO_FEAT.exists() else 0
    log(f"STEP 3 DONE — {existing} WavLM .pt files")


# ── Step 4: Whisper → BERT text features ─────────────────────────────────────
STEP4_MARKER = TEXT_FEAT / ".step4_done"

def step4_whisper_bert():
    if STEP4_MARKER.exists():
        existing = len(list(TEXT_FEAT.glob("*.pt"))) if TEXT_FEAT.exists() else 0
        log(f"STEP 4 SKIP — already completed ({existing} BERT .pt files)")
        return
    existing = len(list(TEXT_FEAT.glob("*.pt"))) if TEXT_FEAT.exists() else 0
    log(f"STEP 4: Whisper→BERT feature extraction ({existing} done so far) ...")
    run([
        PYTHON,
        "scripts/mosei_feature_extraction_seq_level/extract_text_feats_bert_whisper.py",
        "--index_csv",      str(INDEX_CSV),
        "--seg_dir",        str(MOSEI_DIR / "segments"),
        "--out_dir",        str(TEXT_FEAT),
        "--whisper_model",  "base.en",
        "--bert_model",     "bert-base-uncased",
        "--bert_batch_size","32",
        "--save_transcripts",
    ], "whisper_bert_extraction")
    STEP4_MARKER.touch()
    existing = len(list(TEXT_FEAT.glob("*.pt"))) if TEXT_FEAT.exists() else 0
    log(f"STEP 4 DONE — {existing} BERT .pt files")


# ── Step 5: Train ─────────────────────────────────────────────────────────────
def step5_train():
    ckpt = OUT_DIR / "best_mosei_6cls.pt"
    if ckpt.exists():
        log(f"STEP 5 SKIP — checkpoint already exists: {ckpt}")
        return
    log("STEP 5: Training MOSEI 6-class (WavLM+BERT) ...")
    run([
        PYTHON,
        "scripts/fusion/train_mosei_6cls.py",
        "--index_csv",         str(INDEX_CSV),
        "--audio_dir",         str(AUDIO_FEAT),
        "--text_dir",          str(TEXT_FEAT),
        "--out_dir",           str(OUT_DIR),
        "--dropout",           "0.3",
        "--lr",                "3e-5",
        "--beta_hidden",       "256",
        "--num_layers_fusion", "2",
        "--max_len_audio",     "500",
        "--batch_size",        "8",
        "--grad_accum",        "4",
        "--epochs",            "40",
        "--weight_decay",      "0.05",
        "--feat_mask_prob",    "0.15",
    ], "train")
    log(f"STEP 5 DONE — checkpoint: {ckpt}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global log_f, FFMPEG

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(LOG_PATH, "a", encoding="utf-8")

    log("=" * 70)
    log("MOSEI WavLM+BERT Pipeline START")
    log(f"Python : {PYTHON}")
    log(f"yt-dlp : {YTDLP}")

    FFMPEG = find_ffmpeg()
    log(f"ffmpeg : {FFMPEG}")
    log("=" * 70)

    t0 = time.time()
    try:
        step1_download_csd()
        step2_download_and_cut()
        step3_wavlm()
        step4_whisper_bert()
        step5_train()
        elapsed = (time.time() - t0) / 3600
        log(f"\nALL STEPS COMPLETE — total time: {elapsed:.1f} h")
        log(f"Best checkpoint: {OUT_DIR / 'best_mosei_6cls.pt'}")
    except Exception as e:
        log(f"\nPIPELINE ERROR: {e}")
        raise
    finally:
        log_f.close()


if __name__ == "__main__":
    main()
