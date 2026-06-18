#!/usr/bin/env python3
"""Extract 16kHz mono WAV from MELD mp4 clips listed in data/meld_index_full.csv."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

FFMPEG = r"C:\Users\HuRoN Lab\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
INDEX_CSV = "data/meld_index_full.csv"
OUT_DIR = Path("data/meld_raw/wav")


def extract(video_path: Path, wav_path: Path) -> bool:
    if wav_path.is_file():
        return True
    if not video_path.is_file():
        return False
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(video_path), "-ac", "1", "-ar", "16000", str(wav_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0 and wav_path.is_file()


def main() -> None:
    df = pd.read_csv(INDEX_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="ffmpeg extract"):
        wav_path = OUT_DIR / f"{row['uid']}.wav"
        if extract(Path(row["video_path"]), wav_path):
            ok += 1
        else:
            fail += 1
    print(f"\nDone. ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
