#!/usr/bin/env python3
"""
Download all MOSEI YouTube videos and cut every segment to a WAV file.

Prerequisites:
  yt-dlp   (pip install yt-dlp  OR  choco install yt-dlp)
  ffmpeg   (choco install ffmpeg  OR  winget install ffmpeg)
  h5py     (pip install h5py)

Needs:
  data/MOSEI/CMU_MOSEI_Labels.csd   -- segment timestamps (HDF5 / mmsdk CSD)

Outputs:
  <out_dir>/raw/          -- one {video_id}.wav per unique video
  <out_dir>/segments/     -- one {uid}.wav per segment (16 kHz mono)
  <out_dir>/manifest.csv  -- status of every segment

Run from repo root:
  python scripts/mosei_feature_extraction_seq_level/download_and_cut_mosei_audio.py \\
      --csd data/MOSEI/CMU_MOSEI_Labels.csd \\
      --index_csv data/mosei_index_splits.csv \\
      --out_dir data/mosei_wavlm_bert/raw_audio
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csd", default="data/MOSEI/CMU_MOSEI_Labels.csd",
                    help="Path to CMU_MOSEI_Labels.csd (HDF5 file with segment intervals)")
    ap.add_argument("--index_csv", default="data/mosei_index_splits.csv")
    ap.add_argument("--out_dir", default="data/mosei_wavlm_bert/raw_audio")
    ap.add_argument("--yt_dlp", default=None, help="yt-dlp executable (default: from PATH)")
    ap.add_argument("--ffmpeg", default=None, help="ffmpeg executable (default: from PATH)")
    ap.add_argument("--target_sr", type=int, default=16000)
    ap.add_argument("--download_delay", type=float, default=2.0,
                    help="Seconds to wait between downloads (avoid rate-limiting)")
    ap.add_argument("--cookies_from_browser", default=None,
                    help="Browser to pull cookies from, e.g. chrome (helps with 403 errors)")
    ap.add_argument("--skip_download", action="store_true",
                    help="Skip YouTube download, only cut segments from existing raw files")
    ap.add_argument("--skip_cut", action="store_true",
                    help="Skip segment cutting (raw files only)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="Which splits to process")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def _require_tool(name: str, override: str | None) -> str:
    path = override or shutil.which(name)
    if not path:
        raise SystemExit(
            f"{name} not found on PATH. Install it first.\n"
            f"  Windows: choco install {name}  OR  winget install {name}"
        )
    return path


def load_all_intervals(csd_path: Path) -> Dict[str, np.ndarray]:
    """Return {video_id: intervals_array[N, 2]} from CMU_MOSEI_Labels.csd."""
    try:
        import h5py
    except ImportError:
        raise SystemExit("h5py is required: pip install h5py")

    intervals: Dict[str, np.ndarray] = {}
    with h5py.File(csd_path, "r") as f:
        data = f["All Labels"]["data"]
        for vid in data.keys():
            intervals[vid] = np.asarray(data[vid]["intervals"])  # [N, 2]
    return intervals


def download_video(yt_dlp: str, video_id: str, raw_dir: Path, *,
                   dry_run: bool, cookies_from_browser: str | None,
                   delay: float) -> Tuple[Path | None, str | None]:
    """Download audio for one video_id. Returns (path, error_msg)."""
    # Check if already downloaded (any audio format)
    for ext in (".wav", ".webm", ".m4a", ".opus", ".mp3"):
        p = raw_dir / f"{video_id}{ext}"
        if p.exists():
            print(f"  [skip] already have {p.name}")
            return p, None

    url = f"https://www.youtube.com/watch?v={video_id}"
    out_template = str(raw_dir / "%(id)s.%(ext)s")
    cmd = [yt_dlp, "-x", "--audio-format", "wav", "--no-playlist",
           "--retries", "2", "--fragment-retries", "2",
           "--socket-timeout", "30",       # don't hang on slow connections
           "--ignore-no-formats-error",    # skip unavailable/private videos cleanly
           "--no-warnings"]
    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    cmd += ["-o", out_template, url]

    print(f"  [download] {url}")
    if dry_run:
        print("   ", " ".join(cmd))
        return raw_dir / f"{video_id}.wav", None

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        msg = f"yt-dlp timed out after 300s for {video_id}"
        print(f"  [FAILED] {msg}", file=sys.stderr)
        return None, msg
    time.sleep(delay)

    if proc.returncode != 0:
        lines = (proc.stderr or proc.stdout or "").strip().splitlines()
        msg = lines[-1] if lines else f"yt-dlp exit {proc.returncode}"
        print(f"  [SKIP] {video_id}: {msg}", file=sys.stderr)
        return None, msg

    # Find what was saved
    for ext in (".wav", ".webm", ".m4a", ".opus", ".mp3"):
        p = raw_dir / f"{video_id}{ext}"
        if p.exists():
            return p, None
    msg = f"yt-dlp finished but no audio file found for {video_id}"
    print(f"  [FAILED] {video_id}: {msg}", file=sys.stderr)
    return None, msg


def cut_segment(ffmpeg: str, raw_path: Path, out_path: Path,
                start: float, end: float, target_sr: int, *, dry_run: bool) -> str | None:
    """Cut one segment from raw_path → out_path. Returns error msg or None."""
    if out_path.exists():
        return None

    duration = max(0.0, end - start)
    if duration <= 0:
        return f"Invalid interval [{start:.3f}, {end:.3f}]"

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(raw_path),
           "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
           "-ac", "1", "-ar", str(target_sr),
           str(out_path)]

    if dry_run:
        print("   ", " ".join(cmd))
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or f"ffmpeg exit {proc.returncode}").strip()
        return msg
    return None


def write_manifest(path: Path, rows: List[dict]) -> None:
    fields = ["uid", "video_id", "seg_idx", "split",
              "seg_start", "seg_end", "duration_sec",
              "raw_path", "segment_path", "status", "error"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    args = parse_args()
    csd_path = Path(args.csd)
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    seg_dir = out_dir / "segments"

    raw_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        yt_dlp = _require_tool("yt-dlp", args.yt_dlp)
        ffmpeg = _require_tool("ffmpeg", args.ffmpeg)
    else:
        yt_dlp = args.yt_dlp or "yt-dlp"
        ffmpeg = args.ffmpeg or "ffmpeg"

    # Load index
    df = pd.read_csv(args.index_csv)
    df = df[df["split"].isin(args.splits)].reset_index(drop=True)
    print(f"Segments to process: {len(df)} "
          f"(splits={args.splits}, unique videos={df['video_id'].nunique()})")

    # Load all segment intervals from CSD
    print(f"\nLoading segment intervals from {csd_path} ...")
    if not csd_path.exists():
        raise FileNotFoundError(
            f"{csd_path} not found.\n"
            "Download CMU_MOSEI_Labels.csd from the CMU-MultimodalSDK dataset "
            "and place it at data/MOSEI/CMU_MOSEI_Labels.csd"
        )
    all_intervals = load_all_intervals(csd_path)
    print(f"  Loaded intervals for {len(all_intervals)} videos")

    # Group segments by video_id
    by_video: Dict[str, List] = {}
    missing_video: List[str] = []
    for _, row in df.iterrows():
        vid = str(row["video_id"])
        if vid not in all_intervals:
            missing_video.append(vid)
            continue
        by_video.setdefault(vid, []).append(row)

    if missing_video:
        unique_missing = sorted(set(missing_video))
        print(f"  WARNING: {len(unique_missing)} video_ids not found in CSD "
              f"(first 5: {unique_missing[:5]})")

    manifest_rows: List[dict] = []
    downloaded: Dict[str, Path | None] = {}
    n_ok = 0
    n_fail = 0

    print(f"\n{'='*60}")
    print(f"Processing {len(by_video)} unique videos ...")
    print(f"{'='*60}")

    for v_idx, (vid, rows) in enumerate(by_video.items()):
        print(f"\n[{v_idx+1}/{len(by_video)}] {vid}  ({len(rows)} segments)")

        # --- Download ---
        raw_path: Path | None = None
        dl_error: str | None = None

        if not args.skip_download:
            raw_path, dl_error = download_video(
                yt_dlp, vid, raw_dir,
                dry_run=args.dry_run,
                cookies_from_browser=args.cookies_from_browser,
                delay=args.download_delay,
            )
        else:
            # Find existing file
            for ext in (".wav", ".webm", ".m4a", ".opus", ".mp3"):
                p = raw_dir / f"{vid}{ext}"
                if p.exists():
                    raw_path = p
                    break
            if raw_path is None:
                dl_error = f"No raw audio found for {vid} in {raw_dir}"
                print(f"  [skip_download] {dl_error}", file=sys.stderr)

        downloaded[vid] = raw_path

        intervals = all_intervals[vid]  # [N, 2]

        # --- Cut segments ---
        for row in rows:
            uid = str(row["uid"])
            seg_idx = int(row["seg_idx"])
            split = str(row["split"])
            seg_path = seg_dir / f"{uid}.wav"

            if seg_idx >= len(intervals):
                err = f"seg_idx={seg_idx} out of range ({len(intervals)} segs in CSD)"
                print(f"  [WARN] {uid}: {err}", file=sys.stderr)
                manifest_rows.append({
                    "uid": uid, "video_id": vid, "seg_idx": seg_idx, "split": split,
                    "status": "error", "error": err,
                })
                n_fail += 1
                continue

            start, end = float(intervals[seg_idx, 0]), float(intervals[seg_idx, 1])

            if dl_error:
                manifest_rows.append({
                    "uid": uid, "video_id": vid, "seg_idx": seg_idx, "split": split,
                    "seg_start": f"{start:.3f}", "seg_end": f"{end:.3f}",
                    "raw_path": "", "segment_path": "", "status": "download_failed",
                    "error": dl_error,
                })
                n_fail += 1
                continue

            if args.skip_cut:
                manifest_rows.append({
                    "uid": uid, "video_id": vid, "seg_idx": seg_idx, "split": split,
                    "seg_start": f"{start:.3f}", "seg_end": f"{end:.3f}",
                    "raw_path": str(raw_path), "segment_path": "",
                    "status": "cut_skipped", "error": "",
                })
                continue

            cut_err = cut_segment(ffmpeg, raw_path, seg_path, start, end,
                                  args.target_sr, dry_run=args.dry_run)

            if cut_err:
                print(f"  [WARN] {uid}: cut failed: {cut_err}", file=sys.stderr)
                manifest_rows.append({
                    "uid": uid, "video_id": vid, "seg_idx": seg_idx, "split": split,
                    "seg_start": f"{start:.3f}", "seg_end": f"{end:.3f}",
                    "raw_path": str(raw_path), "segment_path": "",
                    "status": "cut_failed", "error": cut_err,
                })
                n_fail += 1
            else:
                dur = end - start
                manifest_rows.append({
                    "uid": uid, "video_id": vid, "seg_idx": seg_idx, "split": split,
                    "seg_start": f"{start:.3f}", "seg_end": f"{end:.3f}",
                    "duration_sec": f"{dur:.3f}",
                    "raw_path": str(raw_path),
                    "segment_path": str(seg_path),
                    "status": "ok", "error": "",
                })
                n_ok += 1

    # Write manifest
    manifest_path = out_dir / "manifest.csv"
    write_manifest(manifest_path, manifest_rows)

    print(f"\n{'='*60}")
    print(f"Done.  OK={n_ok}  FAILED={n_fail}")
    print(f"Segments : {seg_dir}")
    print(f"Raw audio: {raw_dir}")
    print(f"Manifest : {manifest_path}")
    if n_fail > 0:
        print(f"\nRe-run to retry failures (existing files are skipped).")
        print("For HTTP 403: add --cookies_from_browser chrome")


if __name__ == "__main__":
    main()
