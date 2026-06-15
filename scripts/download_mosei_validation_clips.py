#!/usr/bin/env python3
"""
Download MOSEI test-split validation clips (YouTube → WAV segment).

Uses 10 segments verified online (list numbers 1,3,4,5,6,7,11,12,13,14 from
the suggested 15-sample test list).

Prerequisites (outside the server venv is fine):
  brew install yt-dlp ffmpeg
  pip install h5py    # read segment intervals from Labels.csd (no mmsdk needed)

Run from repo root:
  python scripts/download_mosei_validation_clips.py
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# Verified-online entries from the suggested 15-sample test list (1-based index).
SELECTED_SAMPLES: list[dict[str, object]] = [
    {"list_no": 1, "uid": "-6rXp3zJ3kc_1", "video_id": "-6rXp3zJ3kc", "seg_idx": 1},
    {"list_no": 3, "uid": "7f3ndBCx_JE_1", "video_id": "7f3ndBCx_JE", "seg_idx": 1},
    {"list_no": 4, "uid": "CO2YoTZbUr0_5", "video_id": "CO2YoTZbUr0", "seg_idx": 5},
    {"list_no": 5, "uid": "H-74k5vclCU_1", "video_id": "H-74k5vclCU", "seg_idx": 1},
    {"list_no": 6, "uid": "LFOwCSiGOvw_1", "video_id": "LFOwCSiGOvw", "seg_idx": 1},
    {"list_no": 7, "uid": "OaWYjsS02fk_2", "video_id": "OaWYjsS02fk", "seg_idx": 2},
    {"list_no": 11, "uid": "dHk--ExZbHs_2", "video_id": "dHk--ExZbHs", "seg_idx": 2},
    {"list_no": 12, "uid": "gR3igiwaeyc_4", "video_id": "gR3igiwaeyc", "seg_idx": 4},
    {"list_no": 13, "uid": "l1jW3OMXUzs_0", "video_id": "l1jW3OMXUzs", "seg_idx": 0},
    {"list_no": 14, "uid": "p4WmcxrXkc4_2", "video_id": "p4WmcxrXkc4", "seg_idx": 2},
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=str, default="data/MOSEI")
    ap.add_argument("--out-dir", type=str, default="data/online_mosei_validation")
    ap.add_argument(
        "--yt-dlp",
        type=str,
        default=None,
        help="yt-dlp executable (default: first on PATH)",
    )
    ap.add_argument(
        "--ffmpeg",
        type=str,
        default=None,
        help="ffmpeg executable (default: first on PATH)",
    )
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-cut", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first download/cut error (default: skip and continue)",
    )
    ap.add_argument(
        "--cookies-from-browser",
        type=str,
        default=None,
        metavar="BROWSER",
        help="Pass to yt-dlp, e.g. chrome — helps with HTTP 403 on some videos",
    )
    ap.add_argument(
        "--yt-dlp-extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra yt-dlp args before the URL, e.g. --yt-dlp-extra --retries 5",
    )
    return ap.parse_args()


def _require_tool(name: str, override: str | None) -> str:
    path = override or shutil.which(name)
    if not path:
        raise SystemExit(
            f"{name} not found. Install it first, e.g. `brew install {name}`."
        )
    return path


def load_segment_intervals(data_root: Path) -> dict[tuple[str, int], tuple[float, float]]:
    """Read [start, end] seconds from CMU_MOSEI_Labels.csd via h5py (no mmsdk)."""
    label_csd = data_root / "CMU_MOSEI_Labels.csd"
    if not label_csd.exists():
        raise FileNotFoundError(f"{label_csd} not found.")

    try:
        import h5py
    except ImportError as exc:
        raise SystemExit(
            "h5py is required to read segment intervals from Labels.csd.\n"
            "Install with: pip install h5py"
        ) from exc

    intervals: dict[tuple[str, int], tuple[float, float]] = {}
    with h5py.File(label_csd, "r") as f:
        data = f["All Labels"]["data"]
        for sample in SELECTED_SAMPLES:
            vid = str(sample["video_id"])
            seg_idx = int(sample["seg_idx"])
            if vid not in data:
                raise KeyError(f"video_id {vid!r} not found in {label_csd.name}")
            seg_intervals = np.asarray(data[vid]["intervals"])
            if seg_idx >= seg_intervals.shape[0]:
                raise IndexError(
                    f"{vid} seg_idx={seg_idx} out of range ({seg_intervals.shape[0]} segs)"
                )
            start, end = map(float, seg_intervals[seg_idx])
            intervals[(vid, seg_idx)] = (start, end)
    return intervals


def download_video(
    yt_dlp: str,
    video_id: str,
    raw_dir: Path,
    *,
    dry_run: bool,
    cookies_from_browser: str | None = None,
    yt_dlp_extra: list[str] | None = None,
) -> tuple[Path | None, str | None]:
    out_template = str(raw_dir / "%(id)s.%(ext)s")
    wav_path = raw_dir / f"{video_id}.wav"
    if wav_path.exists():
        print(f"[skip download] {wav_path}")
        return wav_path, None

    matches = [p for p in raw_dir.glob(f"{video_id}.*") if p.suffix.lower() in {".wav", ".webm", ".m4a", ".opus"}]
    if matches:
        print(f"[skip download] found {matches[0]}")
        return matches[0], None

    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [yt_dlp, "-x", "--audio-format", "wav", "--no-playlist", "--retries", "3"]
    if cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser])
    if yt_dlp_extra:
        cmd.extend(yt_dlp_extra)
    cmd.extend(["-o", out_template, url])
    print(f"[download] {url}")
    if dry_run:
        print("  ", " ".join(cmd))
        return wav_path, None

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        msg = err[-1] if err else f"yt-dlp exit {proc.returncode}"
        print(f"[download failed] {video_id}: {msg}", file=sys.stderr)
        return None, msg

    if wav_path.exists():
        return wav_path, None
    matches = list(raw_dir.glob(f"{video_id}.*"))
    if not matches:
        msg = f"yt-dlp finished but no file for {video_id} in {raw_dir}"
        print(f"[download failed] {video_id}: {msg}", file=sys.stderr)
        return None, msg
    return matches[0], None


def cut_clip(
    ffmpeg: str,
    source_wav: Path,
    clip_path: Path,
    start: float,
    end: float,
    *,
    dry_run: bool,
) -> str | None:
    if clip_path.exists():
        print(f"[skip cut] {clip_path}")
        return None

    duration = max(0.0, end - start)
    if duration <= 0:
        msg = f"Invalid interval [{start}, {end}] for {clip_path.name}"
        print(f"[cut failed] {msg}", file=sys.stderr)
        return msg

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_wav),
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(clip_path),
    ]
    print(f"[cut] {source_wav.name} [{start:.2f}s, {end:.2f}s] -> {clip_path.name}")
    if dry_run:
        print("  ", " ".join(cmd))
        return None

    clip_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or f"ffmpeg exit {proc.returncode}").strip()
        print(f"[cut failed] {clip_path.name}: {msg}", file=sys.stderr)
        return msg
    return None


def write_manifest(out_dir: Path, rows: list[dict[str, object]]) -> Path:
    manifest = out_dir / "manifest.csv"
    fieldnames = [
        "list_no",
        "uid",
        "video_id",
        "seg_idx",
        "split",
        "seg_start",
        "seg_end",
        "duration_sec",
        "youtube_url",
        "raw_audio_path",
        "clip_path",
        "status",
        "error",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return manifest


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_audio"
    clip_dir = out_dir / "test_clips"

    for path in (out_dir, raw_dir, clip_dir):
        path.mkdir(parents=True, exist_ok=True)
        print(f"[mkdir] {path}")

    yt_dlp = _require_tool("yt-dlp", args.yt_dlp)
    ffmpeg = _require_tool("ffmpeg", args.ffmpeg)

    intervals = load_segment_intervals(data_root)

    manifest_rows: list[dict[str, object]] = []
    downloaded: dict[str, Path | None] = {}
    failed: list[str] = []

    for sample in SELECTED_SAMPLES:
        vid = str(sample["video_id"])
        seg_idx = int(sample["seg_idx"])
        uid = str(sample["uid"])
        start, end = intervals[(vid, seg_idx)]
        status = "ok"
        error = ""
        raw_path: Path | None = raw_dir / f"{vid}.wav"
        clip_path = clip_dir / f"{uid}.wav"

        try:
            if not args.skip_download:
                if vid not in downloaded:
                    raw_path, dl_err = download_video(
                        yt_dlp,
                        vid,
                        raw_dir,
                        dry_run=args.dry_run,
                        cookies_from_browser=args.cookies_from_browser,
                        yt_dlp_extra=args.yt_dlp_extra,
                    )
                    if dl_err:
                        if args.fail_fast:
                            raise RuntimeError(dl_err)
                        status = "download_failed"
                        error = dl_err
                        failed.append(f"{uid}: {dl_err}")
                        downloaded[vid] = None
                    else:
                        assert raw_path is not None
                        downloaded[vid] = raw_path
                else:
                    raw_path = downloaded.get(vid)
                    if raw_path is None:
                        status = "download_failed"
                        error = error or "earlier download failed"
            else:
                if not raw_path.exists():
                    matches = list(raw_dir.glob(f"{vid}.*"))
                    if not matches:
                        msg = f"Missing raw audio for {vid} in {raw_dir}"
                        if args.fail_fast:
                            raise FileNotFoundError(msg)
                        status = "download_failed"
                        error = msg
                        failed.append(f"{uid}: {msg}")
                        raw_path = None
                    else:
                        raw_path = matches[0]

            if status == "ok" and raw_path and not args.skip_cut:
                cut_err = cut_clip(
                    ffmpeg, raw_path, clip_path, start, end, dry_run=args.dry_run
                )
                if cut_err:
                    if args.fail_fast:
                        raise RuntimeError(cut_err)
                    status = "cut_failed"
                    error = cut_err
                    failed.append(f"{uid}: {cut_err}")
        except Exception as exc:
            if args.fail_fast:
                raise
            status = "failed"
            error = str(exc)
            failed.append(f"{uid}: {error}")

        manifest_rows.append(
            {
                "list_no": sample["list_no"],
                "uid": uid,
                "video_id": vid,
                "seg_idx": seg_idx,
                "split": "test",
                "seg_start": f"{start:.3f}",
                "seg_end": f"{end:.3f}",
                "duration_sec": f"{end - start:.3f}",
                "youtube_url": f"https://www.youtube.com/watch?v={vid}",
                "raw_audio_path": (
                    str(raw_path.relative_to(out_dir)) if raw_path and raw_path.exists() else ""
                ),
                "clip_path": (
                    str(clip_path.relative_to(out_dir)) if clip_path.exists() else ""
                ),
                "status": status,
                "error": error,
            }
        )

    if args.dry_run:
        print(f"\n[dry-run] Would write manifest to {out_dir / 'manifest.csv'}")
        return

    manifest = write_manifest(out_dir, manifest_rows)
    ok_count = sum(1 for row in manifest_rows if row["status"] == "ok")
    print(f"\n[OK] manifest: {manifest}")
    print(f"     clips ready: {ok_count}/{len(manifest_rows)}")
    print(f"     raw audio: {raw_dir}")
    print(f"     test clips: {clip_dir}")
    if failed:
        print("\n[FAILED]", file=sys.stderr)
        for item in failed:
            print(f"  - {item}", file=sys.stderr)
        print(
            "\nRe-run the same command to retry (existing files are skipped).\n"
            "For HTTP 403, try: brew upgrade yt-dlp\n"
            "  python scripts/download_mosei_validation_clips.py --cookies-from-browser chrome",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
