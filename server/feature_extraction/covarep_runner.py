"""
Online COVAREP extraction via subprocess (MATLAB/Octave plan B).

Writes a temporary WAV, invokes ``tools/covarep/extract_covarep_segment.m``,
and loads the resulting ``features`` matrix ``[T, 74]``.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy.io
import soundfile as sf

from server.config import (
    COVAREP_ROOT,
    COVAREP_RUNNER_BIN,
    COVAREP_SCRIPT,
    COVAREP_TARGET_DIM,
    COVAREP_TIMEOUT_SEC,
    MOSEI_MAX_LEN_AUDIO,
    TARGET_SAMPLE_RATE,
)


class CovarepRunner:
    """Run COVAREP feature extraction on mono waveforms."""

    def __init__(
        self,
        *,
        covarep_root: str | None = None,
        runner_bin: str | None = None,
        script_path: str | Path | None = None,
        timeout_sec: float | None = None,
        target_dim: int | None = None,
    ):
        self.covarep_root = Path(covarep_root or COVAREP_ROOT).expanduser()
        self.runner_bin = (runner_bin or COVAREP_RUNNER_BIN).strip()
        self.script_path = Path(script_path or COVAREP_SCRIPT).resolve()
        self.timeout_sec = float(timeout_sec if timeout_sec is not None else COVAREP_TIMEOUT_SEC)
        self.target_dim = int(target_dim if target_dim is not None else COVAREP_TARGET_DIM)
        self._runner_exe: Path | None = None
        self._runner_kind: str | None = None

    def _resolve_runner(self) -> tuple[Path, str]:
        """Resolve runner executable and kind (``matlab`` or ``octave``)."""
        if self._runner_exe is not None and self._runner_kind is not None:
            return self._runner_exe, self._runner_kind

        raw = self.runner_bin.strip()
        candidate = Path(raw).expanduser()
        if candidate.is_file():
            exe = candidate.resolve()
        else:
            found = shutil.which(raw)
            if found is None:
                raise RuntimeError(
                    f"COVAREP runner '{raw}' not found. "
                    "Install MATLAB/Octave, add it to PATH, or pass an absolute "
                    "path via COVAREP_RUNNER_BIN / --covarep-runner."
                )
            exe = Path(found).resolve()

        name = exe.stem.lower()
        if name == "matlab":
            kind = "matlab"
        elif name == "octave":
            kind = "octave"
        else:
            raise RuntimeError(
                f"Unsupported COVAREP runner executable: {exe}. "
                "Expected a binary named 'matlab' or 'octave'."
            )

        self._runner_exe = exe
        self._runner_kind = kind
        return exe, kind

    def _validate_setup(self) -> None:
        if not self.covarep_root.is_dir():
            raise RuntimeError(
                "COVAREP_ROOT is not set or does not exist. "
                "Clone https://github.com/covarep/covarep and set server/config.py::COVAREP_ROOT."
            )
        if not self.script_path.is_file():
            raise RuntimeError(f"COVAREP script not found: {self.script_path}")
        self._resolve_runner()

    def _build_command(self, wav_path: Path, mat_path: Path) -> list[str]:
        wav_s = str(wav_path.resolve())
        mat_s = str(mat_path.resolve())
        root_s = str(self.covarep_root.resolve())
        runner_exe, kind = self._resolve_runner()

        if kind == "matlab":
            batch = (
                f"addpath('{self.script_path.parent}'); "
                f"extract_covarep_segment('{wav_s}','{mat_s}','{root_s}');"
            )
            return [str(runner_exe), "-nodisplay", "-batch", batch]

        return [
            str(runner_exe),
            "--no-gui",
            "--quiet",
            "--path",
            str(self.script_path.parent),
            str(self.script_path.parent / "extract_covarep_segment_octave.m"),
            wav_s,
            mat_s,
            root_s,
        ]

    def extract_frames(
        self,
        wav_np: np.ndarray,
        *,
        sample_rate: int = TARGET_SAMPLE_RATE,
        max_len: int | None = MOSEI_MAX_LEN_AUDIO,
    ) -> np.ndarray:
        """
        Extract COVAREP frame features from a mono waveform.

        Args:
            max_len: Center-crop to this many frames when set. Pass ``None`` to
                keep the full sequence (useful for offline validation).

        Returns:
            Float32 array ``[T, target_dim]`` with NaN/Inf cleaned.
        """
        self._validate_setup()

        wav_np = np.asarray(wav_np, dtype=np.float32).reshape(-1)
        if wav_np.size == 0:
            return np.zeros((1, self.target_dim), dtype=np.float32)

        with tempfile.TemporaryDirectory(prefix="hriemo_covarep_") as tmp:
            tmp_dir = Path(tmp)
            wav_path = tmp_dir / "input.wav"
            mat_path = tmp_dir / "features.mat"

            sf.write(str(wav_path), wav_np, sample_rate, subtype="PCM_16")

            cmd = self._build_command(wav_path, mat_path)
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
            )
            if proc.returncode != 0:
                stderr = (proc.stderr or proc.stdout or "").strip()
                raise RuntimeError(
                    f"COVAREP extraction failed (exit {proc.returncode}): {stderr}"
                )
            if not mat_path.is_file():
                raise RuntimeError("COVAREP did not produce features.mat")

            mat = scipy.io.loadmat(str(mat_path))
            features = np.asarray(mat["features"], dtype=np.float32)
            if features.ndim == 1:
                features = features.reshape(1, -1)

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = _fit_feature_dim(features, self.target_dim)
        if max_len is not None:
            features = _crop_center_frames(features, max_len)
        return features.astype(np.float32)


def _fit_feature_dim(frames: np.ndarray, target_dim: int) -> np.ndarray:
    n_frames, n_dims = frames.shape
    if n_dims == target_dim:
        return frames
    if n_dims > target_dim:
        return frames[:, :target_dim]
    pad = np.zeros((n_frames, target_dim - n_dims), dtype=frames.dtype)
    return np.concatenate([frames, pad], axis=1)


def _crop_center_frames(frames: np.ndarray, max_len: int) -> np.ndarray:
    if max_len is None or max_len <= 0 or frames.shape[0] <= max_len:
        return frames
    start = (frames.shape[0] - max_len) // 2
    return frames[start : start + max_len]
