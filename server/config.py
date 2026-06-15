"""Server configuration."""
from __future__ import annotations

from pathlib import Path

# Repo root (parent of server/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Model checkpoints (under repo-root checkpoints/)
# ---------------------------------------------------------------------------
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
MOSEI_CHECKPOINT_DIR = CHECKPOINTS_DIR / "mosei"
MOSEI_CHECKPOINT: str = str(MOSEI_CHECKPOINT_DIR / "best_mosei_fusion_decoder.pt")
MOSEI_INFER_OUTPUT_DIR: str = str(MOSEI_CHECKPOINT_DIR / "infer_outputs")
MOSEI_PLOTS_DIR: str = str(MOSEI_CHECKPOINT_DIR / "plots")

IEMOCAP_CHECKPOINT_DIR = CHECKPOINTS_DIR / "iemocap"
IEMOCAP_CHECKPOINT: str = ""  # e.g. str(IEMOCAP_CHECKPOINT_DIR / "best_fusion_seq_decoder.pt")

# Default benchmark for /predict when not specified
DEFAULT_BENCHMARK: str = "iemocap"

# ---------------------------------------------------------------------------
# Shared audio preprocessing
# ---------------------------------------------------------------------------
TARGET_SAMPLE_RATE: int = 16_000
MAX_AUDIO_SECONDS: float = 10.0

# Whisper ASR (openai-whisper)
WHISPER_MODEL_SIZE: str = "base"  # tiny | base | small | medium | large
# MOSEI text features: use Whisper word-level timestamps before GloVe lookup
WHISPER_WORD_TIMESTAMPS: bool = True

# ---------------------------------------------------------------------------
# MOSEI online feature extraction (COVAREP + GloVe)
# ---------------------------------------------------------------------------
# Clone https://github.com/covarep/covarep and set absolute path:
COVAREP_ROOT: str = "/Users/beimei/tools/covarep-upstream"
# macOS: use full path if matlab is not on PATH
COVAREP_RUNNER_BIN: str = "/Applications/MATLAB_R2026a.app/bin/matlab"
COVAREP_SCRIPT: str = str(ROOT_DIR / "tools" / "covarep" / "extract_covarep_segment.m")
COVAREP_TIMEOUT_SEC: float = 900.0
COVAREP_TARGET_DIM: int = 74

# MOSEI TimestampedWordVectors uses glove.840B.300d (Stanford NLP), NOT wiki-gigaword:
# https://nlp.stanford.edu/projects/glove/  →  glove.840B.300d.txt
GLOVE_MODEL_PATH: str = str(ROOT_DIR / "data/glove/glove.840B.300d.txt")

MOSEI_AUDIO_DIM: int = 74
MOSEI_TEXT_DIM: int = 300
MOSEI_MAX_LEN_AUDIO: int = 300
MOSEI_MAX_LEN_TEXT: int = 128

# ---------------------------------------------------------------------------
# IEMOCAP model hyper-params (defaults from train_fusion_seq_level_decoder.py)
# ---------------------------------------------------------------------------
IEMOCAP_D_MODEL: int = 768
IEMOCAP_N_HEADS: int = 8
IEMOCAP_NUM_LAYERS_FUSION: int = 2
IEMOCAP_NUM_LAYERS_DECODER: int = 2
IEMOCAP_BETA_HIDDEN: int = 256
IEMOCAP_DROPOUT: float = 0.1

IEMOCAP_LABELS: list[str] = [
    "angry",
    "happy",
    "sad",
    "neutral",
    "frustration",
    "excited",
]

# ---------------------------------------------------------------------------
# MOSEI model hyper-params (defaults from train_mosei_fusion_seq_level_decoder.py)
# ---------------------------------------------------------------------------
MOSEI_D_MODEL: int = 256
MOSEI_N_HEADS: int = 4
MOSEI_NUM_LAYERS_FUSION: int = 2
MOSEI_NUM_LAYERS_DECODER: int = 2
MOSEI_BETA_HIDDEN: int = 128
MOSEI_DROPOUT: float = 0.1

MOSEI_LABELS: list[str] = [
    "happy",
    "sad",
    "anger",
    "fear",
    "disgust",
    "surprise",
]

# Fallback when checkpoint has no val_calibrated_thresholds (--save_calibrated_ths)
MOSEI_PREDICTION_THRESHOLD: float = 0.5
