"""
Server configuration for the online emotion recognition pipeline.
All paths are relative to the project root (one level above server/).

The WavLM+BERT checkpoints (IEMOCAP and MELD) need nothing configured here.
The MOSEI checkpoints additionally need COVAREP, MATLAB and GloVe, and the
paths to those are machine-specific: edit COVAREP_ROOT, COVAREP_RUNNER_BIN and
GLOVE_MODEL_PATH near the bottom of this file to match your own installation.
Until they are correct, the MOSEI checkpoints are shown as unavailable on the
platform and the other checkpoints keep working. See server/README.md, Section 6.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
_IS_WINDOWS = sys.platform == "win32"

# ─── Benchmarks ────────────────────────────────────────────────────────────────
# Legacy default used when a client posts an explicit ``benchmark`` form field
# without a checkpoint id. The platform-configured checkpoint (see
# ``server/runtime_settings.py``) takes precedence for normal requests.
DEFAULT_BENCHMARK = "iemocap"

# ─── Audio decoding ────────────────────────────────────────────────────────────
TARGET_SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 10.0

# ─── Whisper ASR ───────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "base"   # tiny / base / small / medium / large
# MOSEI (COVAREP+GloVe) path: order GloVe word vectors by Whisper word timestamps
WHISPER_WORD_TIMESTAMPS = True

# ─── IEMOCAP 4-class ───────────────────────────────────────────────────────────
# Best single-model checkpoint (seed=7777, WA=68.91% on val)
IEMOCAP_CHECKPOINT = str(
    ROOT_DIR / "runs" / "iemocap_4cls_seed7777" / "best_fusion_seq_decoder.pt"
)
# Label order must match label2id stored in the checkpoint (alphabetical)
IEMOCAP_LABELS = ["angry", "happy", "neutral", "sad"]

# Model hyperparameters (overridden by checkpoint["args"] if present)
IEMOCAP_D_MODEL          = 256
IEMOCAP_N_HEADS          = 4
IEMOCAP_NUM_LAYERS_FUSION  = 1
IEMOCAP_NUM_LAYERS_DECODER = 2
IEMOCAP_BETA_HIDDEN      = 128
IEMOCAP_DROPOUT          = 0.1
# Sequence-length caps matching training (300 audio frames, 128 text tokens)
IEMOCAP_MAX_LEN_AUDIO    = 300
IEMOCAP_MAX_LEN_TEXT     = 128

# ─── MOSEI 6-class multi-label ─────────────────────────────────────────────────
MOSEI_CHECKPOINT = str(
    ROOT_DIR / "runs" / "mosei_fusion_decoder_v2" / "best_fusion_seq_decoder.pt"
)
MOSEI_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise"]
MOSEI_AUDIO_DIM            = 74    # COVAREP dimension
MOSEI_TEXT_DIM             = 300   # GloVe dimension
MOSEI_D_MODEL              = 256
MOSEI_N_HEADS              = 4
MOSEI_NUM_LAYERS_FUSION    = 2
MOSEI_NUM_LAYERS_DECODER   = 2
MOSEI_BETA_HIDDEN          = 128
MOSEI_DROPOUT              = 0.2
MOSEI_PREDICTION_THRESHOLD = 0.5
MOSEI_MAX_LEN_AUDIO        = 600
MOSEI_MAX_LEN_TEXT         = None

# ─── COVAREP (MOSEI audio) ─────────────────────────────────────────────────────
# EDIT THESE for your machine. The values below are the ones used during
# development (a lab Windows PC and a Mac); they will not exist on yours.
#   COVAREP_ROOT       clone of https://github.com/covarep/covarep
#   COVAREP_RUNNER_BIN full path to the matlab executable, or "octave"
if _IS_WINDOWS:
    COVAREP_ROOT       = r"C:\Users\HuRoN Lab\audio_to_emotion_server\covarep-upstream"
    COVAREP_RUNNER_BIN = r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe"
else:
    COVAREP_ROOT       = str(Path.home() / "tools" / "covarep-upstream")
    COVAREP_RUNNER_BIN = "/Applications/MATLAB_R2026a.app/bin/matlab"
COVAREP_SCRIPT     = str(ROOT_DIR / "tools" / "covarep" / "extract_covarep_segment.m")
COVAREP_TARGET_DIM = 74
COVAREP_TIMEOUT_SEC = 120.0  # first MATLAB -batch launch can exceed 30 s

# ─── GloVe (MOSEI text) ────────────────────────────────────────────────────────
# EDIT THIS for your machine: full path to glove.840B.300d.txt, downloadable
# from https://nlp.stanford.edu/projects/glove/ (about 2 GB uncompressed).
if _IS_WINDOWS:
    GLOVE_MODEL_PATH = str(ROOT_DIR / "data" / "glove.840B.300d.txt")
else:
    GLOVE_MODEL_PATH = str(
        ROOT_DIR.parent / "HRI-EMO" / "data" / "glove" / "glove.840B.300d.txt"
    )
