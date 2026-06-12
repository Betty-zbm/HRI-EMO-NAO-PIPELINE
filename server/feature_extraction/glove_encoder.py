"""
GloVe 300-d word embeddings for online MOSEI text features.

CMU-MOSEI ``CMU_MOSEI_TimestampedWordVectors`` uses **glove.840B.300d**
(Common Crawl 840B tokens, 300 dimensions) — the same vectors used in
CMU Multimodal SDK / MOSEI baseline papers, NOT ``glove-wiki-gigaword-300``.

Download from Stanford NLP:
  https://nlp.stanford.edu/projects/glove/
  file: ``glove.840B.300d.zip`` → extract ``glove.840B.300d.txt`` (~5 GB)

Set ``server/config.py::GLOVE_MODEL_PATH`` to the ``.txt`` file or its directory.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from server.config import GLOVE_MODEL_PATH, MOSEI_MAX_LEN_TEXT, MOSEI_TEXT_DIM, ROOT_DIR
from server.feature_extraction.whisper import TimedWord

# MOSEI CSD text features match this embedding table
MOSEI_GLOVE_VARIANT = "glove.840B.300d"

_WORD_RE = re.compile(r"\b[\w']+\b")


class GloveEncoder:
    """Lazy-loaded GloVe lookup table for word-level MOSEI text sequences."""

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        dim: int = MOSEI_TEXT_DIM,
    ):
        resolved = model_path or GLOVE_MODEL_PATH
        if resolved:
            p = Path(resolved).expanduser()
            if not p.is_absolute():
                p = ROOT_DIR / p
            self.model_path = p.resolve()
        else:
            self.model_path = Path()
        self.dim = int(dim)
        self._kv = None

    def _resolve_model_file(self) -> Path:
        if not self.model_path.exists():
            raise RuntimeError(
                "GLOVE_MODEL_PATH is not set or does not exist. "
                f"Download {MOSEI_GLOVE_VARIANT}.txt from "
                "https://nlp.stanford.edu/projects/glove/ and set "
                "server/config.py::GLOVE_MODEL_PATH."
            )

        if self.model_path.is_file():
            return self.model_path

        # Prefer MOSEI-compatible 840B table
        for pattern in ("*840B*.txt", "glove.840B.300d.txt", "*.txt", "*.model", "*glove*"):
            matches = sorted(self.model_path.glob(pattern))
            if matches:
                preferred = [p for p in matches if "840B" in p.name]
                return preferred[0] if preferred else matches[0]

        raise RuntimeError(
            f"No GloVe file found under {self.model_path}. "
            f"Expected {MOSEI_GLOVE_VARIANT}.txt"
        )

    def _ensure_loaded(self) -> None:
        if self._kv is not None:
            return

        from gensim.models import KeyedVectors

        model_file = self._resolve_model_file()
        suffix = model_file.suffix.lower()

        if suffix == ".txt" or "840B" in model_file.name:
            # Stanford GloVe text format (MOSEI standard)
            path = str(model_file)
            if hasattr(KeyedVectors, "load_word_vectors_format"):
                self._kv = KeyedVectors.load_word_vectors_format(
                    path, binary=False, no_header=True
                )
            else:
                # gensim 4.x renamed to load_word2vec_format
                self._kv = KeyedVectors.load_word2vec_format(
                    path, binary=False, no_header=True
                )
        else:
            # Optional: gensim .model (e.g. wiki-gigaword) — not MOSEI-aligned
            self._kv = KeyedVectors.load(str(model_file), mmap="r")

        if self._kv.vector_size != self.dim:
            raise RuntimeError(
                f"Expected GloVe dim {self.dim}, got {self._kv.vector_size} from {model_file}"
            )

    def lookup(self, word: str) -> np.ndarray:
        """Return a 300-d vector for ``word`` (zeros for OOV)."""
        self._ensure_loaded()
        key = word.lower()
        if key in self._kv:
            return self._kv[key].astype(np.float32)
        return np.zeros(self.dim, dtype=np.float32)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        words = _WORD_RE.findall(text.lower())
        return words if words else ["<unk>"]

    def encode_text(self, text: str) -> np.ndarray:
        """Encode transcript to a word-level embedding sequence ``[L, dim]``."""
        words = self.tokenize(text)
        return self.encode_words(words)

    def encode_words(
        self,
        words: list[str],
        *,
        max_len: int | None = MOSEI_MAX_LEN_TEXT,
    ) -> np.ndarray:
        """Map an ordered word list to GloVe vectors ``[L, dim]``."""
        if not words:
            words = ["<unk>"]
        vectors = np.stack([self.lookup(word) for word in words], axis=0)
        vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
        if max_len is not None:
            vectors = _crop_center_words(vectors, max_len)
        return vectors.astype(np.float32)

    def encode_timed_words(
        self,
        timed_words: list[TimedWord],
        *,
        max_duration: float | None = None,
        max_len: int | None = MOSEI_MAX_LEN_TEXT,
    ) -> np.ndarray:
        """
        Build GloVe sequence from Whisper word timestamps.

        Words are sorted by ``start`` time, filtered to ``[0, max_duration)``,
        then tokenized for GloVe lookup (punctuation stripped per token).
        """
        ordered: list[str] = []
        for item in sorted(timed_words, key=lambda w: (w.start, w.end)):
            if max_duration is not None and item.start >= max_duration:
                continue
            ordered.extend(self.tokenize(item.word))

        return self.encode_words(ordered, max_len=max_len)

    @property
    def is_configured(self) -> bool:
        try:
            self._resolve_model_file()
            return True
        except RuntimeError:
            return False


def _crop_center_words(vectors: np.ndarray, max_len: int) -> np.ndarray:
    if max_len is None or max_len <= 0 or vectors.shape[0] <= max_len:
        return vectors
    start = (vectors.shape[0] - max_len) // 2
    return vectors[start : start + max_len]
