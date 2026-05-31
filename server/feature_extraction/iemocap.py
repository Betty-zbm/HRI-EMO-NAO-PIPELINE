"""
Online IEMOCAP feature extraction.

Mirrors the offline sequence-level scripts:
  - ``scripts/iemocap_feature_extraction_seq_level/extract_audio_feats_wavlm_seq.py``
  - ``scripts/iemocap_feature_extraction_seq_level/extract_text_feats_bert_seq.py``

Audio and text encoders produce WavLM / BERT hidden states in the same tensor
layout as the precomputed ``{uid}.pt`` files used during fusion training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer

# Fixed IEMOCAP extractor settings (same defaults as offline scripts)
WAVLM_MODEL_NAME = "microsoft/wavlm-base-plus"
BERT_MODEL_NAME = "bert-base-uncased"
TARGET_SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 10.0
BERT_MAX_LEN = 128


def preprocess_audio_for_wavlm(
    wav_np: np.ndarray,
    *,
    target_sr: int = TARGET_SAMPLE_RATE,
    max_seconds: float = MAX_AUDIO_SECONDS,
) -> np.ndarray:
    """
    Pad or truncate a decoded waveform to fixed length for WavLM.

    Expects mono float32 input from ``decode_wav_bytes`` (16 kHz, peak-normalized,
    truncated to ``max_seconds``). This step only adds zero-padding when the clip
    is shorter than ``max_seconds`` — matching
    ``extract_audio_feats_wavlm_seq.py`` lines 81–87.

    Args:
        wav_np: Decoded mono float32 waveform.
        target_sr: Sample rate (default 16 kHz).
        max_seconds: Fixed clip length in seconds for WavLM input.

    Returns:
        Float32 waveform of shape [target_sr * max_seconds].
    """
    max_len_samples = int(target_sr * max_seconds)
    wav_t = torch.from_numpy(wav_np.astype(np.float32))

    if wav_t.numel() > max_len_samples:
        wav_t = wav_t[:max_len_samples]
    else:
        pad = max_len_samples - wav_t.numel()
        if pad > 0:
            wav_t = torch.nn.functional.pad(wav_t, (0, pad))

    return wav_t.detach().cpu().numpy()


@dataclass
class SeqFeatures:
    """
    Sequence features ready for a single-sample fusion-decoder forward pass.

    Online extractors convert encoder outputs into the batch layout expected by
    ``FusionWithEmotionDecoder`` / ``MoseiFusionWithEmotionDecoder`` (B=1).
    """

    hidden: torch.Tensor  # [1, L, D] float
    key_padding_mask: torch.Tensor  # [1, L] bool, True = padded position


def pack_seq_features(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> SeqFeatures:
    """
    Pack per-sequence encoder outputs into fusion-decoder batch tensors.

    Args:
        hidden: Frame/word hidden states [L, D].
        attention_mask: Valid-token mask [L], 1 = valid and 0 = padding (HF convention).

    Returns:
        SeqFeatures with ``hidden`` [1, L, D] and ``key_padding_mask`` [1, L].
    """
    return SeqFeatures(
        hidden=hidden.unsqueeze(0).float(),
        key_padding_mask=(attention_mask == 0).unsqueeze(0),
    )


def downsample_mask_linear(mask_b_l: torch.Tensor, t_prime: int) -> torch.Tensor:
    """
    Downsample a waveform-level attention mask to WavLM hidden-state length.

    WavLM compresses the input waveform along time; the raw feature-extractor
    mask has length equal to input samples (after FE stride), while
    ``last_hidden_state`` has length ``T' < L``. We pick ``T'`` indices via
    linear interpolation — same logic as
    ``extract_audio_feats_wavlm_seq.py::downsample_mask_linear``.

    Args:
        mask_b_l: Binary mask [B, L] from the WavLM feature extractor.
        t_prime: Target sequence length (WavLM hidden time steps).

    Returns:
        Downsampled mask [B, T'] aligned with ``last_hidden_state``.
    """
    _batch, length = mask_b_l.shape
    idx = torch.linspace(0, length - 1, steps=t_prime, device=mask_b_l.device)
    idx = idx.round().long().clamp_(0, length - 1)
    return mask_b_l[:, idx]


class IemocapFeatureExtractor:
    """
    Lazy-loaded WavLM + BERT encoders for online IEMOCAP inference.

    Models are loaded on first use to keep server startup fast when only one
    benchmark is exercised.
    """

    def __init__(self, *, device: Optional[str] = None):
        """
        Args:
            device: Torch device string (``"cuda"`` / ``"cpu"``). Auto-detected
                on first forward pass when omitted.
        """
        self._device = device
        self._wavlm_fe = None
        self._wavlm = None
        self._tokenizer = None
        self._bert = None

    @property
    def device(self) -> str:
        """Resolve compute device once and cache for subsequent extractions."""
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _load_wavlm(self) -> None:
        """Load WavLM feature extractor and backbone if not already in memory."""
        if self._wavlm is not None:
            return
        self._wavlm_fe = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL_NAME)
        self._wavlm = AutoModel.from_pretrained(WAVLM_MODEL_NAME).to(self.device).eval()

    def _load_bert(self) -> None:
        """Load BERT tokenizer and backbone if not already in memory."""
        if self._bert is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        self._bert = AutoModel.from_pretrained(BERT_MODEL_NAME).to(self.device).eval()

    @torch.no_grad()
    def extract_audio(self, wav_np: np.ndarray) -> SeqFeatures:
        """
        Extract WavLM sequence features from a preprocessed mono waveform.

        Expects output of ``preprocess_audio_for_wavlm`` (fixed 10 s length).
        Steps:
          1. Run WavLM feature extractor + forward pass.
          2. Downsample the FE attention mask to hidden-state length.
          3. Return fusion-decoder-ready tensors [1, T', 768].

        Args:
            wav_np: Mono float32 waveform, already padded/truncated for WavLM.

        Returns:
            SeqFeatures with ``hidden`` [1, T', 768] and ``key_padding_mask`` [1, T'].
        """
        self._load_wavlm()

        inputs = self._wavlm_fe(
            [wav_np],
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding="longest",
            return_attention_mask=True,
        )
        # Convert to PyTorch tensors and move to device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Forward pass through WavLM
        outputs = self._wavlm(**inputs)
        # Extract hidden states and attention mask
        hidden_states = outputs.last_hidden_state  # [1, T', H]
        _batch, t_prime, _hidden_dim = hidden_states.shape

        attn_down = downsample_mask_linear(inputs["attention_mask"], t_prime).squeeze(0).cpu().long()
        hidden = hidden_states.squeeze(0).cpu()

        return pack_seq_features(hidden, attn_down)

    @torch.no_grad()
    def extract_text(self, text: str) -> SeqFeatures:
        """
        Extract BERT sequence features from an ASR transcript.

        Tokenization uses fixed ``max_length=BERT_MAX_LEN`` with max padding,
        matching ``extract_text_feats_bert_seq.py``.

        Args:
            text: Transcript string (Whisper output or placeholder).

        Returns:
            SeqFeatures with ``hidden`` [1, L, 768] and ``key_padding_mask`` [1, L].
        """
        self._load_bert()
        # Tokenize transcript into alphanumeric words (regex)
        encodings = self._tokenizer(
            text,
            truncation=True,
            max_length=BERT_MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        # Convert to PyTorch tensors and move to device
        encodings = {key: value.to(self.device) for key, value in encodings.items()}

        # Forward pass through BERT
        outputs = self._bert(**encodings)
        # Extract hidden states and attention mask
        hidden = outputs.last_hidden_state.squeeze(0).cpu()
        attention_mask = encodings["attention_mask"].squeeze(0).cpu()

        return pack_seq_features(hidden, attention_mask)

    @torch.no_grad()
    def extract(self, wav_np: np.ndarray, text: str) -> tuple[SeqFeatures, SeqFeatures]:
        """
        Run audio and text extraction for a single utterance.

        Args:
            wav_np: Decoded mono float32 waveform from ``decode_wav_bytes``.
            text: Transcript paired with the audio clip.

        Returns:
            Tuple of (audio_features, text_features).
        """
        wav_lm = preprocess_audio_for_wavlm(wav_np)
        return self.extract_audio(wav_lm), self.extract_text(text)
