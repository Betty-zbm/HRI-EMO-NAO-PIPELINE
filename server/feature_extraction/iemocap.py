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

from server.audio_utils import prepare_wavlm_input

# ---------------------------------------------------------------------------
# Fixed IEMOCAP extractor settings (same defaults as offline scripts)
# ---------------------------------------------------------------------------
WAVLM_MODEL_NAME = "microsoft/wavlm-base-plus"
BERT_MODEL_NAME = "bert-base-uncased"
TARGET_SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 10.0
BERT_MAX_LEN = 128


@dataclass
class SeqFeatures:
    """
    Sequence-level multimodal features in the same format as offline ``.pt`` files.

    Both IEMOCAP and MOSEI extractors return this container so the inference
    engine can feed fusion models with a uniform interface.
    """

    hidden: torch.Tensor  # [L, D] — frame/word hidden states
    attention_mask: torch.Tensor  # [L], 1 = valid token, 0 = padding

    def to_model_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert stored features into fusion-model batch tensors.

        The fusion decoder expects:
          - hidden with shape [batch, seq_len, dim]
          - key_padding_mask with shape [batch, seq_len] where True marks PAD

        Returns:
            hidden: float tensor [1, L, D]
            key_padding_mask: bool tensor [1, L], True = padded position
        """
        hidden = self.hidden.unsqueeze(0).float()
        key_padding_mask = (self.attention_mask == 0).unsqueeze(0)
        return hidden, key_padding_mask


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

    def _ensure_audio_models(self) -> None:
        """Load WavLM feature extractor and backbone if not already in memory."""
        if self._wavlm is not None:
            return
        self._wavlm_fe = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL_NAME)
        self._wavlm = AutoModel.from_pretrained(WAVLM_MODEL_NAME).to(self.device).eval()

    def _ensure_text_models(self) -> None:
        """Load BERT tokenizer and backbone if not already in memory."""
        if self._bert is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        self._bert = AutoModel.from_pretrained(BERT_MODEL_NAME).to(self.device).eval()

    @torch.no_grad()
    def extract_audio(self, wav_np: np.ndarray) -> SeqFeatures:
        """
        Extract WavLM sequence features from a mono waveform.

        Processing steps:
          1. Pad/truncate waveform to ``MAX_AUDIO_SECONDS`` (offline script parity).
          2. Run WavLM feature extractor + forward pass.
          3. Downsample the FE attention mask to hidden-state length.
          4. Return CPU tensors matching offline ``{uid}.pt`` layout.

        Args:
            wav_np: Mono float32 waveform (any sample rate; resampling happens
                upstream in the pipeline).

        Returns:
            SeqFeatures with ``hidden`` [T', 768] and ``attention_mask`` [T'].
        """
        self._ensure_audio_models()

        wav_np = prepare_wavlm_input(
            wav_np,
            target_sr=TARGET_SAMPLE_RATE,
            max_seconds=MAX_AUDIO_SECONDS,
        )

        inputs = self._wavlm_fe(
            [wav_np],
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding="longest",
            return_attention_mask=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        outputs = self._wavlm(**inputs)
        hidden_states = outputs.last_hidden_state  # [1, T', H]
        _batch, t_prime, _hidden_dim = hidden_states.shape

        attn_down = downsample_mask_linear(inputs["attention_mask"], t_prime).squeeze(0).cpu().long()
        hidden = hidden_states.squeeze(0).cpu()

        return SeqFeatures(hidden=hidden, attention_mask=attn_down)

    @torch.no_grad()
    def extract_text(self, text: str) -> SeqFeatures:
        """
        Extract BERT sequence features from an ASR transcript.

        Tokenization uses fixed ``max_length=BERT_MAX_LEN`` with max padding,
        matching ``extract_text_feats_bert_seq.py``.

        Args:
            text: Transcript string (Whisper output or placeholder).

        Returns:
            SeqFeatures with ``hidden`` [L, 768] and ``attention_mask`` [L].
        """
        self._ensure_text_models()

        encodings = self._tokenizer(
            text,
            truncation=True,
            max_length=BERT_MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        encodings = {key: value.to(self.device) for key, value in encodings.items()}

        outputs = self._bert(**encodings)
        hidden = outputs.last_hidden_state.squeeze(0).cpu()
        attention_mask = encodings["attention_mask"].squeeze(0).cpu()

        return SeqFeatures(hidden=hidden, attention_mask=attention_mask)

    @torch.no_grad()
    def extract(self, wav_np: np.ndarray, text: str) -> tuple[SeqFeatures, SeqFeatures]:
        """
        Run audio and text extraction for a single utterance.

        Args:
            wav_np: Preprocessed mono waveform.
            text: Transcript paired with the audio clip.

        Returns:
            Tuple of (audio_features, text_features).
        """
        return self.extract_audio(wav_np), self.extract_text(text)
