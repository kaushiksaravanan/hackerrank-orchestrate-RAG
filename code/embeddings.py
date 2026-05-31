"""
Embeddings module — dense vector embeddings via HuggingFace Inference API
(preferred) or ONNX Runtime locally.

Cascade:
  1. HFInferenceEmbedder — calls HF Inference API remotely (fast, no local GPU)
  2. ONNXEmbedder — local ONNX Runtime on CPU (slow for large models)
  3. TFIDFEmbedder — sparse fallback (no external model required)
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load env files here as well because this module is imported directly by tests
# and helper scripts that may not import config.py first.
_MODULE_DIR = Path(__file__).resolve().parent
load_dotenv(_MODULE_DIR / ".env")
load_dotenv(_MODULE_DIR.parent / ".env")


def _get_hf_token() -> str | None:
    """Return the first available Hugging Face token from common env vars."""
    for name in (
        "HG_TOKEN",
        "HF_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
    ):
        value = os.getenv(name)
        if value:
            return value
    return None


class HFInferenceEmbedder:
    """Dense embeddings via the HuggingFace Inference API (remote)."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        from huggingface_hub import InferenceClient

        hf_token = _get_hf_token()
        if not hf_token:
            raise RuntimeError("No HuggingFace token found for Inference API")

        self.client = InferenceClient(token=hf_token)
        self.model_name = model_name
        self._dimension: int | None = None

        # Quick probe to discover dimension and verify the model is available
        probe = self._call_api(["hello"])
        self._dimension = probe.shape[1]

        logger.info(
            "HFInferenceEmbedder ready: model=%s, dim=%d",
            model_name,
            self._dimension,
        )

    def _call_api(self, texts: list[str], retries: int = 3) -> np.ndarray:
        """Call the HF feature-extraction endpoint with retry/backoff."""
        last_err = None
        for attempt in range(retries):
            try:
                result = self.client.feature_extraction(
                    texts, model=self.model_name,
                )
                arr = np.asarray(result, dtype=np.float32)
                # The API may return (N, dim) or (N, seq_len, dim).
                # If 3-D, mean-pool over seq_len axis.
                if arr.ndim == 3:
                    arr = arr.mean(axis=1)
                # L2 normalize
                norms = np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-9)
                arr = arr / norms
                return arr
            except Exception as e:
                last_err = e
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "HF Inference API attempt %d failed: %s — retrying in %ds",
                        attempt + 1, e, wait,
                    )
                    time.sleep(wait)
        raise RuntimeError(f"HF Inference API failed after {retries} attempts: {last_err}")

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode texts via the HF Inference API in batches.

        Parameters
        ----------
        texts : list[str]
            Input texts to embed.
        batch_size : int
            Texts per API call (HF limit is ~128 for most models).

        Returns
        -------
        np.ndarray  shape (N, dim), L2-normalized.
        """
        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = self._call_api(batch)
            all_embeddings.append(emb)

        return np.vstack(all_embeddings).astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text, returning a 1-D array."""
        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension or 0


class ONNXEmbedder:
    """Dense embeddings via an ONNX model downloaded from HuggingFace."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str | None = None,
    ):
        """
        Download ONNX model + tokenizer from HuggingFace and create an
        inference session.

        Parameters
        ----------
        model_name : str
            HuggingFace repo id (must contain onnx/model.onnx and tokenizer.json).
        cache_dir : str | None
            Local cache directory for downloaded files.
        """
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        import onnxruntime as ort

        hf_token = _get_hf_token()

        # Download required files
        onnx_path = hf_hub_download(
            repo_id=model_name,
            filename="onnx/model.onnx",
            cache_dir=cache_dir,
            token=hf_token,
        )
        tokenizer_path = hf_hub_download(
            repo_id=model_name,
            filename="tokenizer.json",
            cache_dir=cache_dir,
            token=hf_token,
        )

        # Create ONNX inference session (CPU only)
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )

        # Create tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        # Discover which inputs the model expects
        self._input_names = [inp.name for inp in self.session.get_inputs()]

        # Determine hidden dimension from model output shape
        output_shape = self.session.get_outputs()[0].shape
        # Shape is typically [batch, seq_len, hidden_dim]
        self._dimension: int = output_shape[-1] if output_shape[-1] is not None else 384
        self.model_name = model_name

        logger.info(
            "ONNXEmbedder ready: model=%s, dim=%d, inputs=%s",
            model_name,
            self._dimension,
            self._input_names,
        )

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts into L2-normalized embeddings.

        Parameters
        ----------
        texts : list[str]
            Input texts to embed.
        batch_size : int
            Number of texts to process at once.

        Returns
        -------
        np.ndarray
            Shape (N, hidden_dim) with L2-normalized rows.
        """
        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encodings = self.tokenizer.encode_batch(batch_texts)

            # Build numpy arrays from tokenizer output
            ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
            attention_mask = np.array(
                [enc.attention_mask for enc in encodings], dtype=np.int64
            )

            # Prepare feed dict with only the inputs the model expects
            feed: dict[str, np.ndarray] = {}
            if "input_ids" in self._input_names:
                feed["input_ids"] = ids
            if "attention_mask" in self._input_names:
                feed["attention_mask"] = attention_mask
            if "token_type_ids" in self._input_names:
                token_type_ids = np.array(
                    [enc.type_ids for enc in encodings], dtype=np.int64
                )
                feed["token_type_ids"] = token_type_ids

            # Run inference — outputs[0] is token embeddings (batch, seq_len, hidden_dim)
            outputs = self.session.run(None, feed)
            token_embeddings = outputs[0]  # (batch, seq_len, hidden_dim)

            # Mean pooling with attention mask
            mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
            summed = (token_embeddings * mask_expanded).sum(axis=1)
            counts = mask_expanded.sum(axis=1).clip(min=1e-9)
            embeddings = summed / counts  # (batch, hidden_dim)

            # L2 normalization
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-9)
            embeddings = embeddings / norms

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings).astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text, returning a 1-D array of shape (hidden_dim,)."""
        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (e.g., 384 for bge-small-en-v1.5)."""
        return self._dimension


class TFIDFEmbedder:
    """Sparse TF-IDF fallback embedder (no external model required)."""

    def __init__(self):
        """Create a TF-IDF vectorizer with useful defaults."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.model_name = "tfidf-10k-unigram-bigram"
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            max_features=10000,
            ngram_range=(1, 2),
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> "TFIDFEmbedder":
        """Fit the vectorizer on a corpus of texts."""
        self.vectorizer.fit(texts)
        self._fitted = True
        logger.info(
            "TFIDFEmbedder fitted: vocabulary size=%d",
            len(self.vectorizer.vocabulary_),
        )
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Transform texts into TF-IDF vectors (dense numpy array).

        If the vectorizer hasn't been fitted yet, it will fit_transform on
        the provided texts (useful for one-shot usage).
        """
        if not self._fitted:
            warnings.warn(
                "TFIDFEmbedder.encode() called before fit(); fitting on input texts.",
                stacklevel=2,
            )
            matrix = self.vectorizer.fit_transform(texts)
            self._fitted = True
        else:
            matrix = self.vectorizer.transform(texts)

        return np.asarray(matrix.todense(), dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text, returning a 1-D array."""
        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        """Return the feature count after fitting."""
        if not self._fitted:
            return 0
        return len(self.vectorizer.vocabulary_)


def create_embedder(
    model_name: str | None = None,
    cache_dir: str | None = None,
) -> HFInferenceEmbedder | ONNXEmbedder | TFIDFEmbedder:
    """
    Factory: try HF Inference API first, then ONNX models, then TF-IDF.

    Cascade order:
      1. HFInferenceEmbedder (remote, fast, supports large models)
      2. ONNXEmbedder for each model in [primary] + fallbacks (local CPU)
      3. TFIDFEmbedder (sparse, no model download)

    Parameters
    ----------
    model_name : str | None
        Override the primary model. If None, uses config.EMBEDDING_MODEL.
    cache_dir : str | None
        Cache directory for ONNX model downloads.

    Returns
    -------
    HFInferenceEmbedder, ONNXEmbedder, or TFIDFEmbedder
    """
    from config import EMBEDDING_MODEL, EMBEDDING_MODEL_FALLBACKS

    primary = model_name or EMBEDDING_MODEL

    # 1. Try HF Inference API (remote — works well even for large models)
    try:
        embedder = HFInferenceEmbedder(model_name=primary)
        logger.info("Using HF Inference API embedder: %s (%dd)", primary, embedder.dimension)
        return embedder
    except Exception as e:
        logger.warning("HFInferenceEmbedder failed for %s: %s", primary, e)

    # 2. Try local ONNX models in cascade
    models_to_try = [primary] + EMBEDDING_MODEL_FALLBACKS

    for name in models_to_try:
        try:
            embedder = ONNXEmbedder(model_name=name, cache_dir=cache_dir)
            logger.info("Using ONNX embedding model: %s (%dd)", name, embedder.dimension)
            return embedder
        except Exception as e:
            logger.warning("ONNXEmbedder failed for %s: %s", name, e)
            continue

    # 3. Last resort
    warnings.warn(
        "All dense embedding models failed; falling back to TFIDFEmbedder. "
        "Quality will be lower — set HG_TOKEN for Inference API, or install "
        "onnxruntime + tokenizers for local ONNX.",
        stacklevel=2,
    )
    return TFIDFEmbedder()
