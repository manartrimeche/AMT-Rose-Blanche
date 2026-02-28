"""
Embedder — encapsule le modèle sentence-transformers.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

import config

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Singleton du modèle d'embeddings."""
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode une liste de textes en vecteurs (float32)."""
    model = get_model()
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def embed_query(question: str) -> np.ndarray:
    """Encode une question unique (1-D array, normalized)."""
    model = get_model()
    vec = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vec[0]
