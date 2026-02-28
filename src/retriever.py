"""
Retriever — 3 stratégies de recherche: vector, hybrid, rerank.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import config
from src import db, embedder

# ── Reranker singleton ────────────────────────────────────────────────────────
_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(config.RERANK_MODEL)
    return _reranker


# ── Normalisation min-max ─────────────────────────────────────────────────────
def _min_max_norm(values: list[float]) -> list[float]:
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0
    return [(v - mn) / rng for v in values]


# ── A) Vector-only ────────────────────────────────────────────────────────────
def search_vector(question: str, top_k: int | None = None) -> list[dict]:
    """Recherche purement vectorielle (cosine via pgvector)."""
    top_k = top_k or config.TOPK_FINAL
    query_vec = embedder.embed_query(question)
    rows = db.vector_search(query_vec, top_k=top_k)
    results = []
    for r in rows:
        results.append({
            "id_document": r["id_document"],
            "section": r["section"],
            "texte_fragment": r["texte_fragment"],
            "score_final": float(r["score_vec"]),
            "score_vec": float(r["score_vec"]),
            "score_bm25": None,
            "score_rerank": None,
        })
    return results


# ── B) Hybrid (vector + BM25) ────────────────────────────────────────────────
def search_hybrid(question: str, top_k: int | None = None) -> list[dict]:
    """Hybride: score = 0.7 * cosine_norm + 0.3 * bm25_norm."""
    top_k = top_k or config.TOPK_FINAL
    candidates_k = config.TOPK_CANDIDATES
    query_vec = embedder.embed_query(question)

    rows = db.hybrid_search(query_vec, question, top_k=candidates_k)
    if not rows:
        return []

    # BM25 local sur les candidats
    corpus_texts = [r["texte_fragment"] for r in rows]
    tokenized = [t.lower().split() for t in corpus_texts]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(question.lower().split()).tolist()

    # Normalisation
    vec_scores = [float(r["score_vec"]) for r in rows]
    vec_norm = _min_max_norm(vec_scores)
    bm25_norm = _min_max_norm(bm25_scores)

    results = []
    for i, r in enumerate(rows):
        final = 0.7 * vec_norm[i] + 0.3 * bm25_norm[i]
        results.append({
            "id_document": r["id_document"],
            "section": r["section"],
            "texte_fragment": r["texte_fragment"],
            "score_final": final,
            "score_vec": vec_scores[i],
            "score_bm25": bm25_scores[i],
            "score_rerank": None,
        })

    results.sort(key=lambda x: x["score_final"], reverse=True)
    return results[:top_k]


# ── C) Hybrid + Rerank ───────────────────────────────────────────────────────
def search_rerank(question: str, top_k: int | None = None) -> list[dict]:
    """Hybride + cross-encoder reranking (stratégie recommandée)."""
    top_k = top_k or config.TOPK_FINAL
    candidates_k = config.TOPK_CANDIDATES

    # Phase 1 : récupérer candidats hybrides
    hybrid_results = search_hybrid(question, top_k=candidates_k)
    if not hybrid_results:
        return []

    # Phase 2 : rerank avec CrossEncoder
    reranker = _get_reranker()
    pairs = [(question, r["texte_fragment"]) for r in hybrid_results]
    rerank_scores = reranker.predict(pairs).tolist()

    for i, r in enumerate(hybrid_results):
        r["score_rerank"] = float(rerank_scores[i])
        # Score final pondéré
        r["score_final"] = 0.4 * r["score_vec"] + 0.6 * float(rerank_scores[i])

    hybrid_results.sort(key=lambda x: x["score_final"], reverse=True)
    return hybrid_results[:top_k]


# ── Dispatch ──────────────────────────────────────────────────────────────────
SearchMode = Literal["vector", "hybrid", "rerank"]


def search(
    question: str,
    mode: SearchMode = "rerank",
    top_k: int | None = None,
) -> list[dict]:
    """Point d'entrée unifié."""
    if mode == "vector":
        return search_vector(question, top_k)
    elif mode == "hybrid":
        return search_hybrid(question, top_k)
    elif mode == "rerank":
        return search_rerank(question, top_k)
    else:
        raise ValueError(f"Mode inconnu: {mode}")
