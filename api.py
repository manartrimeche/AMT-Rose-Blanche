#!/usr/bin/env python
"""
api.py — API FastAPI pour la recherche sémantique RAG.

Usage:
    uvicorn api:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.retriever import search, SearchMode
from src import db

app = FastAPI(
    title="RAG Enzymes — Recherche Sémantique",
    description="API de recherche sémantique sur les fiches techniques enzymes BVZyme.",
    version="1.0.0",
)


# ── Modèles Pydantic ─────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Question de recherche")
    mode: SearchMode = Field("hybrid", description="Stratégie: vector | hybrid | rerank")
    top_k: int = Field(3, ge=1, le=20, description="Nombre de résultats")


class SearchResultItem(BaseModel):
    id_document: str
    section: str
    texte_fragment: str
    score_final: float
    score_vec: float | None = None
    score_bm25: float | None = None
    score_rerank: float | None = None


class SearchResponse(BaseModel):
    question: str
    mode: str
    results: list[SearchResultItem]
    count: int


class HealthResponse(BaseModel):
    status: str
    embeddings_count: int | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/search", response_model=SearchResponse)
def api_search(req: SearchRequest):
    """Recherche sémantique dans la base d'enzymes."""
    try:
        results = search(
            question=req.question,
            mode=req.mode,
            top_k=req.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return SearchResponse(
        question=req.question,
        mode=req.mode,
        results=results,
        count=len(results),
    )


@app.get("/health", response_model=HealthResponse)
def api_health():
    """Vérification de la santé de l'API."""
    try:
        count = db.count_embeddings()
        return HealthResponse(status="ok", embeddings_count=count)
    except Exception:
        return HealthResponse(status="ok (db non disponible)")
