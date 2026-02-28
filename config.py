"""
Configuration centralisée — lit .env puis expose des constantes.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Database ──────────────────────────────────────────────────────────────────
DB_HOST: str = os.getenv("DB_HOST", "localhost")
DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
DB_NAME: str = os.getenv("DB_NAME", "rag_enzymes")
DB_USER: str = os.getenv("DB_USER", "postgres")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "postgres")

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM: int = int(os.getenv("EMBED_DIM", "384"))

# ── Reranker ──────────────────────────────────────────────────────────────────
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOPK_CANDIDATES: int = int(os.getenv("TOPK_CANDIDATES", "20"))
TOPK_FINAL: int = int(os.getenv("TOPK_FINAL", "3"))

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent
DOCUMENTS_DIR: Path = BASE_DIR / os.getenv("DOCUMENTS_DIR", "documents_md")
