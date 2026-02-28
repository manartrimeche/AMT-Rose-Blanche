"""
Database helpers - connexion PostgreSQL + operations vectorielles.
Stocke les vecteurs comme double precision[] et delegue
le calcul de similarite cosine a numpy (cote Python).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import numpy as np
import psycopg2
import psycopg2.extras

import config


@contextmanager
def get_connection() -> Generator:
    """Fournit une connexion PostgreSQL."""
    conn = psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema() -> None:
    """Execute le script SQL d initialisation."""
    sql_path = config.BASE_DIR / "sql" / "init.sql"
    sql = sql_path.read_text(encoding="utf-8-sig")  # utf-8-sig strips BOM
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)


def reset_table() -> None:
    """Vide la table embeddings."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE embeddings RESTART IDENTITY;")


def insert_chunks(rows: list[dict]) -> int:
    """
    Insere en batch.
    rows: list of dicts with keys: id_document, section, texte_fragment, tokens, vecteur
    """
    if not rows:
        return 0
    sql = """
        INSERT INTO embeddings (id_document, section, texte_fragment, tokens, vecteur)
        VALUES %s
    """
    values = [
        (
            r["id_document"],
            r["section"],
            r["texte_fragment"],
            r["tokens"],
            r["vecteur"],
        )
        for r in rows
    ]
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, values, page_size=200)
    return len(values)


def fetch_all_embeddings() -> list[dict]:
    """Charge tous les fragments avec leurs vecteurs pour la recherche cote Python."""
    sql = """
        SELECT id, id_document, section, texte_fragment, tokens, vecteur
        FROM embeddings;
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            return [dict(r) for r in cur.fetchall()]


def fetch_with_fts(query_text: str, top_k: int = 200) -> list[dict]:
    """Charge les fragments avec score full-text pour recherche hybride."""
    sql = """
        SELECT id, id_document, section, texte_fragment, tokens, vecteur,
               ts_rank_cd(tsv, plainto_tsquery('simple', %s)) AS score_fts
        FROM embeddings
        ORDER BY ts_rank_cd(tsv, plainto_tsquery('simple', %s)) DESC
        LIMIT %s;
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (query_text, query_text, top_k))
            return [dict(r) for r in cur.fetchall()]


def vector_search(query_vec, top_k: int = 20) -> list[dict]:
    """Recherche vectorielle cosine calculee en Python avec numpy."""
    rows = fetch_all_embeddings()
    if not rows:
        return []

    # Construire la matrice de vecteurs
    db_vecs = np.array([r["vecteur"] for r in rows], dtype=np.float32)
    q = np.array(query_vec, dtype=np.float32)

    # Cosine similarity (vecteurs deja normalises par l embedder)
    scores = db_vecs @ q

    # Trier par score decroissant
    ranked_idx = np.argsort(-scores)[:top_k]

    results = []
    for idx in ranked_idx:
        r = rows[int(idx)]
        results.append({
            "id": r["id"],
            "id_document": r["id_document"],
            "section": r["section"],
            "texte_fragment": r["texte_fragment"],
            "tokens": r["tokens"],
            "score_vec": float(scores[int(idx)]),
        })
    return results


def hybrid_search(query_vec, query_text: str, top_k: int = 20) -> list[dict]:
    """Recherche hybride : vector (numpy) + full-text (PostgreSQL)."""
    rows = fetch_all_embeddings()
    if not rows:
        return []

    # Cosine similarity
    db_vecs = np.array([r["vecteur"] for r in rows], dtype=np.float32)
    q = np.array(query_vec, dtype=np.float32)
    vec_scores = db_vecs @ q

    # Full-text scores via PostgreSQL
    fts_map: dict[int, float] = {}
    fts_rows = fetch_with_fts(query_text, top_k=len(rows))
    for fr in fts_rows:
        fts_map[fr["id"]] = float(fr["score_fts"])

    # Combiner
    results = []
    for i, r in enumerate(rows):
        results.append({
            "id": r["id"],
            "id_document": r["id_document"],
            "section": r["section"],
            "texte_fragment": r["texte_fragment"],
            "tokens": r["tokens"],
            "score_vec": float(vec_scores[i]),
            "score_fts": fts_map.get(r["id"], 0.0),
        })

    # Trier par score vectoriel, prendre top_k
    results.sort(key=lambda x: x["score_vec"], reverse=True)
    return results[:top_k]


def count_embeddings() -> int:
    """Nombre de lignes dans la table."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM embeddings;")
            return cur.fetchone()[0]
