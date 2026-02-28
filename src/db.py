"""
Database helpers - connexion PostgreSQL + operations pgvector.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

import config


@contextmanager
def get_connection() -> Generator:
    """Fournit une connexion PostgreSQL avec pgvector enregistre."""
    conn = psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
    )
    register_vector(conn)
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
    sql = sql_path.read_text(encoding="utf-8")
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


def vector_search(query_vec, top_k: int = 20) -> list[dict]:
    """Recherche vectorielle cosine via pgvector."""
    sql = """
        SELECT id, id_document, section, texte_fragment, tokens,
               1 - (vecteur <=> %s::vector) AS score_vec
        FROM embeddings
        ORDER BY vecteur <=> %s::vector
        LIMIT %s;
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            vec_str = "[" + ",".join(str(float(v)) for v in query_vec) + "]"
            cur.execute(sql, (vec_str, vec_str, top_k))
            return [dict(r) for r in cur.fetchall()]


def hybrid_search(query_vec, query_text: str, top_k: int = 20) -> list[dict]:
    """Recherche hybride : vector + ts_rank_cd full-text."""
    sql = """
        SELECT id, id_document, section, texte_fragment, tokens,
               1 - (vecteur <=> %s::vector) AS score_vec,
               ts_rank_cd(tsv, plainto_tsquery('simple', %s)) AS score_fts
        FROM embeddings
        ORDER BY vecteur <=> %s::vector
        LIMIT %s;
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            vec_str = "[" + ",".join(str(float(v)) for v in query_vec) + "]"
            cur.execute(sql, (vec_str, query_text, vec_str, top_k))
            return [dict(r) for r in cur.fetchall()]


def count_embeddings() -> int:
    """Nombre de lignes dans la table."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM embeddings;")
            return cur.fetchone()[0]
