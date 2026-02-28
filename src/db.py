""""""

















































































































            return cur.fetchone()[0]            cur.execute("SELECT COUNT(*) FROM embeddings;")        with conn.cursor() as cur:    with get_connection() as conn:    """Nombre de lignes dans la table."""def count_embeddings() -> int:            return cur.fetchall()            cur.execute(sql, (vec_str, query_text, vec_str, top_k))            vec_str = "[" + ",".join(str(float(v)) for v in query_vec) + "]"        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:    with get_connection() as conn:    """        LIMIT %s;        ORDER BY vecteur <=> %s::vector        FROM embeddings               ts_rank_cd(tsv, plainto_tsquery('simple', %s)) AS score_fts               1 - (vecteur <=> %s::vector) AS score_vec,        SELECT id, id_document, section, texte_fragment, tokens,    sql = """    """Recherche hybride : vector + ts_rank_cd full-text."""def hybrid_search(query_vec, query_text: str, top_k: int = 20) -> list[dict]:            return cur.fetchall()            cur.execute(sql, (vec_str, vec_str, top_k))            vec_str = "[" + ",".join(str(float(v)) for v in query_vec) + "]"        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:    with get_connection() as conn:    """        LIMIT %s;        ORDER BY vecteur <=> %s::vector        FROM embeddings               1 - (vecteur <=> %s::vector) AS score_vec        SELECT id, id_document, section, texte_fragment, tokens,    sql = """    """Recherche vectorielle cosine via pgvector."""def vector_search(query_vec, top_k: int = 20) -> list[dict]:    return len(rows)            )                page_size=200,                template=template,                rows,                sql,                cur,            psycopg2.extras.execute_values(        with conn.cursor() as cur:    with get_connection() as conn:    template = "(%(id_document)s, %(section)s, %(texte_fragment)s, %(tokens)s, %(vecteur)s)"    """        VALUES %s        INSERT INTO embeddings (id_document, section, texte_fragment, tokens, vecteur)    sql = """    """    rows: [(id_document, section, texte_fragment, tokens, vecteur_np), ...]    Insère en batch.    """def insert_chunks(rows: list[tuple]) -> int:            cur.execute("TRUNCATE TABLE embeddings RESTART IDENTITY;")        with conn.cursor() as cur:    with get_connection() as conn:    """Vide la table embeddings."""def reset_table() -> None:            cur.execute(sql)        with conn.cursor() as cur:    with get_connection() as conn:    sql = sql_path.read_text(encoding="utf-8")    sql_path = config.BASE_DIR / "sql" / "init.sql"    """Exécute le script SQL d'initialisation."""def init_schema() -> None:        conn.close()    finally:        raise        conn.rollback()    except Exception:        conn.commit()        yield conn    try:    register_vector(conn)    )        password=config.DB_PASSWORD,        user=config.DB_USER,        dbname=config.DB_NAME,        port=config.DB_PORT,        host=config.DB_HOST,    conn = psycopg2.connect(    """Fournit une connexion PostgreSQL avec pgvector enregistré."""def get_connection() -> Generator:@contextmanagerimport configfrom pgvector.psycopg2 import register_vectorimport psycopg2.extrasimport psycopg2from typing import Generatorfrom contextlib import contextmanagerfrom __future__ import annotations"""Database helpers — connexion PostgreSQL + opérations pgvector.Database helper — connection pool + utility functions for pgvector.
"""
from __future__ import annotations

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

import config


def get_connection():
    """Return a new psycopg2 connection with pgvector registered."""
    conn = psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
    )
    register_vector(conn)
    return conn


def init_schema(conn) -> None:
    """Run the SQL init script to create tables / indexes / triggers."""
    sql_path = config.BASE_DIR / "sql" / "init.sql"
    with open(sql_path, "r", encoding="utf-8") as f:
        sql = f.read()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def reset_table(conn) -> None:
    """Truncate the embeddings table (used with --reset)."""
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE embeddings RESTART IDENTITY;")
    conn.commit()


def insert_chunks(conn, rows: list[dict]) -> int:
    """
    Batch-insert chunk rows into the embeddings table.
    Each row: {id_document, section, texte_fragment, tokens, vecteur}
    Returns the number of inserted rows.
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
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, values, page_size=200)
    conn.commit()
    return len(values)


def vector_search(conn, query_vec, limit: int = 20) -> list[dict]:
    """Pure vector cosine search."""
    sql = """
        SELECT id, id_document, section, texte_fragment, tokens,
               1 - (vecteur <=> %s::vector) AS score_vec
        FROM embeddings
        ORDER BY vecteur <=> %s::vector
        LIMIT %s
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (query_vec.tolist(), query_vec.tolist(), limit))
        return [dict(r) for r in cur.fetchall()]


def hybrid_search(conn, query_vec, query_text: str, limit: int = 20) -> list[dict]:
    """
    Vector + full-text hybrid search.
    Retrieves vector candidates then combines with ts_rank score.
    """
    sql = """
        WITH vec AS (
            SELECT id, id_document, section, texte_fragment, tokens,
                   1 - (vecteur <=> %s::vector) AS score_vec,
                   ts_rank_cd(tsv, plainto_tsquery('simple', %s)) AS score_bm25
            FROM embeddings
            ORDER BY vecteur <=> %s::vector
            LIMIT %s
        )
        SELECT *,
               0.7 * score_vec + 0.3 * (score_bm25 / GREATEST(MAX(score_bm25) OVER (), 1e-9)) AS score_hybrid
        FROM vec
        ORDER BY score_hybrid DESC
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (query_vec.tolist(), query_text, query_vec.tolist(), limit))
        return [dict(r) for r in cur.fetchall()]
