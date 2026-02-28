#!/usr/bin/env python
"""
ingest.py — Ingestion des documents Markdown dans PostgreSQL + pgvector.

Usage:
    python ingest.py                # ingestion complète
    python ingest.py --reset        # vide la table avant ingestion
    python ingest.py --dry-run      # chunking seul, sans base de données
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

import config
from src.chunker import chunk_all_documents
from src.embedder import embed_texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestion des documents Markdown")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Vide la table embeddings avant ingestion",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chunking + embedding sans écrire en base",
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=None,
        help="Répertoire source (défaut: config.DOCUMENTS_DIR)",
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir) if args.docs_dir else config.DOCUMENTS_DIR
    if not docs_dir.exists():
        print(f"✗ Répertoire introuvable: {docs_dir}", file=sys.stderr)
        sys.exit(1)

    md_files = sorted(docs_dir.glob("*.md"))
    print(f"▶ {len(md_files)} fichiers .md trouvés dans {docs_dir}")

    # ── Phase 1 : Chunking ────────────────────────────────────────────────────
    print("▶ Chunking en cours...")
    chunks = chunk_all_documents(docs_dir)
    print(f"  ✓ {len(chunks)} fragments générés")

    if not chunks:
        print("✗ Aucun fragment généré, arrêt.", file=sys.stderr)
        sys.exit(1)

    # Stats par document
    doc_counts: dict[str, int] = {}
    for c in chunks:
        doc_counts[c["id_document"]] = doc_counts.get(c["id_document"], 0) + 1
    print(f"  ✓ {len(doc_counts)} documents distincts")
    for doc, count in sorted(doc_counts.items()):
        print(f"    - {doc}: {count} fragments")

    # ── Phase 2 : Embedding ───────────────────────────────────────────────────
    print("\n▶ Embedding des fragments...")
    texts = [c["texte_fragment"] for c in chunks]
    vectors = embed_texts(texts, batch_size=64)
    print(f"  ✓ Matrice: {vectors.shape}")

    # Attacher les vecteurs aux chunks
    for i, chunk in enumerate(chunks):
        chunk["vecteur"] = vectors[i].tolist()

    if args.dry_run:
        print("\n✓ Dry-run terminé — aucune écriture en base.")
        print(f"  Fragments: {len(chunks)}")
        print(f"  Dimension: {vectors.shape[1]}")
        avg_tokens = sum(c["tokens"] for c in chunks) / len(chunks)
        print(f"  Tokens moyens/fragment: {avg_tokens:.0f}")
        return

    # ── Phase 3 : Database ────────────────────────────────────────────────────
    from src import db

    print("\n▶ Initialisation du schéma...")
    db.init_schema()

    if args.reset:
        print("▶ Reset de la table embeddings...")
        db.reset_table()

    print("▶ Insertion en base...")
    batch_size = 100
    inserted = 0
    for i in tqdm(range(0, len(chunks), batch_size), desc="Batches"):
        batch = chunks[i : i + batch_size]
        inserted += db.insert_chunks(batch)

    total = db.count_embeddings()
    print(f"\n✓ Ingestion terminée — {inserted} fragments insérés")
    print(f"  Total en base: {total}")


if __name__ == "__main__":
    main()
