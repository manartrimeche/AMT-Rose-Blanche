#!/usr/bin/env python
"""
search.py — CLI de recherche sémantique.

Usage:
    python search.py "Quelle enzyme pour le pain de mie?"
    python search.py --mode hybrid "température optimale xylanase"
    python search.py --mode rerank --top-k 5 "dosage transglutaminase"
    python search.py --json "acide ascorbique"
"""
from __future__ import annotations

import argparse
import json
import sys

from src.retriever import search


def main() -> None:
    parser = argparse.ArgumentParser(description="Recherche sémantique RAG")
    parser.add_argument("question", type=str, help="Question de recherche")
    parser.add_argument(
        "--mode",
        choices=["vector", "hybrid", "rerank"],
        default="hybrid",
        help="Stratégie de recherche (défaut: hybrid)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Nombre de résultats (défaut: 3)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Sortie au format JSON",
    )
    args = parser.parse_args()

    results = search(args.question, mode=args.mode, top_k=args.top_k)

    if args.json_output:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    print(f"\n{'='*70}")
    print(f"Question : {args.question}")
    print(f"Mode     : {args.mode}")
    print(f"Résultats: {len(results)}")
    print(f"{'='*70}\n")

    for i, r in enumerate(results, 1):
        print(f"── Résultat {i} ──────────────────────────────────")
        print(f"  Document : {r['id_document']}")
        print(f"  Section  : {r['section']}")
        print(f"  Score    : {r['score_final']:.4f}", end="")
        if r.get("score_vec") is not None:
            print(f"  (vec={r['score_vec']:.4f}", end="")
            if r.get("score_bm25") is not None:
                print(f", bm25={r['score_bm25']:.2f}", end="")
            if r.get("score_rerank") is not None:
                print(f", rerank={r['score_rerank']:.4f}", end="")
            print(")", end="")
        print()
        # Texte tronqué
        text = r["texte_fragment"]
        if len(text) > 300:
            text = text[:300] + "..."
        print(f"  Texte    : {text}")
        print()


if __name__ == "__main__":
    main()
