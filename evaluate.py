#!/usr/bin/env python
"""
evaluate.py — Évaluation comparative des 3 stratégies de recherche.

Métriques: Recall@3, MRR@3
Export: results.csv

Usage:
    python evaluate.py
    python evaluate.py --modes vector hybrid rerank
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import config
from src.retriever import search


def load_gold(path: str = "gold_questions.json") -> list[dict]:
    """Charge le fichier de questions gold."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_at_k(retrieved_docs: list[str], expected_docs: list[str], k: int = 3) -> float:
    """Recall@k: proportion de documents attendus retrouvés dans le top-k."""
    retrieved_set = set(retrieved_docs[:k])
    expected_set = set(expected_docs)
    if not expected_set:
        return 0.0
    return len(retrieved_set & expected_set) / len(expected_set)


def mrr_at_k(retrieved_docs: list[str], expected_docs: list[str], k: int = 3) -> float:
    """MRR@k: rang réciproque du premier document pertinent dans le top-k."""
    expected_set = set(expected_docs)
    for i, doc in enumerate(retrieved_docs[:k]):
        if doc in expected_set:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_mode(gold: list[dict], mode: str, top_k: int = 3) -> list[dict]:
    """Évalue un mode de recherche sur toutes les questions gold."""
    results = []
    for item in gold:
        question = item["question"]
        expected = item["expected_docs"]

        try:
            search_results = search(question, mode=mode, top_k=top_k)
            retrieved = [r["id_document"] for r in search_results]
        except Exception as e:
            print(f"  ⚠ Erreur pour '{question}': {e}", file=sys.stderr)
            retrieved = []

        r_at_k = recall_at_k(retrieved, expected, k=top_k)
        m_at_k = mrr_at_k(retrieved, expected, k=top_k)

        results.append({
            "question": question,
            "mode": mode,
            "recall@3": r_at_k,
            "mrr@3": m_at_k,
            "retrieved": retrieved,
            "expected": expected,
        })

    return results


def print_summary(all_results: dict[str, list[dict]]) -> None:
    """Affiche un tableau comparatif."""
    print(f"\n{'='*70}")
    print(f"{'ÉVALUATION COMPARATIVE':^70}")
    print(f"{'='*70}")
    print(f"{'Mode':<12} {'Recall@3':>10} {'MRR@3':>10} {'Questions':>12}")
    print(f"{'-'*70}")

    for mode, results in all_results.items():
        avg_recall = sum(r["recall@3"] for r in results) / len(results) if results else 0
        avg_mrr = sum(r["mrr@3"] for r in results) / len(results) if results else 0
        print(f"{mode:<12} {avg_recall:>10.4f} {avg_mrr:>10.4f} {len(results):>12}")

    print(f"{'='*70}\n")


def export_csv(all_results: dict[str, list[dict]], output: str = "results.csv") -> None:
    """Exporte les résultats dans un CSV."""
    rows: list[dict] = []
    for mode, results in all_results.items():
        for r in results:
            rows.append({
                "mode": mode,
                "question": r["question"],
                "recall@3": r["recall@3"],
                "mrr@3": r["mrr@3"],
                "retrieved_1": r["retrieved"][0] if len(r["retrieved"]) > 0 else "",
                "retrieved_2": r["retrieved"][1] if len(r["retrieved"]) > 1 else "",
                "retrieved_3": r["retrieved"][2] if len(r["retrieved"]) > 2 else "",
            })

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Résultats exportés → {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation du module RAG")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["vector", "hybrid", "rerank"],
        default=["vector", "hybrid", "rerank"],
    )
    parser.add_argument("--gold", default="gold_questions.json")
    parser.add_argument("--output", default="results.csv")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    gold = load_gold(args.gold)
    print(f"▶ {len(gold)} questions gold chargées")

    all_results: dict[str, list[dict]] = {}
    for mode in args.modes:
        print(f"\n▶ Évaluation mode: {mode}...")
        all_results[mode] = evaluate_mode(gold, mode, top_k=args.top_k)

    print_summary(all_results)
    export_csv(all_results, args.output)


if __name__ == "__main__":
    main()
