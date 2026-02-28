from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class ChunkRecord:
    chunk_id: int
    source_file: str
    section: str
    text: str


class SemanticRAG:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings: np.ndarray | None = None
        self.chunks: list[ChunkRecord] = []

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _iter_md_files(data_dir: Path) -> Iterable[Path]:
        yield from sorted(data_dir.glob("*.md"))

    @staticmethod
    def _split_md_sections(text: str) -> list[tuple[str, str]]:
        """Split a Markdown file into (section_title, section_body) pairs."""
        sections: list[tuple[str, str]] = []
        current_title = "Introduction"
        current_lines: list[str] = []

        for line in text.splitlines():
            if line.startswith("#"):
                # flush previous section
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append((current_title, body))
                current_title = line.lstrip("# ").strip()
                current_lines = []
            else:
                current_lines.append(line)

        # flush last section
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_title, body))

        return sections

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        words = text.split()
        if not words:
            return []

        chunks: list[str] = []
        step = max(1, chunk_size - chunk_overlap)
        for start in range(0, len(words), step):
            end = start + chunk_size
            chunk_words = words[start:end]
            if not chunk_words:
                continue
            chunks.append(" ".join(chunk_words))
            if end >= len(words):
                break
        return chunks

    def build_index(
        self,
        data_dir: Path,
        output_dir: Path,
        chunk_size: int = 180,
        chunk_overlap: int = 30,
    ) -> None:
        records: list[ChunkRecord] = []
        chunk_id = 0

        md_files = list(self._iter_md_files(data_dir))
        if not md_files:
            raise FileNotFoundError(f"Aucun fichier Markdown (.md) trouvé dans {data_dir}")

        for md_file in md_files:
            raw_content = md_file.read_text(encoding="utf-8")
            sections = self._split_md_sections(raw_content)
            for section_title, section_body in sections:
                text = self._normalize_text(section_body)
                if not text:
                    continue
                for chunk in self._chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
                    records.append(
                        ChunkRecord(
                            chunk_id=chunk_id,
                            source_file=md_file.name,
                            section=section_title,
                            text=chunk,
                        )
                    )
                    chunk_id += 1

        if not records:
            raise RuntimeError("Aucun texte exploitable extrait des fichiers Markdown.")

        texts = [record.text for record in records]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        embeddings = self._l2_normalize(embeddings)

        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "embeddings.npy", embeddings)

        with (output_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

        metadata = {
            "model_name": self.model_name,
            "num_chunks": len(records),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

        self.embeddings = embeddings
        self.chunks = records

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        return matrix / norms

    def load_index(self, output_dir: Path) -> None:
        embeddings_path = output_dir / "embeddings.npy"
        chunks_path = output_dir / "chunks.jsonl"
        metadata_path = output_dir / "metadata.json"

        if not embeddings_path.exists() or not chunks_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "Index incomplet: embeddings.npy, chunks.jsonl, metadata.json sont requis."
            )

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        stored_model_name = metadata.get("model_name")
        if stored_model_name and stored_model_name != self.model_name:
            self.model_name = stored_model_name
            self.model = SentenceTransformer(stored_model_name)

        self.embeddings = np.load(embeddings_path)
        self.chunks = []
        with chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                self.chunks.append(ChunkRecord(**payload))

    def query(self, question: str, top_k: int = 3) -> List[dict]:
        if self.embeddings is None or not self.chunks:
            raise RuntimeError("Index non chargé. Lance load_index() ou build_index() d'abord.")

        question_embedding = self.model.encode([question], convert_to_numpy=True)
        question_embedding = self._l2_normalize(question_embedding)[0]

        similarities = self.embeddings @ question_embedding
        k = min(top_k, len(self.chunks))
        top_indices = np.argpartition(-similarities, range(k))[:k]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        results = []
        for idx in top_indices:
            record = self.chunks[int(idx)]
            results.append(
                {
                    "text": record.text,
                    "score": float(similarities[int(idx)]),
                    "source_file": record.source_file,
                    "section": record.section,
                    "chunk_id": record.chunk_id,
                }
            )
        return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Module de recherche sémantique (RAG) sur un corpus de fichiers Markdown."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Construire la base vectorielle")
    index_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "enzymes",
        help="Dossier contenant les fichiers Markdown (.md)",
    )
    index_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("vector_store"),
        help="Dossier de sortie de la base vectorielle",
    )
    index_parser.add_argument(
        "--chunk-size",
        type=int,
        default=180,
        help="Taille d'un fragment (en mots)",
    )
    index_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=30,
        help="Chevauchement entre fragments (en mots)",
    )
    index_parser.add_argument(
        "--model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Modèle d'embedding sentence-transformers",
    )

    query_parser = subparsers.add_parser("query", help="Interroger la base vectorielle")
    query_parser.add_argument(
        "--question",
        required=True,
        help="Question en langage naturel",
    )
    query_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("vector_store"),
        help="Dossier de la base vectorielle",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Nombre de résultats à retourner",
    )
    query_parser.add_argument(
        "--model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Modèle d'embedding sentence-transformers",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    rag = SemanticRAG(model_name=args.model)

    if args.command == "index":
        rag.build_index(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print("Index créé avec succès.")
        print(f"Fragments indexés: {len(rag.chunks)}")
        print(f"Dossier de sortie: {args.output_dir}")
        return

    if args.command == "query":
        rag.load_index(args.output_dir)
        results = rag.query(args.question, top_k=args.top_k)

        print("\nTop résultats:\n")
        for i, result in enumerate(results, start=1):
            print(f"[{i}] score={result['score']:.4f} | source={result['source_file']} | section={result['section']}")
            print(result["text"])
            print("-" * 80)


if __name__ == "__main__":
    main()
