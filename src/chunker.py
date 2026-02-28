"""
Smart Markdown Chunker — découpe les fiches techniques en fragments RAG-ready.
"""
from __future__ import annotations

import re
from pathlib import Path


# ── Patterns de nettoyage ─────────────────────────────────────────────────────
_RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_PHONE = re.compile(r"(?:\+?\d[\d\s\-().]{7,}\d)")
_RE_ADDRESS = re.compile(
    r"(?:No\.?\s?\d+|Stresemann\s+str)[^,\n]*(?:,\s*[^,\n]+){0,4}",
    re.IGNORECASE,
)
_RE_MULTI_DASH = re.compile(r"-{3,}")
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")

# Unités à normaliser
_UNIT_MAP = {
    "ppm": "ppm",
    "u/g": "U/g",
    "xylh/g": "XylH/g",
    "fau/g": "FAU/g",
    "skb/g": "SKB/g",
    "agi/g": "AGI/g",
    "ufc/g": "UFC/g",
    "mg/kg": "mg/kg",
}

MAX_CHUNK_CHARS = 1200


def _clean_text(text: str) -> str:
    """Supprime emails, URLs, téléphones, adresses, tirets décoratifs."""
    text = _RE_EMAIL.sub("", text)
    text = _RE_URL.sub("", text)
    text = _RE_PHONE.sub("", text)
    text = _RE_ADDRESS.sub("", text)
    text = _RE_MULTI_DASH.sub("", text)
    text = _RE_MULTI_SPACE.sub(" ", text)
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def _normalize_units(text: str) -> str:
    """Normalise les unités courantes (casse cohérente)."""
    def _repl(m: re.Match) -> str:
        return _UNIT_MAP.get(m.group(0).lower(), m.group(0))

    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in _UNIT_MAP) + r")\b",
        re.IGNORECASE,
    )
    return pattern.sub(_repl, text)


def _approx_tokens(text: str) -> int:
    """Approxime le nombre de tokens (split sur espaces)."""
    return len(text.split())


def _split_large_chunk(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Sous-découpe un texte trop long par paragraphes."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para.strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Si un seul paragraphe dépasse, on le coupe par phrases
            if len(para) > max_chars:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sub = ""
                for s in sentences:
                    attempt = (sub + " " + s).strip() if sub else s.strip()
                    if len(attempt) <= max_chars:
                        sub = attempt
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = s.strip()
                if sub:
                    current = sub
                else:
                    current = ""
            else:
                current = para.strip()

    if current:
        chunks.append(current)

    return chunks if chunks else [text[:max_chars]]


def chunk_markdown(file_path: str | Path) -> list[dict]:
    """
    Découpe un fichier Markdown structuré en chunks RAG-ready.

    Retourne une liste de dicts:
        {id_document, section, texte_fragment, tokens}
    """
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")

    # ── Extraire le titre produit (premier # ) ───────────────────────────────
    id_document = path.stem  # fallback
    lines = content.splitlines()
    for line in lines:
        if line.startswith("# ") and not line.startswith("## "):
            id_document = line.lstrip("# ").strip()
            break

    # ── Découper par sections ## ──────────────────────────────────────────────
    sections: list[tuple[str, str]] = []
    current_section = "Introduction"
    current_lines: list[str] = []

    for line in lines:
        if line.startswith("## "):
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_section, body))
            current_section = line.lstrip("# ").strip()
            current_lines = []
        elif line.startswith("# ") and not line.startswith("## "):
            # Skip le titre produit (déjà capturé)
            continue
        else:
            current_lines.append(line)

    # Flush dernière section
    body = "\n".join(current_lines).strip()
    if body:
        sections.append((current_section, body))

    # ── Nettoyage + chunking ──────────────────────────────────────────────────
    chunks: list[dict] = []

    for section_title, section_body in sections:
        cleaned = _clean_text(section_body)
        cleaned = _normalize_units(cleaned)
        if not cleaned or len(cleaned) < 10:
            continue

        sub_chunks = _split_large_chunk(cleaned, MAX_CHUNK_CHARS)

        for i, sub in enumerate(sub_chunks):
            # Préfixer par le contexte produit + section
            prefix = f"Produit: {id_document} | Section: {section_title}"
            if len(sub_chunks) > 1:
                prefix += f" (partie {i + 1}/{len(sub_chunks)})"

            fragment = f"{prefix}\n{sub}"
            chunks.append({
                "id_document": id_document,
                "section": section_title,
                "texte_fragment": fragment,
                "tokens": _approx_tokens(fragment),
            })

    return chunks


def chunk_all_documents(docs_dir: str | Path) -> list[dict]:
    """Chunke tous les .md d'un répertoire."""
    docs_dir = Path(docs_dir)
    all_chunks: list[dict] = []
    for md_file in sorted(docs_dir.glob("*.md")):
        all_chunks.extend(chunk_markdown(md_file))
    return all_chunks
