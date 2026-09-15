from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Chunk:
    index: int
    text: str
    source: str
    page: int | None = None


def extract_text(filename: str, content: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext in {".txt", ".md"}:
        return content.decode("utf-8", errors="replace")
    if ext == ".csv":
        rows = csv.reader(io.StringIO(content.decode("utf-8", errors="replace")))
        return "\n".join(" | ".join(row) for row in rows)
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if ext == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError("Unsupported document type")


def chunk_text(text: str, source: str, size: int = 900, overlap: int = 120) -> list[Chunk]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(clean):
        end = min(start + size, len(clean))
        if end < len(clean):
            boundary = clean.rfind(" ", start, end)
            if boundary > start + size // 2:
                end = boundary
        chunks.append(Chunk(index=index, text=clean[start:end], source=source))
        index += 1
        if end >= len(clean):
            break
        start = max(0, end - overlap)
    return chunks


def rank_chunks(query: str, chunks: list[Chunk], k: int = 5) -> list[Chunk]:
    terms = {t.lower() for t in re.findall(r"[a-zA-Z0-9]{3,}", query)}
    scored = []
    for chunk in chunks:
        words = set(re.findall(r"[a-zA-Z0-9]{3,}", chunk.text.lower()))
        score = len(terms & words)
        if score:
            scored.append((score, chunk))
    return [chunk for _, chunk in sorted(scored, key=lambda item: item[0], reverse=True)[:k]]
