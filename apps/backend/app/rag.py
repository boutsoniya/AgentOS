"""RAG boundary for the next evidence-backed implementation phase."""

from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    source: str
    locator: str = ''


def rank_chunks(query: str, chunks: list[Chunk], k: int = 5) -> list[Chunk]:
    """Deterministic lexical baseline; replace with embeddings + reranking later."""
    terms = {t.lower() for t in query.split() if len(t) > 2}
    scored = []
    for chunk in chunks:
        text = chunk.text.lower()
        score = sum(1 for term in terms if term in text)
        scored.append((score, chunk))
    return [chunk for _, chunk in sorted(scored, key=lambda item: item[0], reverse=True)[:k]]
