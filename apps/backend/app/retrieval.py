from __future__ import annotations

from typing import Any

from .db import available as db_available, connection
from .embeddings import cosine_similarity, embed_texts
from .rag import Chunk, rank_chunks


def _db_semantic_retrieve(query_vector: list[float], k: int) -> list[dict[str, Any]]:
    """Retrieve persisted chunks with pgvector cosine distance."""
    if not db_available():
        return []
    literal = "[" + ",".join(format(value, ".8g") for value in query_vector) + "]"
    with connection() as conn:
        rows = conn.execute(
            """SELECT dc.id, dc.chunk_index, dc.content, dc.page, d.name,
                      1 - (dc.embedding <=> %s::vector) AS similarity
               FROM document_chunks dc
               JOIN documents d ON d.id = dc.document_id
              WHERE dc.embedding IS NOT NULL
              ORDER BY dc.embedding <=> %s::vector
              LIMIT %s""",
            (literal, literal, k),
        ).fetchall()
    return [
        {
            "id": f"DB{i}",
            "source_type": "document",
            "source": row[4],
            "page": row[3],
            "chunk_index": row[1],
            "text": row[2],
            "retrieval_score": round(float(row[5]), 4),
            "retrieval_method": "pgvector",
            "persisted_chunk_id": str(row[0]),
        }
        for i, row in enumerate(rows, 1)
    ]


async def hybrid_retrieve(question: str, chunks: list[Chunk], k: int = 6) -> list[dict[str, Any]]:
    """Prefer persisted pgvector retrieval, with a safe in-memory hybrid fallback."""
    query_embedding = await embed_texts([question])
    if query_embedding:
        try:
            persisted = await __import__("asyncio").to_thread(_db_semantic_retrieve, query_embedding[0], k)
            if persisted:
                return persisted
        except Exception:
            # A missing/unavailable pgvector index must not take down research.
            pass

    lexical = rank_chunks(question, chunks, k=max(k * 2, 12))
    if query_embedding and lexical:
        vectors = await embed_texts([c.text for c in lexical])
        scored = sorted(
            ((cosine_similarity(query_embedding[0], vector), chunk) for chunk, vector in zip(lexical, vectors)),
            key=lambda item: item[0],
            reverse=True,
        )[:k]
    else:
        scored = [(0.0, c) for c in lexical[:k]]

    return [
        {
            "id": f"E{i}",
            "source_type": "document",
            "source": chunk.source,
            "page": chunk.page,
            "chunk_index": chunk.index,
            "text": chunk.text,
            "retrieval_score": round(float(score), 4),
            "retrieval_method": "hybrid-memory" if query_embedding else "lexical",
        }
        for i, (score, chunk) in enumerate(scored, 1)
    ]


def reciprocal_rank_fusion(lexical_ids: list[str], semantic_ids: list[str], k: int = 60) -> list[str]:
    """Deterministic RRF utility for combining independent rankings."""
    scores: dict[str, float] = {}
    for rank, item_id in enumerate(lexical_ids, 1):
        scores[item_id] = scores.get(item_id, 0.0) + 1 / (k + rank)
    for rank, item_id in enumerate(semantic_ids, 1):
        scores[item_id] = scores.get(item_id, 0.0) + 1 / (k + rank)
    return [item_id for item_id, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)]
