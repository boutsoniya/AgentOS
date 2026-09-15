from __future__ import annotations

import re
from typing import Any

from .embeddings import cosine_similarity, embed_texts
from .rag import Chunk, rank_chunks


async def hybrid_retrieve(question: str, chunks: list[Chunk], k: int = 6) -> list[dict[str, Any]]:
    """Combine lexical retrieval with embeddings when configured.

    The lexical path is always available; semantic ranking is an enhancement,
    so local development still works without an API key.
    """
    lexical = rank_chunks(question, chunks, k=max(k * 2, 12))
    candidate_map = {id(c): c for c in lexical}

    query_embedding = (await embed_texts([question]))
    if query_embedding:
        candidate_chunks = list(candidate_map.values())
        vectors = await embed_texts([c.text for c in candidate_chunks])
        scored = [
            (cosine_similarity(query_embedding[0], vector), chunk)
            for chunk, vector in zip(candidate_chunks, vectors)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[:k]
    else:
        selected = [(0.0, c) for c in lexical[:k]]

    return [
        {
            "id": f"E{i}",
            "source_type": "document",
            "source": chunk.source,
            "page": chunk.page,
            "chunk_index": chunk.index,
            "text": chunk.text,
            "retrieval_score": round(float(score), 4),
            "retrieval_method": "hybrid" if query_embedding else "lexical",
        }
        for i, (score, chunk) in enumerate(selected, 1)
    ]


def reciprocal_rank_fusion(lexical_ids: list[str], semantic_ids: list[str], k: int = 60) -> list[str]:
    """Small, deterministic RRF utility for future database-backed retrieval."""
    scores: dict[str, float] = {}
    for rank, item_id in enumerate(lexical_ids, 1):
        scores[item_id] = scores.get(item_id, 0.0) + 1 / (k + rank)
    for rank, item_id in enumerate(semantic_ids, 1):
        scores[item_id] = scores.get(item_id, 0.0) + 1 / (k + rank)
    return [item_id for item_id, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)]
