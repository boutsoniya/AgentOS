from __future__ import annotations

import os
from typing import Any

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    AsyncOpenAI = None


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Create embeddings when an OpenAI key is configured; otherwise return []."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or AsyncOpenAI is None or not texts:
        return []
    client = AsyncOpenAI(api_key=api_key)
    response = await client.embeddings.create(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        input=texts,
    )
    return [item.embedding for item in response.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
