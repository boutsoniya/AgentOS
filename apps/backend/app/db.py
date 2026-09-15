from __future__ import annotations

import os
from contextlib import contextmanager

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS documents (
    id uuid PRIMARY KEY,
    name text NOT NULL,
    content_type text NOT NULL,
    size_bytes integer NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS document_chunks (
    id uuid PRIMARY KEY,
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index integer NOT NULL,
    content text NOT NULL,
    embedding vector(1536),
    page integer,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS document_chunks_document_idx ON document_chunks(document_id, chunk_index);
CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw ON document_chunks USING hnsw (embedding vector_cosine_ops);
"""

def database_url() -> str | None:
    return os.getenv("DATABASE_URL")

def available() -> bool:
    return bool(database_url() and psycopg)

def initialize() -> None:
    if not available():
        return
    with psycopg.connect(database_url()) as conn:
        try:
            conn.execute(SCHEMA)
            conn.commit()
        except Exception:
            conn.rollback()
            # pgvector is optional in development. The API can continue in memory.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id uuid PRIMARY KEY,
                    name text NOT NULL,
                    content_type text NOT NULL,
                    size_bytes integer NOT NULL,
                    created_at timestamptz NOT NULL DEFAULT now()
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id uuid PRIMARY KEY,
                    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index integer NOT NULL,
                    content text NOT NULL,
                    page integer,
                    created_at timestamptz NOT NULL DEFAULT now()
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS document_chunks_document_idx ON document_chunks(document_id, chunk_index)")
            conn.commit()

@contextmanager
def connection():
    if not available():
        raise RuntimeError("DATABASE_URL and psycopg are required")
    with psycopg.connect(database_url()) as conn:
        yield conn
