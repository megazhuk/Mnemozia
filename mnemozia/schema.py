"""
PostgreSQL + pgvector schema for Mnemozia.

Vector dims: 1024 (Qwen3-Embedding-0.6B via llama.cpp)
"""

from __future__ import annotations

import os
from typing import Optional

import psycopg2
import psycopg2.extras

# ---------------------------------------------------------------------------
# Embedding via llama-server HTTP API
# ---------------------------------------------------------------------------

_EMBEDDING_DIMS = 1024
_LLAMA_SERVER_URL = os.environ.get(
    "MNEMOZIA_EMBED_URL",
    "http://127.0.0.1:18080/embedding",
)


def compute_embeddings(texts: list[str]) -> list[list[float]]:
    """Compute embeddings for a list of texts via llama-server.

    Each text is sent as a separate POST to the llama-server /embedding endpoint.
    Uses prompt caching on the server side — repeated texts are near-instant.
    """
    import requests

    results = []
    for text in texts:
        resp = requests.post(
            _LLAMA_SERVER_URL,
            json={"content": text},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        # Response: [{ "index": 0, "embedding": [[vec...]] }]
        emb = data[0]["embedding"][0]
        results.append(emb)
    return results


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

_DEFAULT_URI = os.environ.get(
    "Mnemozia_URI",
    "postgresql://postgres:postgres@127.0.0.1:5432/postgres",
)


def connect(db_uri: Optional[str] = None):
    """Connect to PostgreSQL. Falls back to env var or default pg0 URI."""
    uri = db_uri or _DEFAULT_URI
    conn = psycopg2.connect(uri)
    conn.autocommit = True
    return conn


def ensure_schema(conn) -> None:
    """Create pgvector extension and notes table if they don't exist."""
    cur = conn.cursor()

    # Enable pgvector
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create the notes table (1024-dim for Qwen3 embeddings)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS notes (
            id TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding vector({_EMBEDDING_DIMS}),
            category TEXT DEFAULT 'general',
            tags TEXT DEFAULT '',
            language TEXT DEFAULT 'auto',
            confidence DOUBLE PRECISION DEFAULT 1.0,
            source TEXT DEFAULT '',
            source_detail TEXT DEFAULT '',
            version INTEGER DEFAULT 1,
            is_active BOOLEAN DEFAULT TRUE,
            ttl_days INTEGER DEFAULT 0,
            importance DOUBLE PRECISION DEFAULT 0.5,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            supersedes TEXT DEFAULT '',
            related_to TEXT DEFAULT '',
            PRIMARY KEY (id, version)
        )
    """)

    # Create indexes for fast filtering
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_notes_is_active
        ON notes (is_active)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_notes_category
        ON notes (category)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_notes_id
        ON notes (id)
    """)
    # HNSW index for fast vector search (only if table has data to index)
    cur.execute("""
        SELECT COUNT(*) FROM notes
    """)
    count = cur.fetchone()[0]
    if count > 1000:
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_notes_embedding
            ON notes USING hnsw (embedding vector_cosine_ops)
        """)

    cur.close()


def _now() -> str:
    """Return current timestamp string."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _uid() -> str:
    """Return a 12-character hex ID."""
    import uuid
    return uuid.uuid4().hex[:12]
