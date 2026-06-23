"""
PostgreSQL + pgvector schema for Mnemozia.

Vector dims: 384 (intfloat/multilingual-e5-small)
"""

from __future__ import annotations

import os
from typing import Optional

import psycopg2
import psycopg2.extras

# ---------------------------------------------------------------------------
# Embedding model — lazy-loaded, so importing schema.py doesn't allocate GPU/RAM
# ---------------------------------------------------------------------------

_embedding_model = None
_EMBEDDING_DIMS = 384
_MODEL_NAME = "intfloat/multilingual-e5-small"

# Prefixes required by the e5 family for asymmetric search
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


def get_embedding_model():
    """Lazy singleton — loads the model once, only when first needed.

    Uses LanceDB's embedding registry (sentence-transformers).
    Falls back to direct SentenceTransformer if LanceDB is unavailable.
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    # Try LanceDB embedding registry first
    try:
        from lancedb.embeddings import EmbeddingFunctionRegistry
        registry = EmbeddingFunctionRegistry.get_instance()
        _embedding_model = registry.get("sentence-transformers").create(
            name=_MODEL_NAME
        )
        return _embedding_model
    except Exception:
        pass

    # Fallback: direct SentenceTransformer
    from sentence_transformers import SentenceTransformer
    _embedding_model = SentenceTransformer(_MODEL_NAME)
    return _embedding_model


def compute_embeddings(texts: list[str]) -> list[list[float]]:
    """Compute embeddings for a list of texts using the lazy-loaded model."""
    model = get_embedding_model()
    # Try LanceDB embedding registry API first
    if hasattr(model, "compute_query_embeddings"):
        return model.compute_query_embeddings(texts)
    # Direct SentenceTransformer API
    import numpy as np
    results = model.encode(texts)
    if isinstance(results, np.ndarray):
        return results.tolist()
    return [r.tolist() if hasattr(r, 'tolist') else list(r) for r in results]


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

    # Create the notes table with same schema as LanceDB version
    cur.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding vector(384),
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
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_notes_embedding
            ON notes USING hnsw (embedding vector_cosine_ops)
        """)
    else:
        # IVFFlat requires data, so only create after inserts
        pass

    cur.close()


def _now() -> str:
    """Return current timestamp string."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _uid() -> str:
    """Return a 12-character hex ID."""
    import uuid
    return uuid.uuid4().hex[:12]
