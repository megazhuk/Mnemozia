"""
LanceDB schema and constants for Mnemozia.

Vector dims: 384 (intfloat/multilingual-e5-small)
"""

import lancedb
from lancedb.embeddings import EmbeddingFunctionRegistry
from lancedb.pydantic import LanceModel, Vector


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
    """Lazy singleton — loads the model once, only when first needed."""
    global _embedding_model
    if _embedding_model is None:
        registry = EmbeddingFunctionRegistry.get_instance()
        _embedding_model = registry.get("sentence-transformers").create(
            name=_MODEL_NAME
        )
    return _embedding_model


# ---------------------------------------------------------------------------
# Pydantic schema (LanceDB-native)
# ---------------------------------------------------------------------------

def _default_now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class NoteSchema(LanceModel):
    """A single versioned fact in the knowledge base.

    Vector is stored as a plain fixed-size list — we embed manually with
    e5 prefixes (passage: / query:) instead of relying on LanceDB auto-embedding.
    """

    # ── identity ──────────────────────────────────────────────────────
    id: str                    # shared across versions of the same fact
    text: str                  # the fact itself
    vector: Vector(_EMBEDDING_DIMS)  # manually embedded via _embed()

    # ── organisation ──────────────────────────────────────────────────
    category: str = "general"
    tags: str = ""             # comma-separated, e.g. "python,fastapi,orm"
    language: str = "auto"     # "ru", "en", "auto"

    # ── confidence & provenance ───────────────────────────────────────
    confidence: float = 1.0    # 0.0 (hypothesis) … 1.0 (verified fact)
    source: str = ""           # "user", "session:abc123", "url:https://..."
    source_detail: str = ""    # free-form: "From conversation on 2024-06-18"

    # ── lifecycle ─────────────────────────────────────────────────────
    version: int = 1
    is_active: bool = True
    ttl_days: int = 0          # 0 = permanent; >0 = auto-archive after N days
    importance: float = 0.5    # 0.0–1.0; boosts ranking in results

    # ── versioning ────────────────────────────────────────────────────
    created_at: str = _default_now()
    updated_at: str = _default_now()
    supersedes: str = ""       # comma-separated IDs this fact replaces (merge)

    # ── relationships ─────────────────────────────────────────────────
    related_to: str = ""       # comma-separated IDs of related facts


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def open_or_create_table(db_path: str, table_name: str = "notes"):
    """Connect to LanceDB and return the table, creating it if absent."""
    db = lancedb.connect(db_path)
    if table_name in db.table_names():
        tbl = db.open_table(table_name)
        # Ensure FTS index exists for hybrid/keyword search
        _ensure_indices(tbl)
        return tbl
    else:
        tbl = db.create_table(table_name, schema=NoteSchema)
        _ensure_indices(tbl)
        return tbl


def _ensure_indices(tbl):
    """Create scalar indices for fast filtering (idempotent).

    Note: LanceDB 0.33 does not support INVERTED (FTS) indices,
    so keyword search falls back to semantic search.
    """
    for col in ("is_active", "category", "id"):
        try:
            tbl.create_scalar_index(col, replace=False)
        except Exception:
            pass
