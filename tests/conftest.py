"""
Pytest configuration — mocks psycopg2 for unit tests.
Uses a real pg0 PostgreSQL instance when available (set PG0_TEST=1).
"""

import os
import sys
from unittest.mock import MagicMock, patch
from types import ModuleType

import pytest

# ── Build psycopg2 mock module tree ──

_psycopg2 = ModuleType("psycopg2")
_psycopg2.__version__ = "2.9.12"

# In-memory storage for mock database
_mock_db: dict[str, list[dict]] = {"notes": []}

class MockCursor:
    """Simulates a PostgreSQL cursor with RealDictCursor-like behavior."""

    def __init__(self):
        self._rows: list[dict] = []
        self._rowcount = 0
        self._query = ""
        self._params: list = []
        self._next_id = 0

    def execute(self, query: str, params=None):
        self._query = query
        self._params = params or []
        query_upper = query.upper().strip()

        if query_upper.startswith("CREATE EXTENSION") or query_upper.startswith("CREATE TABLE IF NOT EXISTS NOTES") or query_upper.startswith("CREATE TABLE"):
            return  # DDL is a no-op in mock

        if query_upper.startswith("CREATE INDEX"):
            return  # Index creation is a no-op

        if query_upper.startswith("INSERT INTO NOTES"):
            # params: (id, text, embedding, category, tags, language, confidence, ...)
            if params and len(params) > 1:
                row = {
                    "id": params[0],
                    "text": params[1],
                    "embedding": params[2] if len(params) > 2 else [],
                    "category": params[3] if len(params) > 3 else "general",
                    "tags": params[4] if len(params) > 4 else "",
                    "language": params[5] if len(params) > 5 else "auto",
                    "confidence": float(params[6]) if len(params) > 6 and params[6] is not None else 1.0,
                    "importance": float(params[7]) if len(params) > 7 and params[7] is not None else 0.5,
                    "ttl_days": int(params[8]) if len(params) > 8 and params[8] is not None else 0,
                    "source": params[9] if len(params) > 9 else "",
                    "source_detail": params[10] if len(params) > 10 else "",
                    "version": int(params[11]) if len(params) > 11 else 1,
                    "is_active": True,
                    "created_at": str(params[13]) if len(params) > 13 else "2026-06-20 12:00:00",
                    "updated_at": str(params[14]) if len(params) > 14 else "2026-06-20 12:00:00",
                    "supersedes": params[15] if len(params) > 15 else "",
                    "related_to": params[16] if len(params) > 16 else "",
                }
                _mock_db["notes"].append(row)
            self._rows = []
            self._rowcount = 1
            return

        if query_upper.startswith("UPDATE NOTES"):
            where_parts = query.upper().split("WHERE")[-1].strip() if "WHERE" in query.upper() else ""
            matched = []
            for row in _mock_db["notes"]:
                matches = True
                if "ID =" in where_parts and params:
                    id_idx = 0
                    if row.get("id") == params[id_idx]:
                        pass
                    else:
                        matches = False
                if "VERSION =" in where_parts:
                    ver_idx = 1 if "ID =" in where_parts else 0
                    if params and len(params) > ver_idx:
                        if row.get("version") != params[ver_idx]:
                            matches = False
                if matches:
                    # Apply SET values
                    set_part = query.upper().split("SET")[1].split("WHERE")[0] if "WHERE" in query.upper() else ""
                    if "IS_ACTIVE = FALSE" in set_part or "is_active = false" in query.lower():
                        row["is_active"] = False
                    if "IS_ACTIVE = TRUE" in set_part or "is_active = true" in query.lower():
                        row["is_active"] = True
                    matched.append(row)
            self._rowcount = len(matched)
            self._rows = []
            return

        if query_upper.startswith("DELETE FROM NOTES"):
            self._rowcount = min(1, len(_mock_db["notes"]))
            _mock_db["notes"].clear()
            self._rows = []
            return

        if query_upper.startswith("SELECT"):
            self._rows = self._filter_rows(query, params)
            self._rowcount = len(self._rows)
            return

        self._rows = []
        self._rowcount = 0

    def _filter_rows(self, query: str, params: list) -> list[dict]:
        """Filter mock rows based on WHERE clause patterns."""
        rows = _mock_db.get("notes", [])
        query_upper = query.upper().strip()

        # Handle COUNT(*) queries
        if "COUNT(*)" in query_upper or "COUNT(*) AS" in query_upper:
            total = len(rows)
            return [RealDictRow({"total": total})]

        # Extract LIMIT
        import re
        limit_match = re.search(r"LIMIT\s+(\d+)", query_upper)
        limit = int(limit_match.group(1)) if limit_match else 100

        # Filter rows
        result = []
        for row in rows:
            # Check is_active filter
            if "IS_ACTIVE = TRUE" in query_upper or "IS_ACTIVE = TRUE\n" in query_upper:
                if not row.get("is_active"):
                    continue

            # Check id filter: "id = %s AND is_active = TRUE" or just "id = %s"
            if "ID = %S" in query_upper or "ID = %S " in query_upper:
                if params and params[0] == row.get("id"):
                    result.append(row)
                    if len(result) >= limit:
                        break
                    continue
                else:
                    continue

            # Check text filter for exact match: "text = %s AND is_active = TRUE"
            if "TEXT = %S" in query_upper:
                if params:
                    text_param = params[0]
                    if row.get("text") == text_param:
                        result.append(row)
                        if len(result) >= limit:
                            break
                    continue

            # Default: add if no specific filter matched
            result.append(row)
            if len(result) >= limit:
                break

        return result[:limit]

    def fetchone(self):
        if not self._rows:
            return None
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def close(self):
        pass


class MockConnection:
    """Simulates a psycopg2 connection. Reuses the same cursor."""

    def __init__(self, dsn=""):
        self.autocommit = True
        self.closed = 0
        self.dsn = dsn
        self.notices = []
        self._cursor = MockCursor()

    def cursor(self, cursor_factory=None):
        return self._cursor

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        self.closed = 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def mock_connect(dsn=None, **kwargs):
    return MockConnection(dsn)


_psycopg2.connect = mock_connect
_psycopg2.extensions = ModuleType("psycopg2.extensions")
_psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT = 0

# psycopg2.extras
_extras = ModuleType("psycopg2.extras")


class RealDictRow(dict):
    """Simulates psycopg2's RealDictRow — supports both key and integer index access."""
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            if isinstance(key, (int,)):
                vals = list(self.values())
                if 0 <= key < len(vals):
                    return vals[key]
            raise


class RealDictCursor:
    def __init__(self, *args, **kwargs):
        self.connection = args[0] if args else MagicMock()


_extras.RealDictRow = RealDictRow
_extras.RealDictCursor = RealDictCursor

_psycopg2.extras = _extras

# Register mocks
sys.modules["psycopg2"] = _psycopg2
sys.modules["psycopg2.extras"] = _extras
sys.modules["psycopg2.extensions"] = _psycopg2.extensions

# Mock lancedb.embeddings (used by schema.py for embedding registry fallback)
_lancedb = ModuleType("lancedb")
_lancedb.__version__ = "0.0.0"
_emb = ModuleType("lancedb.embeddings")
_emb.EmbeddingFunctionRegistry = MagicMock()
_lancedb.embeddings = _emb
sys.modules["lancedb"] = _lancedb
sys.modules["lancedb.embeddings"] = _emb

# Clean up mock DB before each session
@pytest.fixture(autouse=True)
def _clean_mock_db():
    _mock_db["notes"].clear()
    yield
