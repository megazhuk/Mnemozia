"""
Integration tests for Mnemozia — run against a real pg0 PostgreSQL instance.

Prerequisites:
    pg0 installed and started: su - pg0user -c "pg0 start"
"""

from __future__ import annotations

import os
import re
from unittest.mock import MagicMock, patch

import pytest

from mnemozia.core import MnemoziaKB

# pg0 runs as pg0user on port 5432
P0_URI = os.environ.get(
    "PG0_URI",
    "postgresql://postgres:postgres@127.0.0.1:5432/postgres",
)

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("PG0_TEST"),
        reason="Set PG0_TEST=1 to run integration tests",
    ),
]


@pytest.fixture
def kb(request):
    """MnemoziaKB connected to real pg0. Cleans up after test."""
    _kb = MnemoziaKB(db_uri=P0_URI)

    # Mock embeddings (we don't want to load the real model for tests)
    _kb._embed = MagicMock(return_value=[0.1] * 384)
    _kb._embed_passage = MagicMock(return_value=[0.1] * 384)

    yield _kb

    # Cleanup: drop all test data
    cur = _kb.cur
    cur.execute("DELETE FROM notes")
    _kb.conn.commit()


class TestIntegrationAdd:
    def test_add_basic(self, kb):
        result = kb.execute({"action": "add", "text": "Test fact"})
        assert "✅ Stored" in result
        assert re.search(r"\[[a-f0-9]{12}\]", result)

    def test_add_missing_text(self, kb):
        result = kb.execute({"action": "add"})
        assert "text" in result.lower()

    def test_add_exact_duplicate(self, kb):
        r1 = kb.execute({"action": "add", "text": "Exact duplicate test"})
        assert "✅ Stored" in r1
        r2 = kb.execute({"action": "add", "text": "Exact duplicate test"})
        assert "Exact duplicate" in r2 or "Near-duplicate" in r2

    def test_add_with_all_fields(self, kb):
        result = kb.execute({
            "action": "add",
            "text": "Full test fact",
            "category": "devops/networking",
            "tags": "test,example",
            "confidence": 0.7,
            "importance": 0.8,
            "ttl_days": 30,
            "source": "user",
            "source_detail": "Manual test",
            "language": "en",
        })
        assert "✅ Stored" in result


class TestIntegrationSearch:
    def test_search_empty(self, kb):
        result = kb.execute({"action": "search", "query": "anything"})
        assert "No results" in result

    def test_search_after_add(self, kb):
        kb.execute({"action": "add", "text": "PostgreSQL port is 5432"})
        kb.execute({"action": "add", "text": "Nginx reload: nginx -s reload"})

        result = kb.execute({"action": "search", "query": "database port"})
        assert "PostgreSQL" in result or "5432" in result

    def test_search_all_modes(self, kb):
        kb.execute({"action": "add", "text": "Test fact for search modes"})
        for mode in ["semantic", "hybrid", "keyword"]:
            result = kb.execute({
                "action": "search", "query": "test", "mode": mode,
            })
            assert isinstance(result, str)


class TestIntegrationUpdate:
    def test_update_creates_new_version(self, kb):
        r1 = kb.execute({"action": "add", "text": "Original version"})
        fid = re.search(r"\[([a-f0-9]{12})\]", r1).group(1)
        r2 = kb.execute({"action": "update", "id": fid, "text": "Updated version"})
        assert "v2" in r2 or "Updated" in r2

    def test_update_no_change(self, kb):
        r1 = kb.execute({"action": "add", "text": "Same text"})
        fid = re.search(r"\[([a-f0-9]{12})\]", r1).group(1)
        r2 = kb.execute({"action": "update", "id": fid, "text": "Same text"})
        assert "unchanged" in r2 or "no change" in r2.lower() or "no update" in r2.lower()


class TestIntegrationDeactivate:
    def test_deactivate_reactivate(self, kb):
        r1 = kb.execute({"action": "add", "text": "To be toggled"})
        fid = re.search(r"\[([a-f0-9]{12})\]", r1).group(1)

        r2 = kb.execute({"action": "deactivate", "id": fid})
        assert "Archived" in r2 or "🗄" in r2

        r3 = kb.execute({"action": "reactivate", "id": fid})
        assert "Reactivated" in r3

    def test_deactivate_invalid(self, kb):
        result = kb.execute({"action": "deactivate", "id": "ghost"})
        assert "No active fact" in result


class TestIntegrationHistory:
    def test_history(self, kb):
        r1 = kb.execute({"action": "add", "text": "Version 1"})
        fid = re.search(r"\[([a-f0-9]{12})\]", r1).group(1)
        kb.execute({"action": "update", "id": fid, "text": "Version 2"})

        result = kb.execute({"action": "history", "id": fid})
        assert "Version 2" in result


class TestIntegrationStats:
    def test_stats(self, kb):
        kb.execute({"action": "add", "text": "Fact 1", "category": "devops"})
        kb.execute({"action": "add", "text": "Fact 2", "category": "personal"})

        result = kb.execute({"action": "stats"})
        assert "2" in result or "Stats" in result


class TestIntegrationExport:
    def test_export_markdown(self, kb):
        kb.execute({"action": "add", "text": "Export test"})
        result = kb.execute({"action": "export", "format": "markdown"})
        assert "Export test" in result

    def test_export_invalid_format(self, kb):
        kb.execute({"action": "add", "text": "Test"})
        result = kb.execute({"action": "export", "format": "csv"})
        assert "Unsupported" in result


class TestIntegrationVacuum:
    def test_vacuum(self, kb):
        result = kb.execute({"action": "vacuum", "older_than_days": 1})
        assert "removed" in result.lower() or "Vacuumed" in result


class TestIntegrationEdgeCases:
    def test_unicode(self, kb):
        result = kb.execute({
            "action": "add",
            "text": "Привет, Mnemozia! 汉字 español français",
        })
        assert "✅ Stored" in result

    def test_merge_self(self, kb):
        result = kb.execute({"action": "merge", "id": "a", "with": "a"})
        assert "itself" in result.lower()

    def test_split_too_few_parts(self, kb):
        r1 = kb.execute({"action": "add", "text": "Test"})
        fid = re.search(r"\[([a-f0-9]{12})\]", r1).group(1)
        result = kb.execute({"action": "split", "id": fid, "parts": "OnlyOne"})
        assert "at least 2" in result.lower()
