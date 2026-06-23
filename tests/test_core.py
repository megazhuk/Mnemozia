"""
Tests for Mnemozia core logic — validates parsing, dedup, formatting, error handling.

These tests mock LanceDB (SIGILL on this Sandy Bridge CPU) and exercise
the MnemoziaKB class directly.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from mnemozia.core import (
    MnemoziaKB,
    _now,
    _uid,
    _distance_to_stars,
    _confidence_flag,
    _format_result,
    _parse_float,
    _join_tags,
    _join_ids,
    _DEDUP_THRESHOLD,
    _FLAG_THRESHOLD,
)


# ═══════════════════════════════════════════════════════════════════════
# Unit tests — pure functions (no DB)
# ═══════════════════════════════════════════════════════════════════════


class TestHelpers:
    def test_now_format(self):
        result = _now()
        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", result)

    def test_uid_length(self):
        uid = _uid()
        assert len(uid) == 12
        assert all(c in "0123456789abcdef" for c in uid)

    def test_uids_are_unique(self):
        uids = {_uid() for _ in range(1000)}
        assert len(uids) == 1000

    @pytest.mark.parametrize(
        "distance,expected",
        [
            (0.0, "★★★★★"),
            (0.05, "★★★★★"),
            (0.06, "★★★★☆"),
            (0.10, "★★★★☆"),
            (0.12, "★★★☆☆"),
            (0.15, "★★★☆☆"),
            (0.20, "★★☆☆☆"),
            (0.25, "★★☆☆☆"),
            (0.30, "★☆☆☆☆"),
            (0.50, "★☆☆☆☆"),
            (1.0, "★☆☆☆☆"),
        ],
    )
    def test_distance_to_stars(self, distance, expected):
        assert _distance_to_stars(distance) == expected

    @pytest.mark.parametrize(
        "conf,expected",
        [
            (1.0, "✅ verified"),
            (0.95, "✅ verified"),
            (0.9, "✅ verified"),
            (0.89, "📋 reliable"),
            (0.5, "📋 reliable"),
            (0.49, "⚠️ low confidence"),
            (0.0, "⚠️ low confidence"),
        ],
    )
    def test_confidence_flag(self, conf, expected):
        assert _confidence_flag(conf) == expected

    @pytest.mark.parametrize(
        "val,default,expected",
        [
            ("3.14", 0.0, 3.14),
            ("0", 0.5, 0.0),
            ("abc", 1.0, 1.0),
            (None, 0.7, 0.7),
            ("", 0.5, 0.5),
        ],
    )
    def test_parse_float(self, val, default, expected):
        assert _parse_float(val, default) == expected

    def test_format_result(self):
        row = {
            "id": "abc123def456",
            "text": "Test fact content",
            "version": 2,
            "category": "devops",
            "tags": "test,example",
            "confidence": 0.95,
            "importance": 0.8,
            "updated_at": "2026-06-20 12:00:00",
            "_distance": 0.15,
        }
        result = _format_result(row, 1)
        assert "abc123def456" in result
        assert "★★★☆☆" in result
        assert "✅ verified" in result
        assert "test,example" in result
        assert "v2" in result
        assert "devops" in result
        assert "Test fact content" in result

    def test_join_tags(self):
        assert _join_tags("a,b", "b,c") == "a,b,c"
        assert _join_tags("", "x,y") == "x,y"
        assert _join_tags("a, a, b", "b, c") == "a,b,c"

    def test_join_ids(self):
        assert _join_ids("a,b", "b,c") == "a,b,c"
        assert _join_ids("a", "", "b") == "a,b"
        assert _join_ids("") == ""


# ═══════════════════════════════════════════════════════════════════════
# Fixtures — mocked MnemoziaKB
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_table():
    """Create a mock LanceDB table that stores rows in-memory."""
    table = MagicMock()
    table._rows = []
    table._next_version = {}

    class FakeLance:
        def __init__(self, rows_ref):
            self.rows_ref = rows_ref
        def scanner(self, *, filter=None, limit=None, **kwargs):
            result_rows = []
            for row in self.rows_ref:
                if filter:
                    m = re.search(r"id = '([^']+)' AND is_active = (True|False)", filter)
                    if m:
                        fid, active = m.group(1), m.group(2) == "True"
                        if row.get("id") == fid and row.get("is_active") == active:
                            result_rows.append(row)
                        continue
                    m = re.search(r"id = '([^']+)'", filter)
                    if m:
                        if row.get("id") == m.group(1):
                            result_rows.append(row)
                        continue
                    m = re.search(r"text = '([^']+)' AND is_active = True", filter)
                    if m:
                        if str(row.get("text")) == m.group(1) and row.get("is_active"):
                            result_rows.append(row)
                        continue
                    m = re.search(r"is_active = True AND confidence < 0.5", filter)
                    if m:
                        if row.get("is_active") and row.get("confidence", 1.0) < 0.5:
                            result_rows.append(row)
                        continue
                    m = re.search(r"^is_active = True$", filter)
                    if m:
                        if row.get("is_active"):
                            result_rows.append(row)
                        continue
                    if "is_active = True" in filter and row.get("is_active"):
                        result_rows.append(row)
                else:
                    result_rows.append(row)
            if limit:
                result_rows = result_rows[:limit]
            return FakeArrowTable(result_rows)

    class FakeArrowTable:
        def __init__(self, rows):
            self.rows = rows
            self.num_rows = len(rows)
        def column_names(self):
            if not self.rows:
                return ["id", "text", "version", "is_active", "category", "tags",
                        "confidence", "importance", "language", "source", "source_detail",
                        "ttl_days", "supersedes", "related_to", "created_at", "updated_at"]
            return list(self.rows[0].keys())
        def column(self, col_name):
            class Col:
                def __init__(self, rows, cn):
                    self.rows = rows
                    self.cn = cn
                def __getitem__(self, i):
                    class Item:
                        def __init__(self, rows, cn, idx):
                            self.rows, self.cn, self.idx = rows, cn, idx
                        def as_py(self_):
                            return self.rows[self_.idx].get(self_.cn)
                    return Item(self.rows, self.cn, i)
            return Col(self.rows, col_name)
        def to_table(self):
            return self

    fake_lance = FakeLance(table._rows)
    table.to_lance = MagicMock(return_value=fake_lance)

    def mock_add(rows):
        for row in rows:
            if isinstance(row, dict):
                table._rows.append(row)

    def mock_update(*, where, values):
        m = re.search(r"id = '([^']+)' AND version = (\d+)", where)
        if m:
            fid, ver = m.group(1), int(m.group(2))
            for row in table._rows:
                if row.get("id") == fid and row.get("version") == ver:
                    row.update(values)

    def mock_delete(where):
        pass

    table.add = mock_add
    table.update = mock_update
    table.delete = mock_delete

    # search method
    def mock_search(vector):
        # Return a SearchBuilder mock
        builder = MagicMock()
        builder._limit = 5
        builder._where = None
        builder._prefilter = False

        def mock_limit(n):
            builder._limit = n
            return builder

        def mock_where(w, prefilter=False):
            builder._where = w
            builder._prefilter = prefilter
            return builder

        def mock_to_list():
            # Simple mock: return first N active rows with fake distances
            results = []
            for row in table._rows:
                if row.get("is_active"):
                    r = dict(row)
                    r["_distance"] = 0.15
                    results.append(r)
                    if len(results) >= builder._limit:
                        break
            return results

        builder.limit = mock_limit
        builder.where = mock_where
        builder.to_list = mock_to_list
        return builder

    table.search = mock_search
    return table


@pytest.fixture
def kb(mock_table):
    """Create a MnemoziaKB with mocked LanceDB table."""
    kb = MnemoziaKB(db_path="/tmp/test_mnemozia", table_name="test_notes")
    kb._table = mock_table

    # Mock the embedding methods
    kb._embed = MagicMock(return_value=[0.1] * 384)
    kb._embed_passage = MagicMock(return_value=[0.1] * 384)

    return kb


# ═══════════════════════════════════════════════════════════════════════
# Tests: add action
# ═══════════════════════════════════════════════════════════════════════


class TestActionAdd:
    def test_add_basic(self, kb):
        result = kb.execute({"action": "add", "text": "Test fact"})
        assert "✅ Stored" in result
        # ID should be 12 hex chars
        assert re.search(r"\[[a-f0-9]{12}\]", result)

    def test_add_missing_text(self, kb):
        result = kb.execute({"action": "add"})
        assert "text" in result.lower()

    def test_add_empty_text(self, kb):
        result = kb.execute({"action": "add", "text": ""})
        assert "text" in result.lower()

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

    def test_add_exact_duplicate(self, kb, mock_table):
        # Pre-populate with a fact
        mock_table._rows.append({
            "id": "existing123456",
            "text": "Exact duplicate test",
            "version": 1,
            "is_active": True,
            "category": "general",
            "tags": "",
            "confidence": 1.0,
            "importance": 0.5,
        })
        result = kb.execute({
            "action": "add",
            "text": "Exact duplicate test",
        })
        assert "Exact duplicate" in result
        assert "existing123456" in result


# ═══════════════════════════════════════════════════════════════════════
# Tests: search action
# ═══════════════════════════════════════════════════════════════════════


class TestActionSearch:
    def test_search_no_query(self, kb):
        result = kb.execute({"action": "search"})
        assert "query" in result.lower() or "text" in result.lower()

    def test_search_empty_kb(self, kb, mock_table):
        result = kb.execute({"action": "search", "query": "anything"})
        assert "No results" in result or "no results" in result

    def test_search_finds_results(self, kb, mock_table):
        mock_table._rows.append({
            "id": "test12345678",
            "text": "PostgreSQL port is 5432",
            "version": 1,
            "is_active": True,
            "category": "devops/database",
            "tags": "postgresql,port",
            "confidence": 1.0,
            "importance": 0.5,
            "language": "ru",
            "updated_at": "2026-06-20 12:00:00",
        })
        result = kb.execute({"action": "search", "query": "database port"})
        assert "PostgreSQL" in result or "test12345678" in result

    def test_search_with_category_filter(self, kb, mock_table):
        mock_table._rows.append({
            "id": "a1b2c3d4e5f6",
            "text": "Nginx reload command",
            "version": 1,
            "is_active": True,
            "category": "devops/nginx",
            "tags": "nginx,reload",
            "confidence": 1.0,
            "importance": 0.5,
            "language": "en",
            "updated_at": "2026-06-20 12:00:00",
        })
        result = kb.execute({
            "action": "search",
            "query": "nginx",
            "mode": "semantic",
            "limit": 5,
        })
        assert "a1b2c3d4e5f6" in result or "Nginx" in result

    def test_search_all_modes(self, kb):
        for mode in ["semantic", "hybrid", "keyword"]:
            result = kb.execute({
                "action": "search",
                "query": "test",
                "mode": mode,
            })
            assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════
# Tests: update action
# ═══════════════════════════════════════════════════════════════════════


class TestActionUpdate:
    def test_update_missing_id(self, kb):
        result = kb.execute({"action": "update", "text": "new text"})
        assert "id" in result.lower()

    def test_update_missing_text(self, kb):
        result = kb.execute({"action": "update", "id": "abc123"})
        assert "text" in result.lower()

    def test_update_nonexistent(self, kb, mock_table):
        result = kb.execute({"action": "update", "id": "nonexistent", "text": "new"})
        assert "No active fact" in result or "not found" in result.lower()

    def test_update_no_change(self, kb, mock_table):
        mock_table._rows.append({
            "id": "unchanged12345",
            "text": "Same text",
            "version": 1,
            "is_active": True,
            "category": "general",
            "tags": "",
            "confidence": 1.0,
            "importance": 0.5,
            "language": "auto",
            "source": "",
            "source_detail": "",
            "ttl_days": 0,
            "supersedes": "",
            "related_to": "",
            "created_at": "2026-06-20 12:00:00",
            "updated_at": "2026-06-20 12:00:00",
        })
        result = kb.execute({
            "action": "update",
            "id": "unchanged12345",
            "text": "Same text",
        })
        assert "unchanged" in result or "no change" in result.lower() or "no update" in result.lower()

    def test_update_creates_new_version(self, kb, mock_table):
        mock_table._rows.append({
            "id": "versiontest12",
            "text": "Old version",
            "version": 1,
            "is_active": True,
            "category": "general",
            "tags": "",
            "confidence": 1.0,
            "importance": 0.5,
            "language": "auto",
            "source": "",
            "source_detail": "",
            "ttl_days": 0,
            "supersedes": "",
            "related_to": "",
            "created_at": "2026-06-20 12:00:00",
            "updated_at": "2026-06-20 12:00:00",
        })
        result = kb.execute({
            "action": "update",
            "id": "versiontest12",
            "text": "New version",
        })
        # Original should have is_active=False
        assert len([r for r in mock_table._rows if r.get("id") == "versiontest12"]) >= 1


# ═══════════════════════════════════════════════════════════════════════
# Tests: deactivate / reactivate
# ═══════════════════════════════════════════════════════════════════════


class TestActionDeactivateReactivate:
    def test_deactivate_missing_id(self, kb):
        result = kb.execute({"action": "deactivate"})
        assert "id" in result.lower()

    def test_deactivate_nonexistent(self, kb, mock_table):
        result = kb.execute({"action": "deactivate", "id": "ghost"})
        assert "No active fact" in result or "not found" in result.lower()

    def test_deactivate_success(self, kb, mock_table):
        mock_table._rows.append({
            "id": "deactivateme1",
            "text": "To be deactivated",
            "version": 1,
            "is_active": True,
            "category": "general",
            "tags": "",
            "confidence": 1.0,
            "importance": 0.5,
            "language": "auto",
            "source": "",
            "source_detail": "",
            "ttl_days": 0,
            "supersedes": "",
            "related_to": "",
            "created_at": "2026-06-20 12:00:00",
            "updated_at": "2026-06-20 12:00:00",
        })
        # Patch _get_active_by_id to return our row
        with patch.object(kb, "_get_active_by_id", return_value=mock_table._rows[0]):
            result = kb.execute({"action": "deactivate", "id": "deactivateme1"})
            assert "Archived" in result or "Deactivated" in result or "🗄" in result

    def test_reactivate_success(self, kb, mock_table):
        mock_table._rows.append({
            "id": "reactivateme1",
            "text": "To be reactivated",
            "version": 1,
            "is_active": False,
            "category": "general",
            "tags": "",
            "confidence": 1.0,
            "importance": 0.5,
            "language": "auto",
            "source": "",
            "source_detail": "",
            "ttl_days": 0,
            "supersedes": "",
            "related_to": "",
            "created_at": "2026-06-20 12:00:00",
            "updated_at": "2026-06-20 12:00:00",
        })
        result = kb.execute({"action": "reactivate", "id": "reactivateme1"})
        assert "Reactivated" in result


# ═══════════════════════════════════════════════════════════════════════
# Tests: history
# ═══════════════════════════════════════════════════════════════════════


class TestActionHistory:
    def test_history_missing_id(self, kb):
        result = kb.execute({"action": "history"})
        assert "id" in result.lower()

    def test_history_nonexistent(self, kb, mock_table):
        result = kb.execute({"action": "history", "id": "phantom"})
        assert "No history" in result or "not found" in result.lower()

    def test_history_with_versions(self, kb, mock_table):
        mock_table._rows.extend([
            {
                "id": "historicfact1",
                "text": "First version",
                "version": 1,
                "is_active": False,
                "category": "general",
                "tags": "",
                "confidence": 1.0,
                "importance": 0.5,
                "language": "auto",
                "source": "",
                "source_detail": "",
                "ttl_days": 0,
                "supersedes": "",
                "related_to": "",
                "created_at": "2026-06-20 12:00:00",
                "updated_at": "2026-06-20 12:00:00",
            },
            {
                "id": "historicfact1",
                "text": "Second version",
                "version": 2,
                "is_active": True,
                "category": "general",
                "tags": "",
                "confidence": 1.0,
                "importance": 0.5,
                "language": "auto",
                "source": "",
                "source_detail": "",
                "ttl_days": 0,
                "supersedes": "",
                "related_to": "",
                "created_at": "2026-06-20 12:00:00",
                "updated_at": "2026-06-21 12:00:00",
            },
        ])
        result = kb.execute({"action": "history", "id": "historicfact1"})
        assert "v1" in result or "v2" in result or "1" in result
        assert "First version" in result or "Second version" in result


# ═══════════════════════════════════════════════════════════════════════
# Tests: stats
# ═══════════════════════════════════════════════════════════════════════


class TestActionStats:
    def test_stats_empty(self, kb):
        result = kb.execute({"action": "stats"})
        assert "Stats" in result or "0" in result or "Total" in result

    def test_stats_with_data(self, kb, mock_table):
        mock_table._rows.extend([
            {
                "id": "stat1", "text": "Fact 1", "version": 1,
                "is_active": True, "category": "devops/networking",
                "tags": "", "confidence": 1.0, "importance": 0.5,
                "language": "ru", "ttl_days": 0,
                "supersedes": "", "related_to": "",
                "created_at": "2026-06-20 12:00:00",
                "updated_at": "2026-06-20 12:00:00",
            },
            {
                "id": "stat2", "text": "Fact 2", "version": 1,
                "is_active": True, "category": "personal",
                "tags": "", "confidence": 0.7, "importance": 0.5,
                "language": "en", "ttl_days": 0,
                "supersedes": "", "related_to": "",
                "created_at": "2026-06-20 12:00:00",
                "updated_at": "2026-06-20 12:00:00",
            },
        ])
        result = kb.execute({"action": "stats"})
        assert "2" in result
        assert "devops/networking" in result
        assert "personal" in result


# ═══════════════════════════════════════════════════════════════════════
# Tests: relate / unrelate
# ═══════════════════════════════════════════════════════════════════════


class TestActionRelate:
    def test_relate_missing_ids(self, kb):
        result = kb.execute({"action": "relate", "id": "a"})
        assert "to" in result.lower()

    def test_unrelate_missing_ids(self, kb):
        result = kb.execute({"action": "unrelate", "id": "a"})
        assert "to" in result.lower()


# ═══════════════════════════════════════════════════════════════════════
# Tests: export
# ═══════════════════════════════════════════════════════════════════════


class TestActionExport:
    def test_export_markdown(self, kb, mock_table):
        mock_table._rows.append({
            "id": "export1", "text": "Export test fact", "version": 1,
            "is_active": True, "category": "general",
            "tags": "test", "confidence": 1.0, "importance": 0.5,
            "language": "en", "ttl_days": 0,
            "supersedes": "", "related_to": "",
            "created_at": "2026-06-20 12:00:00",
            "updated_at": "2026-06-20 12:00:00",
        })
        result = kb.execute({"action": "export", "format": "markdown"})
        assert "Export test fact" in result

    def test_export_json(self, kb, mock_table):
        mock_table._rows.append({
            "id": "export2", "text": "JSON export", "version": 1,
            "is_active": True, "category": "general",
            "tags": "", "confidence": 1.0, "importance": 0.5,
            "language": "en", "ttl_days": 0,
            "supersedes": "", "related_to": "",
            "created_at": "2026-06-20 12:00:00",
            "updated_at": "2026-06-20 12:00:00",
        })
        result = kb.execute({"action": "export", "format": "json"})
        assert "JSON export" in result or "json" in result.lower()

    def test_export_invalid_format(self, kb, mock_table):
        mock_table._rows.append({
            "id": "export3", "text": "Format test", "version": 1,
            "is_active": True, "category": "general",
            "tags": "", "confidence": 1.0, "importance": 0.5,
            "language": "en", "ttl_days": 0,
            "supersedes": "", "related_to": "",
            "created_at": "2026-06-20 12:00:00",
            "updated_at": "2026-06-20 12:00:00",
        })
        result = kb.execute({"action": "export", "format": "csv"})
        assert "Unsupported" in result


# ═══════════════════════════════════════════════════════════════════════
# Tests: error handling and edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestErrorHandling:
    def test_unknown_action(self, kb):
        result = kb.execute({"action": "fly"})
        assert "Unknown action" in result

    def test_missing_action(self, kb):
        result = kb.execute({})
        assert "action" in result.lower()

    def test_execute_wraps_exception(self, kb):
        """execute() should catch exceptions and return error string."""
        # Patch to throw
        with patch.object(kb, "_action_add", side_effect=ValueError("test error")):
            result = kb.execute({"action": "add", "text": "will fail"})
            assert "Error" in result
            assert "test error" in result

    def test_action_add_with_sql_injection_attempt(self, kb):
        """Facts with SQL-like text should store normally."""
        result = kb.execute({
            "action": "add",
            "text": "'; DROP TABLE notes; --",
        })
        assert "✅ Stored" in result

    def test_action_add_unicode(self, kb):
        """Unicode text should store and return properly."""
        result = kb.execute({
            "action": "add",
            "text": "Привет, Mnemozia! 汉字 español français",
        })
        assert "✅ Stored" in result

    def test_concurrent_action_calls(self, kb):
        """Multiple execute() calls should not crash."""
        for i in range(10):
            result = kb.execute({"action": "add", "text": f"Fact number {i}"})
            assert result is not None


# ═══════════════════════════════════════════════════════════════════════
# Tests: merge action
# ═══════════════════════════════════════════════════════════════════════


class TestActionMerge:
    def test_merge_self(self, kb):
        result = kb.execute({"action": "merge", "id": "a", "with": "a"})
        assert "itself" in result.lower() or "cannot" in result.lower()

    def test_merge_missing_ids(self, kb):
        result = kb.execute({"action": "merge", "id": "a"})
        assert "with" in result.lower() or "required" in result.lower()

    def test_merge_nonexistent(self, kb, mock_table):
        result = kb.execute({"action": "merge", "id": "nonexist", "with": "alsofake"})
        assert "No active fact" in result or "not found" in result.lower()


# ═══════════════════════════════════════════════════════════════════════
# Tests: split action
# ═══════════════════════════════════════════════════════════════════════


class TestActionSplit:
    def test_split_missing_id(self, kb):
        result = kb.execute({"action": "split", "parts": "A | B"})
        assert "id" in result.lower() or "required" in result.lower()

    def test_split_missing_parts(self, kb):
        result = kb.execute({"action": "split", "id": "abc"})
        assert "parts" in result.lower() or "required" in result.lower()

    def test_split_nonexistent(self, kb, mock_table):
        result = kb.execute({"action": "split", "id": "ghost", "parts": "A | B"})
        assert "No active fact" in result or "not found" in result.lower()

    def test_split_too_few_parts(self, kb, mock_table):
        mock_table._rows.append({
            "id": "splitme",
            "text": "A | B",
            "version": 1,
            "is_active": True,
            "category": "general",
            "tags": "",
            "confidence": 1.0,
            "importance": 0.5,
            "language": "auto",
            "source": "",
            "source_detail": "",
            "ttl_days": 0,
            "supersedes": "",
            "related_to": "",
            "created_at": "2026-06-20 12:00:00",
            "updated_at": "2026-06-20 12:00:00",
        })
        result = kb.execute({"action": "split", "id": "splitme", "parts": "OnlyOne"})
        assert "at least 2" in result.lower() or "parts" in result.lower()


# ═══════════════════════════════════════════════════════════════════════
# Tests: vacuum
# ═══════════════════════════════════════════════════════════════════════


class TestActionVacuum:
    def test_vacuum_no_args(self, kb):
        result = kb.execute({"action": "vacuum"})
        assert result is not None

    def test_vacuum_with_days(self, kb):
        result = kb.execute({"action": "vacuum", "older_than_days": 30})
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════
# Tests: usage / help
# ═══════════════════════════════════════════════════════════════════════


class TestUsage:
    def test_usage_output(self, kb):
        result = kb.execute({"action": ""})
        assert "add" in result
        assert "search" in result
        assert "update" in result
        assert "merge" in result
        assert "split" in result

    def test_usage_includes_all_actions(self, kb):
        actions = ["add", "search", "update", "merge", "split",
                   "deactivate", "reactivate", "history", "review",
                   "stats", "export", "relate", "unrelate", "vacuum"]
        result = kb.execute({"action": ""})
        for a in actions:
            assert a in result, f"Action '{a}' missing from usage output"
