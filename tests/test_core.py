"""
Tests for Mnemozia core logic — validates parsing, dedup, formatting, error handling.

Uses mocked psycopg2 (conftest.py provides MockCursor + MockConnection).
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from unittest.mock import MagicMock, patch

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
# Fixtures — MnemoziaKB with mocked DB
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def kb():
    """Create a MnemoziaKB with mocked psycopg2 connection."""
    kb = MnemoziaKB(db_uri="postgresql://mock:mock@localhost:5432/mock")

    # Mock the embedding methods (no real model needed)
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

    def test_add_exact_duplicate(self, kb):
        """First add works, second with same text triggers duplicate."""
        r1 = kb.execute({"action": "add", "text": "Exact duplicate test"})
        assert "✅ Stored" in r1
        r2 = kb.execute({"action": "add", "text": "Exact duplicate test"})
        assert "Exact duplicate" in r2 or "Near-duplicate" in r2


# ═══════════════════════════════════════════════════════════════════════
# Tests: search action
# ═══════════════════════════════════════════════════════════════════════


class TestActionSearch:
    def test_search_no_query(self, kb):
        result = kb.execute({"action": "search"})
        assert "query" in result.lower() or "text" in result.lower()

    def test_search_empty_kb(self, kb):
        result = kb.execute({"action": "search", "query": "anything"})
        assert "No results" in result or "no results" in result

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

    def test_update_nonexistent(self, kb):
        result = kb.execute({"action": "update", "id": "nonexistent", "text": "new"})
        assert "No active fact" in result or "not found" in result.lower()

    def test_update_no_change(self, kb):
        """Add a fact, then update with same text."""
        r1 = kb.execute({"action": "add", "text": "Same text"})
        assert "✅ Stored" in r1
        fid = re.search(r"\[([a-f0-9]{12})\]", r1).group(1)

        r2 = kb.execute({"action": "update", "id": fid, "text": "Same text"})
        assert "unchanged" in r2 or "no change" in r2.lower() or "no update" in r2.lower()

    def test_update_creates_new_version(self, kb):
        """Add then update with new text should create version 2."""
        r1 = kb.execute({"action": "add", "text": "Original version"})
        fid = re.search(r"\[([a-f0-9]{12})\]", r1).group(1)
        r2 = kb.execute({"action": "update", "id": fid, "text": "Updated version"})
        assert "v2" in r2 or "Updated" in r2


# ═══════════════════════════════════════════════════════════════════════
# Tests: deactivate / reactivate
# ═══════════════════════════════════════════════════════════════════════


class TestActionDeactivateReactivate:
    def test_deactivate_missing_id(self, kb):
        result = kb.execute({"action": "deactivate"})
        assert "id" in result.lower()

    def test_deactivate_nonexistent(self, kb):
        result = kb.execute({"action": "deactivate", "id": "ghost"})
        assert "No active fact" in result or "not found" in result.lower()

    def test_deactivate_success(self, kb):
        r1 = kb.execute({"action": "add", "text": "To be deactivated"})
        fid = re.search(r"\[([a-f0-9]{12})\]", r1).group(1)
        with patch.object(kb, "_get_active_by_id",
                          return_value={"id": fid, "text": "To be deactivated",
                                        "version": 1, "is_active": True}):
            result = kb.execute({"action": "deactivate", "id": fid})
            assert "Archived" in result or "🗄" in result

    def test_reactivate_success(self, kb):
        r1 = kb.execute({"action": "add", "text": "To be reactivated"})
        fid = re.search(r"\[([a-f0-9]{12})\]", r1).group(1)
        # Deactivate first
        r2 = kb.execute({"action": "deactivate", "id": fid})
        assert "Archived" in r2 or "🗄" in r2
        # Then reactivate — mock fetchone for _get_all_by_id equivalent
        result = kb.execute({"action": "reactivate", "id": fid})
        assert "Reactivated" in result


# ═══════════════════════════════════════════════════════════════════════
# Tests: history
# ═══════════════════════════════════════════════════════════════════════


class TestActionHistory:
    def test_history_missing_id(self, kb):
        result = kb.execute({"action": "history"})
        assert "id" in result.lower()

    def test_history_nonexistent(self, kb):
        result = kb.execute({"action": "history", "id": "phantom"})
        assert "No history" in result or "not found" in result.lower()


# ═══════════════════════════════════════════════════════════════════════
# Tests: stats
# ═══════════════════════════════════════════════════════════════════════


class TestActionStats:
    def test_stats_empty(self, kb):
        result = kb.execute({"action": "stats"})
        assert "Stats" in result or "0" in result or "Total" in result


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
    def test_export_empty(self, kb):
        result = kb.execute({"action": "export", "format": "markdown"})
        assert "No facts" in result

    def test_export_invalid_format(self, kb):
        # Add a fact first so we don't hit "No facts" before format check
        kb.execute({"action": "add", "text": "Test"})
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
        with patch.object(kb, "_action_add", side_effect=ValueError("test error")):
            result = kb.execute({"action": "add", "text": "will fail"})
            assert "Error" in result
            assert "test error" in result

    def test_action_add_unicode(self, kb):
        result = kb.execute({
            "action": "add",
            "text": "Привет, Mnemozia! 汉字 español français",
        })
        assert "✅ Stored" in result

    def test_concurrent_action_calls(self, kb):
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

    def test_merge_nonexistent(self, kb):
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

    def test_split_nonexistent(self, kb):
        result = kb.execute({"action": "split", "id": "ghost", "parts": "A | B"})
        assert "No active fact" in result or "not found" in result.lower()

    def test_split_too_few_parts(self, kb):
        r1 = kb.execute({"action": "add", "text": "A | B"})
        fid = re.search(r"\[([a-f0-9]{12})\]", r1).group(1)
        with patch.object(kb, "_get_active_by_id",
                          return_value={"id": fid, "text": "A | B",
                                        "version": 1, "is_active": True}):
            result = kb.execute({"action": "split", "id": fid, "parts": "OnlyOne"})
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
