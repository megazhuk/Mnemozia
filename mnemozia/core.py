"""
Mnemozia — semantic knowledge base with versioning, dedup, and hybrid search.

Named after Mnemosyne (Μνημοσύνη), the Greek goddess of memory.
Built on PostgreSQL + pgvector + intfloat/multilingual-e5-small (384-dim, ~400 MB RAM).
Designed for VPS with tight memory limits — model is lazy-loaded, searches use
pgvector's HNSW/IVFFlat indexes for fast vector search.

Operations:
    add       search     update     merge      split
    deactivate  reactivate  history    review     stats
    export    relate     unrelate   vacuum
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras

from .schema import (
    compute_embeddings,
    connect,
    ensure_schema,
    _now,
    _uid,
)


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

_DEDUP_THRESHOLD = 0.08   # cosine distance: ≤ this → near-duplicate
_FLAG_THRESHOLD = 0.25    # cosine distance: ≤ this → flag for review

_PRESET_CATEGORIES = {
    "general", "work", "personal", "finance", "credentials",
    "ideas", "tech", "devops", "programming", "schedule",
    "contacts", "health", "travel", "learning",
}


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _distance_to_stars(distance: float) -> str:
    if distance < 0.06:
        return "★★★★★"
    if distance < 0.12:
        return "★★★★☆"
    if distance < 0.20:
        return "★★★☆☆"
    if distance < 0.30:
        return "★★☆☆☆"
    return "★☆☆☆☆"


def _confidence_flag(conf: float) -> str:
    if conf >= 0.9:
        return "✅ verified"
    if conf >= 0.5:
        return "📋 reliable"
    return "⚠️ low confidence"


def _format_result(row: dict, rank: int) -> str:
    stars = _distance_to_stars(row.get("_distance", 0.0))
    dist = row.get("_distance", 0.0)
    conf = row.get("confidence", 1.0)
    flag = _confidence_flag(conf)
    tags = row.get("tags", "") or ""
    tag_str = f" | 🏷 {tags}" if tags else ""

    return (
        f"{rank}. [{row['id']}] {stars} (d={dist:.3f}) | {flag}{tag_str} | "
        f"v{row['version']} | {row['updated_at']}\n"
        f"   📂 {row.get('category', 'general')}\n"
        f"   {row['text']}"
    )


def _parse_float(val, default: float) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _row_to_dict(row) -> dict:
    """Convert a psycopg2 RealDictRow or plain tuple to a plain dict with string timestamps."""
    if isinstance(row, dict):
        d = dict(row)
    else:
        # NamedTuple or tuple — convert
        d = {}
        for desc in row._fields if hasattr(row, '_fields') else []:
            d[desc] = getattr(row, desc)
    # Convert datetime to string
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(v, (list,)) and k == "embedding":
            # Store embedding as list for JSON serialization
            pass
    return d


# ═══════════════════════════════════════════════════════════════════════
# Core engine
# ═══════════════════════════════════════════════════════════════════════


class MnemoziaKB:
    """
    Chronological semantic memory engine (PostgreSQL + pgvector).

    Usage:
        kb = MnemoziaKB("postgresql://postgres:postgres@127.0.0.1:5432/postgres")
        result = kb.execute({"action": "search", "query": "OpenRouter proxy"})
    """

    def __init__(self, db_uri: Optional[str] = None, table_name: str = "notes"):
        self.db_uri = db_uri or os.environ.get(
            "Mnemozia_URI",
            "postgresql://postgres:postgres@127.0.0.1:5432/postgres",
        )
        self.table_name = table_name
        self._conn = None  # lazy-open on first execute()

    # ── lazy accessor ──

    @property
    def conn(self):
        if self._conn is None:
            self._conn = connect(self.db_uri)
            # Use RealDictCursor so rows are returned as dicts
            ensure_schema(self._conn)
        return self._conn

    @property
    def cur(self):
        """Return a cursor with RealDictCursor factory."""
        return self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # ═══════════════════════════════════════════════════════════════════
    # Public entry point
    # ═══════════════════════════════════════════════════════════════════

    def execute(self, arguments: Dict[str, Any]) -> str:
        """Dispatch to the appropriate action handler."""
        action = arguments.get("action", "").strip().lower()
        if not action:
            return self._usage()

        handler = getattr(self, f"_action_{action}", None)
        if handler is None:
            return f"❌ Unknown action '{action}'.\n{self._usage()}"

        try:
            return handler(arguments)
        except Exception as exc:
            return f"❌ Error in action '{action}': {exc}"

    # ═══════════════════════════════════════════════════════════════════
    # Embedding helpers
    # ═══════════════════════════════════════════════════════════════════

    def _embed(self, text: str, is_query: bool = True) -> list[float]:
        """Embed text via llama-server. Qwen3 handles instructions natively."""
        result = compute_embeddings([text])
        return result[0]

    def _embed_passage(self, text: str) -> list[float]:
        return self._embed(text, is_query=False)

    # ═══════════════════════════════════════════════════════════════════
    # ACTION: add  (with dedup)
    # ═══════════════════════════════════════════════════════════════════

    def _action_add(self, args: dict) -> str:
        text = (args.get("text") or "").strip()
        if not text:
            return "❌ Missing required field: 'text'."

        category = (args.get("category") or "general").lower().strip()
        tags = (args.get("tags") or "").strip()
        confidence = _parse_float(args.get("confidence"), 1.0)
        importance = _parse_float(args.get("importance"), 0.5)
        ttl_days = int(args.get("ttl_days", 0))
        source = (args.get("source") or "").strip()
        source_detail = (args.get("source_detail") or "").strip()
        language = (args.get("language") or "auto").strip()

        # ── Step 1: exact text match ──
        cur = self.cur
        cur.execute(
            "SELECT id FROM notes WHERE text = %s AND is_active = TRUE LIMIT 1",
            (text,)
        )
        row = cur.fetchone()
        if row:
            return (
                f"🔁 Exact duplicate found [ID: {row['id']}].\n"
                f"   Use update to change it, or add with different text."
            )

        # ── Step 2: semantic near-duplicate check ──
        similar = self._search_semantic(text, limit=3, threshold=_DEDUP_THRESHOLD)
        if similar:
            s = similar[0]
            return (
                f"🔁 Near-duplicate found [ID: {s['id']}] (distance={s['_distance']:.3f}).\n"
                f"   Existing: \"{s['text'][:120]}{'…' if len(s['text']) > 120 else ''}\"\n"
                f"   Use update to replace, or merge to combine both facts."
            )

        # ── Step 3: potential contradiction flag ──
        flagged = self._search_semantic(text, limit=1, threshold=_FLAG_THRESHOLD)
        warning = ""
        if flagged:
            f = flagged[0]
            warning = (
                f"\n   ⚠️  Related fact exists [ID: {f['id']}] (distance={f['_distance']:.3f}):\n"
                f"   \"{f['text'][:120]}{'…' if len(f['text']) > 120 else ''}\"\n"
                f"   Review for contradiction before trusting this fact."
            )

        # ── Step 4: embed and insert ──
        now = _now()
        fact_id = _uid()
        vector = self._embed_passage(text)

        cur.execute(
            """INSERT INTO notes
               (id, text, embedding, category, tags, language,
                confidence, importance, ttl_days, source, source_detail,
                version, is_active, created_at, updated_at, supersedes, related_to)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                       1, TRUE, %s, %s, '', '')""",
            (fact_id, text, vector, category, tags, language,
             confidence, importance, ttl_days, source, source_detail,
             now, now)
        )
        self.conn.commit()
        return f"✅ Stored [{fact_id}] | {category} | confidence={confidence:.1f}{warning}"

    # ═══════════════════════════════════════════════════════════════════
    # ACTION: search
    # ═══════════════════════════════════════════════════════════════════

    def _action_search(self, args: dict) -> str:
        query = (args.get("query") or args.get("text") or "").strip()
        if not query:
            return "❌ Missing required field: 'query'."

        mode = (args.get("mode") or "hybrid").strip()
        limit = int(args.get("limit", 5))
        category = (args.get("category") or "").strip()
        tags = (args.get("tags") or "").strip()
        language = (args.get("language") or "").strip()
        min_confidence = _parse_float(args.get("min_confidence"), 0.0)
        since = (args.get("since") or "").strip()

        # ── build filter clauses ──
        filters = ["n.is_active = TRUE"]
        params: list = []

        if category:
            filters.append("n.category LIKE %s")
            params.append(f"{category}%")
        if tags:
            for tag in tags.split(","):
                t = tag.strip()
                if t:
                    filters.append("n.tags LIKE %s")
                    params.append(f"%{t}%")
        if language and language != "auto":
            filters.append("n.language = %s")
            params.append(language)
        if min_confidence > 0:
            filters.append("n.confidence >= %s")
            params.append(min_confidence)
        if since:
            filters.append("n.updated_at >= %s")
            params.append(since)

        where = " AND ".join(filters)

        # ── search ──
        if mode == "keyword":
            results = self._search_keyword(query, where, params, limit)
        elif mode == "semantic":
            results = self._search_semantic(query, limit=limit)
        else:  # hybrid
            results = self._search_hybrid(query, where, params, limit)

        if not results:
            return (
                f"🔍 No results for '{query}'"
                + (f" (category: {category})" if category else "")
                + (f" (tags: {tags})" if tags else "")
                + ".\n💡 Try broader terms, different mode, or check category/tags."
            )

        # ── boost by importance then format ──
        results.sort(
            key=lambda r: (r.get("_distance", 1.0) - 0.5 * r.get("importance", 0.5))
        )

        header = (
            f"🔍 *{query}*  ·  {mode}  ·  {len(results)} results"
            + (f"  ·  {category}" if category else "")
            + "\n"
        )
        body = "\n\n".join(_format_result(r, i + 1) for i, r in enumerate(results))

        # ── dedup suggestion ──
        tip = ""
        if len(results) >= 2 and results[0].get("_distance", 1) < 0.06:
            tip = (
                f"\n\n💡 *Tip:* facts [{results[0]['id']}] and [{results[1]['id']}] "
                f"are near-identical. Consider `action=merge id={results[0]['id']} "
                f"with={results[1]['id']}`."
            )

        return header + body + tip

    def _search_semantic(
        self,
        query: str,
        limit: int = 5,
        threshold: Optional[float] = None,
        active_only: bool = True,
    ) -> list:
        """Pure vector search via pgvector cosine distance."""
        vec = self._embed(query, is_query=True)
        vec_str = f"[{','.join(str(v) for v in vec)}]"

        active_clause = "AND n.is_active = TRUE" if active_only else ""
        threshold_clause = f"AND n.embedding <=> '{vec_str}'::vector <= {threshold}" if threshold is not None else ""

        cur = self.cur
        sql = f"""
            SELECT n.*, n.embedding <=> %s::vector AS _distance
            FROM notes n
            WHERE 1=1 {active_clause} {threshold_clause}
            ORDER BY _distance
            LIMIT %s
        """
        cur.execute(sql, (vec_str, limit))
        rows = cur.fetchall()
        return [_row_to_dict(r) for r in rows]

    def _search_keyword(self, query: str, where: str, params: list, limit: int) -> list:
        """Keyword search — full SQL ILIKE + tsvector."""
        cur = self.cur
        # tsvector search for full-text, ILIKE as fallback
        sql = f"""
            SELECT n.*, 0.0 AS _distance
            FROM notes n
            WHERE {where}
              AND (n.text ILIKE %s OR n.text ILIKE %s)
            ORDER BY n.updated_at DESC
            LIMIT %s
        """
        like_param = f"%{query}%"
        words = query.split()
        word_likes = " OR ".join(f"n.text ILIKE %s" for _ in words)
        # Use word-level ILIKE for better keyword search
        sql = f"""
            SELECT n.*, 0.0 AS _distance
            FROM notes n
            WHERE {where}
              AND ({word_likes})
            ORDER BY n.updated_at DESC
            LIMIT %s
        """
        params_keyword = params + [f"%{w}%" for w in words] + [limit]
        try:
            cur.execute(sql, params_keyword)
            rows = cur.fetchall()
            return [_row_to_dict(r) for r in rows]
        except Exception:
            # Fallback: simple ILIKE
            params_fallback = params + [f"%{query}%", limit]
            cur.execute(
                f"""
                SELECT n.*, 0.0 AS _distance
                FROM notes n
                WHERE {where}
                  AND n.text ILIKE %s
                ORDER BY n.updated_at DESC
                LIMIT %s
                """,
                params_fallback
            )
            rows = cur.fetchall()
            return [_row_to_dict(r) for r in rows]

    def _search_hybrid(self, query: str, where: str, params: list, limit: int) -> list:
        """Semantic search with pre-filter."""
        vec = self._embed(query, is_query=True)
        vec_str = f"[{','.join(str(v) for v in vec)}]"

        l = max(limit * 2, 10)  # fetch more for re-ranking
        cur = self.cur
        sql = f"""
            SELECT n.*, n.embedding <=> %s::vector AS _distance
            FROM notes n
            WHERE {where}
            ORDER BY _distance
            LIMIT %s
        """
        try:
            cur.execute(sql, [vec_str] + params + [l])
            rows = cur.fetchall()
            return [_row_to_dict(r) for r in rows][:limit]
        except Exception:
            # Fallback to pure semantic
            return self._search_semantic(query, limit=limit)

    # ═══════════════════════════════════════════════════════════════════
    # ACTION: update
    # ═══════════════════════════════════════════════════════════════════

    def _action_update(self, args: dict) -> str:
        fact_id = (args.get("id") or "").strip()
        new_text = (args.get("text") or "").strip()
        if not fact_id:
            return "❌ Missing required field: 'id'."
        if not new_text:
            return "❌ Missing required field: 'text'."

        # ── find active version ──
        old = self._get_active_by_id(fact_id)
        if not old:
            return f"❌ No active fact with ID '{fact_id}'."

        # ── no-change guard ──
        if old["text"] == new_text:
            return f"ℹ️ Text unchanged for [{fact_id}] — no update needed."

        now = _now()
        new_version = int(old["version"]) + 1

        new_category = (args.get("category") or old.get("category", "general")).lower()
        new_tags = args.get("tags", old.get("tags", ""))
        new_confidence = _parse_float(args.get("confidence"), old.get("confidence", 1.0))
        new_importance = _parse_float(args.get("importance"), old.get("importance", 0.5))
        new_ttl = int(args.get("ttl_days", old.get("ttl_days", 0)))

        # ── soft-delete old version ──
        cur = self.cur
        cur.execute(
            "UPDATE notes SET is_active = FALSE WHERE id = %s AND version = %s",
            (fact_id, old["version"])
        )

        # ── insert new version ──
        vector = self._embed_passage(new_text)
        cur.execute(
            """INSERT INTO notes
               (id, text, embedding, category, tags, language,
                confidence, importance, ttl_days, source, source_detail,
                version, is_active, created_at, updated_at, supersedes, related_to)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                       %s, TRUE, %s, %s, %s, %s)""",
            (fact_id, new_text, vector, new_category, new_tags,
             old.get("language", "auto"), new_confidence, new_importance,
             new_ttl, old.get("source", ""), old.get("source_detail", ""),
             new_version, old.get("created_at", now), now,
             old.get("supersedes", ""), old.get("related_to", ""))
        )
        self.conn.commit()

        return (
            f"✅ Updated [{fact_id}] → v{new_version} at {now}.\n"
            f"   Old (v{old['version']}): \"{old['text'][:100]}"
            f"{'…' if len(old['text']) > 100 else ''}\"\n"
            f"   New (v{new_version}): \"{new_text[:100]}"
            f"{'…' if len(new_text) > 100 else ''}\""
        )

    # ═══════════════════════════════════════════════════════════════════
    # ACTION: merge
    # ═══════════════════════════════════════════════════════════════════

    def _action_merge(self, args: dict) -> str:
        primary_id = (args.get("id") or "").strip()
        with_id = (args.get("with") or "").strip()
        merged_text = (args.get("text") or "").strip()

        if not primary_id or not with_id:
            return "❌ Both 'id' and 'with' fields are required."
        if primary_id == with_id:
            return "❌ Cannot merge a fact with itself."

        p = self._get_active_by_id(primary_id)
        w = self._get_active_by_id(with_id)

        if not p:
            return f"❌ No active fact with ID '{primary_id}'."
        if not w:
            return f"❌ No active fact with ID '{with_id}'."

        now = _now()
        new_text = merged_text or f"{p['text']} | {w['text']}"

        # ── deactivate both originals ──
        cur = self.cur
        cur.execute(
            "UPDATE notes SET is_active = FALSE WHERE id = %s AND version = %s",
            (primary_id, p["version"])
        )
        cur.execute(
            "UPDATE notes SET is_active = FALSE WHERE id = %s AND version = %s",
            (with_id, w["version"])
        )

        # ── create merged fact ──
        new_id = _uid()
        joined_related = _join_ids(
            p.get("related_to", ""), w.get("related_to", ""),
            primary_id, with_id
        )
        vector = self._embed_passage(new_text)

        cur.execute(
            """INSERT INTO notes
               (id, text, embedding, category, tags, language,
                confidence, importance, ttl_days, source, source_detail,
                version, is_active, created_at, updated_at, supersedes, related_to)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                       1, TRUE, %s, %s, %s, %s)""",
            (new_id, new_text, vector, p.get("category", "general"),
             _join_tags(p.get("tags", ""), w.get("tags", "")),
             p.get("language", "auto"),
             max(p.get("confidence", 0), w.get("confidence", 0)),
             max(p.get("importance", 0.5), w.get("importance", 0.5)),
             max(p.get("ttl_days", 0), w.get("ttl_days", 0)),
             "merge",
             f"Merged from {primary_id} + {with_id} on {now}",
             now, now, f"{primary_id},{with_id}", joined_related)
        )
        self.conn.commit()

        return (
            f"✅ Merged [{primary_id}] + [{with_id}] → [{new_id}].\n"
            f"   Originals archived (v{p['version']} and v{w['version']})."
        )

    # ═══════════════════════════════════════════════════════════════════
    # ACTION: split
    # ═══════════════════════════════════════════════════════════════════

    def _action_split(self, args: dict) -> str:
        fact_id = (args.get("id") or "").strip()
        parts_raw = args.get("parts")
        if not fact_id:
            return "❌ Missing required field: 'id'."
        if not parts_raw:
            return "❌ Missing required field: 'parts' (pipe-separated texts or JSON list)."

        if isinstance(parts_raw, list):
            parts = [str(p).strip() for p in parts_raw if str(p).strip()]
        else:
            parts = [p.strip() for p in str(parts_raw).split("|") if p.strip()]
        if len(parts) < 2:
            return "❌ Need at least 2 parts to split. Use '|' as separator."

        orig = self._get_active_by_id(fact_id)
        if not orig:
            return f"❌ No active fact with ID '{fact_id}'."

        now = _now()
        cur = self.cur

        # ── deactivate original ──
        cur.execute(
            "UPDATE notes SET is_active = FALSE WHERE id = %s AND version = %s",
            (fact_id, orig["version"])
        )

        # ── create child facts ──
        new_ids = []
        for i, part in enumerate(parts):
            nid = _uid()
            new_ids.append(nid)
            vector = self._embed_passage(part)

            related = ",".join(n for n in new_ids if n != nid)
            cur.execute(
                """INSERT INTO notes
                   (id, text, embedding, category, tags, language,
                    confidence, importance, source, source_detail,
                    version, is_active, created_at, updated_at, supersedes, related_to)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                           1, TRUE, %s, %s, %s, %s)""",
                (nid, part, vector, orig.get("category", "general"),
                 orig.get("tags", ""), orig.get("language", "auto"),
                 orig.get("confidence", 1.0), orig.get("importance", 0.5),
                 "split",
                 f"Split from {fact_id} (part {i+1}/{len(parts)})",
                 now, now, fact_id, related)
            )
        self.conn.commit()

        return (
            f"✅ Split [{fact_id}] → {len(parts)} atomic facts:\n"
            + "\n".join(
                f"   [{nid}] {part[:100]}{'…' if len(part) > 100 else ''}"
                for nid, part in zip(new_ids, parts)
            )
        )

    # ═══════════════════════════════════════════════════════════════════
    # ACTION: deactivate / reactivate
    # ═══════════════════════════════════════════════════════════════════

    def _action_deactivate(self, args: dict) -> str:
        fact_id = (args.get("id") or "").strip()
        if not fact_id:
            return "❌ Missing required field: 'id'."
        old = self._get_active_by_id(fact_id)
        if not old:
            return f"❌ No active fact with ID '{fact_id}'."
        cur = self.cur
        cur.execute(
            "UPDATE notes SET is_active = FALSE WHERE id = %s AND version = %s",
            (fact_id, old["version"])
        )
        self.conn.commit()
        return f"🗄 Archived [{fact_id}] (v{old['version']}). Use 'reactivate' to restore."

    def _action_reactivate(self, args: dict) -> str:
        fact_id = (args.get("id") or "").strip()
        if not fact_id:
            return "❌ Missing required field: 'id'."
        cur = self.cur
        # Find the latest version
        cur.execute(
            "SELECT * FROM notes WHERE id = %s ORDER BY version DESC LIMIT 1",
            (fact_id,)
        )
        latest = cur.fetchone()
        if not latest:
            return f"❌ No fact with ID '{fact_id}'."
        cur.execute(
            "UPDATE notes SET is_active = TRUE, updated_at = %s "
            "WHERE id = %s AND version = %s",
            (_now(), fact_id, latest["version"])
        )
        self.conn.commit()
        return f"♻️ Reactivated [{fact_id}] (v{latest['version']})."

    # ═══════════════════════════════════════════════════════════════════
    # ACTION: history
    # ═══════════════════════════════════════════════════════════════════

    def _action_history(self, args: dict) -> str:
        fact_id = (args.get("id") or "").strip()
        if not fact_id:
            return "❌ Missing required field: 'id'."
        cur = self.cur
        cur.execute(
            "SELECT * FROM notes WHERE id = %s ORDER BY version",
            (fact_id,)
        )
        rows = cur.fetchall()
        if not rows:
            return f"❌ No history for ID '{fact_id}'."

        created = rows[0].get("created_at", "?")
        if isinstance(created, datetime):
            created = created.strftime("%Y-%m-%d %H:%M:%S")
        out = [f"📜 *Evolution of [{fact_id}]* (created {created})\n"]
        for r in rows:
            r = _row_to_dict(r)
            status = "✅ active" if r.get("is_active") else "🗄 archived"
            upd = r.get("updated_at", "?")
            if isinstance(upd, datetime):
                upd = upd.strftime("%Y-%m-%d %H:%M:%S")
            out.append(
                f"  v{r['version']}  {upd}  [{status}]\n"
                f"    {r['text']}"
            )
        return "\n".join(out)

    # ═══════════════════════════════════════════════════════════════════
    # ACTION: review
    # ═══════════════════════════════════════════════════════════════════

    def _action_review(self, args: dict) -> str:
        limit = int(args.get("limit", 10))
        cur = self.cur
        out = []

        # ── low confidence ──
        cur.execute(
            "SELECT * FROM notes WHERE is_active = TRUE AND confidence < 0.5 LIMIT %s",
            (limit,)
        )
        low = cur.fetchall()
        if low:
            out.append(f"⚠️ *Low confidence (<0.5)* — {len(low)} facts:")
            for r in low:
                r = _row_to_dict(r)
                out.append(f"  [{r['id']}] c={r['confidence']:.1f} | {r['text'][:100]}")

        # ── stale (not updated in 90+ days) ──
        cutoff = datetime.now() - timedelta(days=90)
        cur.execute(
            "SELECT * FROM notes WHERE is_active = TRUE AND updated_at < %s LIMIT %s",
            (cutoff, limit)
        )
        stale = cur.fetchall()
        if stale:
            out.append(f"\n🕰 *Stale (>90 days)* — {len(stale)} facts:")
            for r in stale:
                r = _row_to_dict(r)
                upd = r.get("updated_at", "?")
                if isinstance(upd, datetime):
                    upd = upd.strftime("%Y-%m-%d")
                out.append(f"  [{r['id']}] {upd} | {r['text'][:100]}")

        if not out:
            return "✅ Knowledge base is clean — no facts need review."

        out.append("\n💡 Use `deactivate` for obsolete facts, `update` to refresh confidence.")
        return "\n".join(out)

    # ═══════════════════════════════════════════════════════════════════
    # ACTION: stats
    # ═══════════════════════════════════════════════════════════════════

    def _action_stats(self, args: dict) -> str:  # noqa: ARG002
        cur = self.cur

        cur.execute("SELECT COUNT(*) AS total FROM notes")
        total = cur.fetchone()["total"]

        cur.execute("SELECT COUNT(*) AS active FROM notes WHERE is_active = TRUE")
        active = cur.fetchone()["active"]
        archived = total - active

        # ── by category ──
        cur.execute(
            "SELECT category, COUNT(*) AS cnt FROM notes "
            "WHERE is_active = TRUE GROUP BY category ORDER BY cnt DESC LIMIT 10"
        )
        cat_rows = cur.fetchall()
        cat_lines = "\n".join(
            f"  {r['category']}: {r['cnt']}" for r in cat_rows
        )

        # ── by language ──
        cur.execute(
            "SELECT language, COUNT(*) AS cnt FROM notes "
            "WHERE is_active = TRUE GROUP BY language ORDER BY cnt DESC"
        )
        lang_rows = cur.fetchall()
        lang_lines = "\n".join(
            f"  {r['language']}: {r['cnt']}" for r in lang_rows
        )

        # ── confidence distribution ──
        cur.execute(
            "SELECT COUNT(*) AS cnt FROM notes WHERE is_active = TRUE AND confidence >= 0.9"
        )
        high_conf = cur.fetchone()["cnt"]
        cur.execute(
            "SELECT COUNT(*) AS cnt FROM notes WHERE is_active = TRUE "
            "AND confidence >= 0.5 AND confidence < 0.9"
        )
        med_conf = cur.fetchone()["cnt"]
        cur.execute(
            "SELECT COUNT(*) AS cnt FROM notes WHERE is_active = TRUE AND confidence < 0.5"
        )
        low_conf = cur.fetchone()["cnt"]

        return (
            f"📊 *Knowledge Base Stats*\n\n"
            f"  Total facts: {total}\n"
            f"  Active: {active}  |  Archived: {archived}\n"
            f"  High confidence (≥0.9): {high_conf}\n"
            f"  Medium confidence:     {med_conf}\n"
            f"  Low confidence (<0.5): {low_conf}\n"
            f"\n📂 *By category:*\n{cat_lines or '  (none)'}\n"
            f"\n🌐 *By language:*\n{lang_lines or '  (none)'}\n"
            f"\n💾 DB: pg0 (PostgreSQL + pgvector)"
        )

    # ═══════════════════════════════════════════════════════════════════
    # ACTION: export
    # ═══════════════════════════════════════════════════════════════════

    def _action_export(self, args: dict) -> str:
        fmt = (args.get("format") or "markdown").strip()
        path = (args.get("path") or "").strip()
        category = (args.get("category") or "").strip()

        cur = self.cur
        if category:
            cur.execute(
                "SELECT * FROM notes WHERE is_active = TRUE AND category LIKE %s LIMIT 5000",
                (f"{category}%",)
            )
        else:
            cur.execute(
                "SELECT * FROM notes WHERE is_active = TRUE LIMIT 5000"
            )
        rows = cur.fetchall()
        if not rows:
            return "📭 No facts to export."

        rows_dict = [_row_to_dict(r) for r in rows]

        if fmt == "markdown":
            return self._export_markdown(rows_dict, path)
        elif fmt == "json":
            return self._export_json(rows_dict, path)
        else:
            return f"❌ Unsupported format: '{fmt}'. Use 'markdown' or 'json'."

    def _export_markdown(self, rows: list, path: str) -> str:
        by_cat: Dict[str, list] = {}
        for r in rows:
            cat = r.get("category", "general")
            by_cat.setdefault(cat, []).append(r)

        lines = ["# Hermes Knowledge Base", f"\nExported {_now()}\n"]
        for cat in sorted(by_cat):
            lines.append(f"## {cat}\n")
            for r in by_cat[cat]:
                tags = f" `{r.get('tags', '')}`" if r.get("tags") else ""
                upd = r.get("updated_at", "")
                if isinstance(upd, datetime):
                    upd = upd.strftime("%Y-%m-%d %H:%M:%S")
                lines.append(
                    f"- `[{r['id']}]`{tags} (v{r['version']}, {upd})\n"
                    f"  {r['text']}\n"
                )

        content = "\n".join(lines)
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            with open(os.path.expanduser(path), "w") as f:
                f.write(content)
            return f"✅ Exported {len(rows)} facts → {path} ({len(content)} bytes)"
        return content

    def _export_json(self, rows: list, path: str) -> str:
        cleaned = []
        for r in rows:
            item = {k: v for k, v in r.items() if k != "embedding"}
            # Convert datetime to string
            for k, v in item.items():
                if isinstance(v, datetime):
                    item[k] = v.strftime("%Y-%m-%d %H:%M:%S")
            cleaned.append(item)

        content = json.dumps(cleaned, ensure_ascii=False, indent=2)
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            with open(os.path.expanduser(path), "w") as f:
                f.write(content)
            return f"✅ Exported {len(rows)} facts → {path} ({len(content)} bytes)"
        return content

    # ═══════════════════════════════════════════════════════════════════
    # ACTION: relate / unrelate
    # ═══════════════════════════════════════════════════════════════════

    def _action_relate(self, args: dict) -> str:
        id_a = (args.get("id") or "").strip()
        id_b = (args.get("to") or "").strip()
        if not id_a or not id_b:
            return "❌ Both 'id' and 'to' are required."
        if id_a == id_b:
            return "❌ Cannot relate a fact to itself."
        self._toggle_relation(id_a, id_b, add=True)
        return f"🔗 Linked [{id_a}] ↔ [{id_b}]."

    def _action_unrelate(self, args: dict) -> str:
        id_a = (args.get("id") or "").strip()
        id_b = (args.get("to") or "").strip()
        if not id_a or not id_b:
            return "❌ Both 'id' and 'to' are required."
        self._toggle_relation(id_a, id_b, add=False)
        return f"✂️ Unlinked [{id_a}] ↔ [{id_b}]."

    def _toggle_relation(self, id_a: str, id_b: str, *, add: bool):
        cur = self.cur
        for src, tgt in [(id_a, id_b), (id_b, id_a)]:
            cur.execute(
                "SELECT * FROM notes WHERE id = %s AND is_active = TRUE LIMIT 1",
                (src,)
            )
            r = cur.fetchone()
            if not r:
                continue
            related = set(r.get("related_to", "").split(",")) if r.get("related_to") else set()
            related.discard("")
            if add:
                related.add(tgt)
            else:
                related.discard(tgt)
            cur.execute(
                "UPDATE notes SET related_to = %s WHERE id = %s AND version = %s",
                (",".join(sorted(related)), src, r["version"])
            )
        self.conn.commit()

    # ═══════════════════════════════════════════════════════════════════
    # ACTION: vacuum
    # ═══════════════════════════════════════════════════════════════════

    def _action_vacuum(self, args: dict) -> str:
        days = int(args.get("older_than_days", 365))
        cutoff = datetime.now() - timedelta(days=days)
        cur = self.cur
        cur.execute(
            "DELETE FROM notes WHERE is_active = FALSE AND updated_at < %s",
            (cutoff,)
        )
        deleted = cur.rowcount
        self.conn.commit()
        return f"🧹 Vacuumed: removed {deleted} archived facts older than {days} days."

    # ═══════════════════════════════════════════════════════════════════
    # Internal helpers
    # ═══════════════════════════════════════════════════════════════════

    def _get_active_by_id(self, fact_id: str) -> Optional[dict]:
        cur = self.cur
        cur.execute(
            "SELECT * FROM notes WHERE id = %s AND is_active = TRUE LIMIT 1",
            (fact_id,)
        )
        row = cur.fetchone()
        return _row_to_dict(row) if row else None

    # ═══════════════════════════════════════════════════════════════════
    # Usage
    # ═══════════════════════════════════════════════════════════════════

    def _usage(self) -> str:
        return (
            "📚 *Mnemozia — available actions:*\n\n"
            "  `add`         — store a fact (auto-dedup: exact → near-duplicate → contradiction flag)\n"
            "  `search`      — semantic vector search (`mode: hybrid|semantic|keyword`)\n"
            "  `update`      — new version of an existing fact (preserves history)\n"
            "  `merge`       — combine two facts into one (originals archived)\n"
            "  `split`       — break a fact into atomic parts\n"
            "  `deactivate`  — soft-delete (archived, recoverable)\n"
            "  `reactivate`  — restore an archived fact\n"
            "  `history`     — full version history of a fact\n"
            "  `review`      — show facts needing attention (low confidence, stale)\n"
            "  `stats`       — totals, by category, confidence distribution\n"
            "  `export`      — export as markdown or JSON (`format: markdown|json`, optional `path`)\n"
            "  `relate`      — link two facts (`id` + `to`)\n"
            "  `unrelate`    — remove link between facts\n"
            "  `vacuum`      — hard-delete old archived rows (`older_than_days: 365`)\n"
        )


# ═══════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════


def _join_tags(tags_a: str, tags_b: str) -> str:
    a_set = {t.strip() for t in tags_a.split(",") if t.strip()}
    b_set = {t.strip() for t in tags_b.split(",") if t.strip()}
    return ",".join(sorted(a_set | b_set))


def _join_ids(*id_strings: str) -> str:
    ids: set[str] = set()
    for s in id_strings:
        ids.update(i.strip() for i in s.split(",") if i.strip())
    return ",".join(sorted(ids))
