"""
Mnemozia — semantic knowledge base with versioning, dedup, and hybrid search.

Named after Mnemosyne (Μνημοσύνη), the Greek goddess of memory.
Built on LanceDB + intfloat/multilingual-e5-small (384-dim embeddings, ~400 MB RAM).
Designed for VPS with tight memory limits — model is lazy-loaded, searches use
LanceDB-native filtering (no pandas DataFrames pulled into memory).

Operations:
    add       search     update     merge      split
    deactivate  reactivate  history    review     stats
    export    relate     unrelate   vacuum
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .schema import (
    NoteSchema,
    QUERY_PREFIX,
    PASSAGE_PREFIX,
    get_embedding_model,
    open_or_create_table,
)


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

_DEDUP_THRESHOLD = 0.08   # cosine distance: ≤ this → near-duplicate
_FLAG_THRESHOLD = 0.25    # cosine distance: ≤ this → flag for review

# Valid categories (extensible — new ones are auto-added)
_PRESET_CATEGORIES = {
    "general", "work", "personal", "finance", "credentials",
    "ideas", "tech", "devops", "programming", "schedule",
    "contacts", "health", "travel", "learning",
}


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _uid() -> str:
    return uuid.uuid4().hex[:12]


def _distance_to_stars(distance: float) -> str:
    """Convert cosine distance (0=identical, 1=unrelated) to visual stars."""
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
    """LLM-optimised single-result formatter."""
    stars = _distance_to_stars(row.get("_distance", 0.0))
    dist = row.get("_distance", 0.0)
    conf = row.get("confidence", 1.0)
    flag = _confidence_flag(conf)
    tags = row.get("tags", "")
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


# ═══════════════════════════════════════════════════════════════════════
# Core engine
# ═══════════════════════════════════════════════════════════════════════

class MnemoziaKB:
    """
    Chronological semantic memory engine.

    Usage:
        kb = MnemoziaKB("/home/user/.hermes/knowledge_base")
        result = kb.execute({"action": "search", "query": "OpenRouter proxy"})
    """

    def __init__(self, db_path: str = "~/.hermes/knowledge_base", table_name: str = "notes"):
        import os
        self.db_path = os.path.expanduser(db_path)
        self.table_name = table_name
        self._table = None           # lazy-open on first execute()

    # ── lazy accessor (model + table only load when actually used) ──

    @property
    def table(self):
        if self._table is None:
            self._table = open_or_create_table(self.db_path, self.table_name)
        return self._table

    # ═══════════════════════════════════════════════════════════════════
    # Public entry point (called by Hermes tool / CLI)
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
        try:
            # Use table query (not FTS search) to find exact text duplicates
            exact = self.table.to_lance().scanner(
                filter=f"text = '{text}' AND is_active = True"
            ).to_table()
            if exact.num_rows > 0:
                eid = exact.column("id")[0].as_py()
                return (
                    f"🔁 Exact duplicate found [ID: {eid}].\n"
                    f"   Use update to change it, or add with different text."
                )
        except Exception:
            pass  # scanner fallback; proceed

        # ── Step 2: semantic near-duplicate check ──
        similar = self._search_semantic(
            text, limit=3, threshold=_DEDUP_THRESHOLD, active_only=True
        )
        if similar:
            s = similar[0]
            sid = s["id"]
            return (
                f"🔁 Near-duplicate found [ID: {sid}] (distance={s['_distance']:.3f}).\n"
                f"   Existing: \"{s['text'][:120]}{'…' if len(s['text'])>120 else ''}\"\n"
                f"   Use update to replace, or merge to combine both facts."
            )

        # ── Step 3: potential contradiction flag ──
        flagged = self._search_semantic(
            text, limit=1, threshold=_FLAG_THRESHOLD, active_only=True
        )
        warning = ""
        if flagged:
            f = flagged[0]
            warning = (
                f"\n   ⚠️  Related fact exists [ID: {f['id']}] (distance={f['_distance']:.3f}):\n"
                f"   \"{f['text'][:120]}{'…' if len(f['text'])>120 else ''}\"\n"
                f"   Review for contradiction before trusting this fact."
            )

        # ── Step 4: embed and insert ──
        now = _now()
        fact_id = _uid()
        vector = self._embed_passage(text)
        self.table.add([{
            "id": fact_id,
            "text": text,
            "vector": vector,
            "category": category,
            "tags": tags,
            "language": language,
            "confidence": confidence,
            "importance": importance,
            "ttl_days": ttl_days,
            "source": source,
            "source_detail": source_detail,
            "version": 1,
            "is_active": True,
            "created_at": now,
            "updated_at": now,
            "supersedes": "",
            "related_to": "",
        }])
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
        since = (args.get("since") or "").strip()  # YYYY-MM-DD

        # ── build where clause ──
        where_parts = ["is_active = True"]
        if category:
            # hierarchical: "devops/networking" matches "devops/networking/proxy"
            where_parts.append(f"category LIKE '{category}%'")
        if tags:
            for tag in tags.split(","):
                t = tag.strip()
                if t:
                    where_parts.append(f"tags LIKE '%{t}%'")
        if language and language != "auto":
            where_parts.append(f"language = '{language}'")
        if min_confidence > 0:
            where_parts.append(f"confidence >= {min_confidence}")
        if since:
            where_parts.append(f"updated_at >= '{since}'")
        where = " AND ".join(where_parts)

        # ── search ──
        if mode == "keyword":
            results = self._search_keyword(query, where, limit)
        elif mode == "semantic":
            results = self._search_semantic(query, limit=limit)
        else:  # hybrid
            results = self._search_hybrid(query, where, limit)

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
                f"are near-identical. Consider `action=merge id={results[0]['id']} with={results[1]['id']}`."
            )

        return header + body + tip

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

        # ── update category / tags if provided ──
        new_category = (args.get("category") or old.get("category", "general")).lower()
        new_tags = args.get("tags", old.get("tags", ""))
        new_confidence = _parse_float(args.get("confidence"), old.get("confidence", 1.0))
        new_importance = _parse_float(args.get("importance"), old.get("importance", 0.5))
        new_ttl = int(args.get("ttl_days", old.get("ttl_days", 0)))

        # ── soft-delete old version ──
        self.table.update(
            where=f"id = '{fact_id}' AND version = {old['version']}",
            values={"is_active": False}
        )

        # ── insert new version ──
        vector = self._embed_passage(new_text)
        self.table.add([{
            "id": fact_id,
            "text": new_text,
            "vector": vector,
            "category": new_category,
            "tags": new_tags,
            "language": old.get("language", "auto"),
            "confidence": new_confidence,
            "importance": new_importance,
            "ttl_days": new_ttl,
            "source": old.get("source", ""),
            "source_detail": old.get("source_detail", ""),
            "version": new_version,
            "is_active": True,
            "created_at": old["created_at"],
            "updated_at": now,
            "supersedes": old.get("supersedes", ""),
            "related_to": old.get("related_to", ""),
        }])
        return (
            f"✅ Updated [{fact_id}] → v{new_version} at {now}.\n"
            f"   Old (v{old['version']}): \"{old['text'][:100]}"
            f"{'…' if len(old['text'])>100 else ''}\"\n"
            f"   New (v{new_version}): \"{new_text[:100]}"
            f"{'…' if len(new_text)>100 else ''}\""
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

        # ── fetch both facts ──
        p = self._get_active_by_id(primary_id)
        w = self._get_active_by_id(with_id)

        if not p:
            return f"❌ No active fact with ID '{primary_id}'."
        if not w:
            return f"❌ No active fact with ID '{with_id}'."
        now = _now()
        new_text = merged_text or f"{p['text']} | {w['text']}"

        # ── deactivate both originals ──
        self.table.update(
            where=f"id = '{primary_id}' AND version = {p['version']}",
            values={"is_active": False}
        )
        self.table.update(
            where=f"id = '{with_id}' AND version = {w['version']}",
            values={"is_active": False}
        )

        # ── create merged fact ──
        new_id = _uid()
        joined_related = _join_ids(p.get("related_to", ""), w.get("related_to", ""),
                                  primary_id, with_id)
        vector = self._embed_passage(new_text)
        self.table.add([{
            "id": new_id,
            "text": new_text,
            "vector": vector,
            "category": p.get("category", "general"),
            "tags": _join_tags(p.get("tags", ""), w.get("tags", "")),
            "language": p.get("language", "auto"),
            "confidence": max(p.get("confidence", 0), w.get("confidence", 0)),
            "importance": max(p.get("importance", 0.5), w.get("importance", 0.5)),
            "ttl_days": max(p.get("ttl_days", 0), w.get("ttl_days", 0)),
            "source": "merge",
            "source_detail": f"Merged from {primary_id} + {with_id} on {now}",
            "version": 1,
            "is_active": True,
            "created_at": now,
            "updated_at": now,
            "supersedes": f"{primary_id},{with_id}",
            "related_to": joined_related,
        }])
        return (
            f"✅ Merged [{primary_id}] + [{with_id}] → [{new_id}].\n"
            f"   Originals archived (v{p['version']} and v{w['version']})."
        )

    # ═══════════════════════════════════════════════════════════════════
    # ACTION: split
    # ═══════════════════════════════════════════════════════════════════

    def _action_split(self, args: dict) -> str:
        fact_id = (args.get("id") or "").strip()
        parts_raw = args.get("parts")  # string or list
        if not fact_id:
            return "❌ Missing required field: 'id'."
        if not parts_raw:
            return "❌ Missing required field: 'parts' (comma-separated texts or JSON list)."

        # ── parse parts ──
        if isinstance(parts_raw, list):
            parts = [str(p).strip() for p in parts_raw if str(p).strip()]
        else:
            parts = [p.strip() for p in str(parts_raw).split("|") if p.strip()]
        if len(parts) < 2:
            return "❌ Need at least 2 parts to split. Use '|' as separator."

        # ── fetch original ──
        orig = self._get_active_by_id(fact_id)
        if not orig:
            return f"❌ No active fact with ID '{fact_id}'."
        now = _now()

        # ── deactivate original ──
        self.table.update(
            where=f"id = '{fact_id}' AND version = {orig['version']}",
            values={"is_active": False}
        )

        # ── create child facts ──
        new_ids = []
        for i, part in enumerate(parts):
            nid = _uid()
            new_ids.append(nid)
            vector = self._embed_passage(part)
            self.table.add([{
                "id": nid,
                "text": part,
                "vector": vector,
                "category": orig.get("category", "general"),
                "tags": orig.get("tags", ""),
                "language": orig.get("language", "auto"),
                "confidence": orig.get("confidence", 1.0),
                "importance": orig.get("importance", 0.5),
                "source": "split",
                "source_detail": f"Split from {fact_id} (part {i+1}/{len(parts)})",
                "version": 1,
                "is_active": True,
                "created_at": now,
                "updated_at": now,
                "supersedes": fact_id,
                "related_to": ",".join(n for n in new_ids if n != nid),
            }])
        return (
            f"✅ Split [{fact_id}] → {len(parts)} atomic facts:\n"
            + "\n".join(f"   [{nid}] {part[:100]}{'…' if len(part)>100 else ''}"
                        for nid, part in zip(new_ids, parts))
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
        self.table.update(
            where=f"id = '{fact_id}' AND version = {old['version']}",
            values={"is_active": False}
        )
        return f"🗄 Archived [{fact_id}] (v{old['version']}). Use 'reactivate' to restore."

    def _action_reactivate(self, args: dict) -> str:
        fact_id = (args.get("id") or "").strip()
        if not fact_id:
            return "❌ Missing required field: 'id'."
        all_versions = self._get_all_by_id(fact_id)
        if not all_versions:
            return f"❌ No fact with ID '{fact_id}'."
        latest = max(all_versions, key=lambda r: r.get("version", 0))
        self.table.update(
            where=f"id = '{fact_id}' AND version = {latest['version']}",
            values={"is_active": True, "updated_at": _now()}
        )
        return f"♻️ Reactivated [{fact_id}] (v{latest['version']})."

    # ═══════════════════════════════════════════════════════════════════
    # ACTION: history
    # ═══════════════════════════════════════════════════════════════════

    def _action_history(self, args: dict) -> str:
        fact_id = (args.get("id") or "").strip()
        if not fact_id:
            return "❌ Missing required field: 'id'."
        rows = self._get_all_by_id(fact_id)
        if not rows:
            return f"❌ No history for ID '{fact_id}'."
        rows.sort(key=lambda r: r.get("version", 0))
        created = rows[0].get("created_at", "?")
        out = [f"📜 *Evolution of [{fact_id}]* (created {created})\n"]
        for r in rows:
            status = "✅ active" if r.get("is_active") else "🗄 archived"
            out.append(
                f"  v{r['version']}  {r['updated_at']}  [{status}]\n"
                f"    {r['text']}"
            )
        return "\n".join(out)

    # ═══════════════════════════════════════════════════════════════════
    # ACTION: review
    # ═══════════════════════════════════════════════════════════════════

    def _action_review(self, args: dict) -> str:
        """Show facts that need human attention: low confidence, stale, or TTL-expired."""
        limit = int(args.get("limit", 10))
        now = datetime.now()
        out = []

        # ── low confidence ──
        try:
            low_tbl = self.table.to_lance().scanner(
                filter="is_active = True AND confidence < 0.5",
                limit=limit,
            ).to_table()
            low = []
            col_names = low_tbl.column_names()
            for i in range(low_tbl.num_rows):
                low.append({col: low_tbl.column(col)[i].as_py() for col in col_names})
        except Exception:
            low = self.table.search().where("is_active = True AND confidence < 0.5").limit(limit).to_list()
        if low:
            out.append(f"⚠️ *Low confidence (<0.5)* — {len(low)} facts:")
            for r in low:
                out.append(f"  [{r['id']}] c={r['confidence']:.1f} | {r['text'][:100]}")

        # ── stale (not updated in 90+ days) ──
        try:
            all_tbl = self.table.to_lance().scanner(
                filter="is_active = True"
            ).to_table()
            col_names = all_tbl.column_names()
            all_active = []
            for i in range(all_tbl.num_rows):
                all_active.append({col: all_tbl.column(col)[i].as_py() for col in col_names})
        except Exception:
            all_active = []
        try:
            from datetime import timedelta
            stale = [
                r for r in all_active
                if (now - datetime.strptime(r["updated_at"], "%Y-%m-%d %H:%M:%S")).days > 90
            ][:limit]
            if stale:
                out.append(f"\n🕰 *Stale (>90 days)* — {len(stale)} facts:")
                for r in stale:
                    out.append(f"  [{r['id']}] {r['updated_at']} | {r['text'][:100]}")
        except Exception:
            pass

        if not out:
            return "✅ Knowledge base is clean — no facts need review."

        out.append("\n💡 Use `deactivate` for obsolete facts, `update` to refresh confidence.")
        return "\n".join(out)

    # ═══════════════════════════════════════════════════════════════════
    # ACTION: stats
    # ═══════════════════════════════════════════════════════════════════

    def _action_stats(self, args: dict) -> str:  # noqa: ARG002
        # Use scanner for full table scan (no FTS needed)
        try:
            tbl = self.table.to_lance().scanner().to_table()
            col_names = tbl.column_names()
            all_rows = []
            for i in range(tbl.num_rows):
                all_rows.append({col: tbl.column(col)[i].as_py() for col in col_names})
        except Exception:
            all_rows = self.table.search().limit(10000).to_list()
        total = len(all_rows)
        active = sum(1 for r in all_rows if r.get("is_active"))
        archived = total - active

        # ── by category ──
        cat_counts: Dict[str, int] = {}
        for r in all_rows:
            if r.get("is_active"):
                cat_counts[r.get("category", "general")] = cat_counts.get(r.get("category", "general"), 0) + 1
        cat_lines = "\n".join(f"  {c}: {n}" for c, n in sorted(cat_counts.items(), key=lambda x: -x[1])[:10])

        # ── by language ──
        lang_counts: Dict[str, int] = {}
        for r in all_rows:
            if r.get("is_active"):
                lang_counts[r.get("language", "auto")] = lang_counts.get(r.get("language", "auto"), 0) + 1
        lang_lines = "\n".join(f"  {l}: {n}" for l, n in sorted(lang_counts.items(), key=lambda x: -x[1]))

        # ── confidence distribution ──
        high_conf = sum(1 for r in all_rows if r.get("is_active") and r.get("confidence", 0) >= 0.9)
        med_conf = sum(1 for r in all_rows if r.get("is_active") and 0.5 <= r.get("confidence", 0) < 0.9)
        low_conf = sum(1 for r in all_rows if r.get("is_active") and r.get("confidence", 0) < 0.5)

        return (
            f"📊 *Knowledge Base Stats*\n\n"
            f"  Total facts: {total}\n"
            f"  Active: {active}  |  Archived: {archived}\n"
            f"  High confidence (≥0.9): {high_conf}\n"
            f"  Medium confidence:     {med_conf}\n"
            f"  Low confidence (<0.5): {low_conf}\n"
            f"\n📂 *By category:*\n{cat_lines or '  (none)'}\n"
            f"\n🌐 *By language:*\n{lang_lines or '  (none)'}\n"
            f"\n💾 DB: {self.db_path}/{self.table_name}"
        )

    # ═══════════════════════════════════════════════════════════════════
    # ACTION: export
    # ═══════════════════════════════════════════════════════════════════

    def _action_export(self, args: dict) -> str:
        fmt = (args.get("format") or "markdown").strip()
        path = (args.get("path") or "").strip()
        category = (args.get("category") or "").strip()

        where = "is_active = True"
        if category:
            where += f" AND category LIKE '{category}%'"

        try:
            tbl = self.table.to_lance().scanner(filter=where, limit=5000).to_table()
            col_names = tbl.column_names()
            rows = []
            for i in range(tbl.num_rows):
                rows.append({col: tbl.column(col)[i].as_py() for col in col_names})
        except Exception:
            rows = self.table.search().where(where).limit(5000).to_list()
        if not rows:
            return "📭 No facts to export."

        if fmt == "markdown":
            return self._export_markdown(rows, path)
        elif fmt == "json":
            return self._export_json(rows, path)
        else:
            return f"❌ Unsupported format: '{fmt}'. Use 'markdown' or 'json'."

    def _export_markdown(self, rows: list, path: str) -> str:
        import os
        # ── group by category ──
        by_cat: Dict[str, list] = {}
        for r in rows:
            cat = r.get("category", "general")
            by_cat.setdefault(cat, []).append(r)

        lines = ["# Hermes Knowledge Base", f"\nExported {_now()}\n"]
        for cat in sorted(by_cat):
            lines.append(f"## {cat}\n")
            for r in by_cat[cat]:
                tags = f" `{r.get('tags', '')}`" if r.get("tags") else ""
                lines.append(
                    f"- `[{r['id']}]`{tags} (v{r['version']}, {r['updated_at']})\n"
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
        import json, os
        # ── strip vectors, keep everything else ──
        cleaned = []
        for r in rows:
            item = {k: v for k, v in r.items() if k != "vector"}
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
        for (src, tgt) in [(id_a, id_b), (id_b, id_a)]:
            r = self._get_active_by_id(src)
            if not r:
                continue
            related = set(r.get("related_to", "").split(",")) if r.get("related_to") else set()
            related.discard("")  # clean empty strings from split
            if add:
                related.add(tgt)
            else:
                related.discard(tgt)
            self.table.update(
                where=f"id = '{src}' AND version = {r['version']}",
                values={"related_to": ",".join(sorted(related))}
            )

    # ═══════════════════════════════════════════════════════════════════
    # ACTION: vacuum  (hard-delete archived rows older than N days)
    # ═══════════════════════════════════════════════════════════════════

    def _action_vacuum(self, args: dict) -> str:
        days = int(args.get("older_than_days", 365))
        cutoff = (datetime.now() - __import__("datetime").timedelta(days=days)).strftime("%Y-%m-%d")
        try:
            self.table.delete(
                f"is_active = False AND updated_at < '{cutoff}'"
            )
        except Exception as e:
            # LanceDB may not support DELETE with complex WHERE
            return f"⚠️ Vacuum skipped: {e}\n   LanceDB doesn't support DELETE with date filters in all versions."
        return f"🧹 Vacuumed: removed archived facts older than {cutoff}."

    # ═══════════════════════════════════════════════════════════════════
    # Internal helpers
    # ═══════════════════════════════════════════════════════════════════

    def _embed(self, text: str, is_query: bool = True) -> list:
        """Embed text using the lazy-loaded model. Adds query/passage prefix.

        The e5 family requires asymmetric prefixes:
        - queries:  "query: <text>"
        - passages: "passage: <text>"
        """
        model = get_embedding_model()
        prefix = QUERY_PREFIX if is_query else PASSAGE_PREFIX
        result = model.compute_query_embeddings([f"{prefix}{text}"])
        return result[0]  # returns list of vectors, take the first

    def _embed_passage(self, text: str) -> list:
        """Shortcut for embedding a document/passage."""
        return self._embed(text, is_query=False)

    def _get_active_by_id(self, fact_id: str) -> Optional[dict]:
        """Fetch the active version of a fact by ID (scalar filter, no FTS)."""
        try:
            tbl = self.table.to_lance().scanner(
                filter=f"id = '{fact_id}' AND is_active = True",
                limit=1,
            ).to_table()
            if tbl.num_rows == 0:
                return None
            return {col: tbl.column(col)[0].as_py() for col in tbl.column_names()}
        except Exception:
            return None

    def _get_all_by_id(self, fact_id: str) -> list:
        """Fetch all versions of a fact (active + archived)."""
        try:
            tbl = self.table.to_lance().scanner(
                filter=f"id = '{fact_id}'",
            ).to_table()
            rows = []
            col_names = tbl.column_names()
            for i in range(tbl.num_rows):
                rows.append({col: tbl.column(col)[i].as_py() for col in col_names})
            return rows
        except Exception:
            return []

    def _search_semantic(
        self,
        query: str,
        limit: int = 5,
        threshold: Optional[float] = None,
        active_only: bool = True,
    ) -> list:
        """Pure vector search — embeds query manually, passes vector to LanceDB."""
        vec = self._embed(query, is_query=True)
        results = self.table.search(vec).limit(limit).to_list()
        if active_only:
            results = [r for r in results if r.get("is_active")]
        if threshold is not None:
            results = [r for r in results if r.get("_distance", 1.0) <= threshold]
        return results

    def _search_keyword(self, query: str, where: str, limit: int) -> list:
        """FTS/text search — falls back to semantic because LanceDB lacks INVERTED index."""
        # LanceDB 0.33 does not support INVERTED index; keyword search is unavailable.
        # Fall back to semantic with the query text.
        return self._search_semantic(query, limit=limit)

    def _search_hybrid(self, query: str, where: str, limit: int) -> list:
        """Semantic search with pre-filter (category, tags, etc.)."""
        vec = self._embed(query, is_query=True)
        try:
            return (
                self.table.search(vec)
                .where(where, prefilter=True)
                .limit(limit)
                .to_list()
            )
        except Exception:
            # Pre-filter failed; fall back to semantic
            results = self._search_semantic(query, limit=limit * 2)
            return results[:limit]

    # ═══════════════════════════════════════════════════════════════════
    # Usage
    # ═══════════════════════════════════════════════════════════════════

    def _usage(self) -> str:
        return (
            "📚 *Mnemozia — available actions:*\n\n"
            "  `add`         — store a fact (auto-dedup: exact → near-duplicate → contradiction flag)\n"
            "  `search`      — semantic vector search (`mode: hybrid|semantic`; keyword falls back to semantic in LanceDB 0.33)\n"
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
# Helper functions used by merge/relate
# ═══════════════════════════════════════════════════════════════════════

def _join_tags(tags_a: str, tags_b: str) -> str:
    """Merge two comma-separated tag strings, deduplicating."""
    a_set = {t.strip() for t in tags_a.split(",") if t.strip()}
    b_set = {t.strip() for t in tags_b.split(",") if t.strip()}
    return ",".join(sorted(a_set | b_set))


def _join_ids(*id_strings: str) -> str:
    """Merge comma-separated ID strings, deduplicating."""
    ids: set[str] = set()
    for s in id_strings:
        ids.update(i.strip() for i in s.split(",") if i.strip())
    return ",".join(sorted(ids))
