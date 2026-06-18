"""
Mnemozia tool — registers the `kb` tool with Hermes Agent.

When installed as a Hermes plugin, this file is auto-discovered.
The tool appears as `kb` and provides all 14 knowledge-base operations.
"""

from __future__ import annotations

# ── Mnemozia engine (lazy-loaded — model only loads on first kb() call) ──
_kb_instance = None


def _get_kb():
    """Lazy singleton — creates the engine only when first used."""
    global _kb_instance
    if _kb_instance is None:
        from mnemozia import MnemoziaKB
        _kb_instance = MnemoziaKB()
    return _kb_instance


# ── Handler dispatched by Hermes ──

def kb_handler(args: dict, **_kwargs) -> str:
    """Entry point called by Hermes. Returns JSON string with result."""
    import json
    kb = _get_kb()
    result = kb.execute(args)
    return json.dumps({"success": True, "data": result}, ensure_ascii=False)


# ── Register with Hermes tool registry ──

try:
    from tools.registry import registry

    registry.register(
        name="kb",
        toolset="mnemozia",
        schema={
            "name": "kb",
            "description": (
                "Semantic knowledge base — store, search, update, and manage "
                "facts with versioning, deduplication, and hybrid (semantic + keyword) search. "
                "14 operations: add, search, update, merge, split, deactivate, reactivate, "
                "history, review, stats, export, relate, unrelate, vacuum."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Operation to perform.",
                        "enum": [
                            "add", "search", "update", "merge", "split",
                            "deactivate", "reactivate", "history", "review",
                            "stats", "export", "relate", "unrelate", "vacuum",
                        ],
                    },
                    "text": {
                        "type": "string",
                        "description": "Fact text. Required for add, update, search (as query).",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query (alias for 'text' in search mode).",
                    },
                    "id": {
                        "type": "string",
                        "description": "Fact ID. Required for update, deactivate, reactivate, history, split, merge (primary).",
                    },
                    "with": {
                        "type": "string",
                        "description": "Second fact ID for merge (the one being merged into 'id').",
                    },
                    "to": {
                        "type": "string",
                        "description": "Target fact ID for relate/unrelate.",
                    },
                    "category": {
                        "type": "string",
                        "description": "Category: general, work, personal, finance, credentials, ideas, tech, devops, programming, schedule, contacts, health, travel, learning. Supports hierarchical: 'devops/networking'.",
                    },
                    "tags": {
                        "type": "string",
                        "description": "Comma-separated tags, e.g. 'python,fastapi,orm'.",
                    },
                    "language": {
                        "type": "string",
                        "description": "Language: ru, en, auto (default).",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence 0.0–1.0. 1.0 = verified fact, <0.5 = hypothesis. Default 1.0.",
                    },
                    "importance": {
                        "type": "number",
                        "description": "Importance 0.0–1.0. Higher = boosted in search results. Default 0.5.",
                    },
                    "ttl_days": {
                        "type": "integer",
                        "description": "Auto-archive after N days. 0 = permanent (default).",
                    },
                    "source": {
                        "type": "string",
                        "description": "Origin: 'user', 'session:abc123', 'url:https://...'.",
                    },
                    "source_detail": {
                        "type": "string",
                        "description": "Free-form provenance note.",
                    },
                    "mode": {
                        "type": "string",
                        "description": "Search mode. Default: hybrid. Note: keyword mode falls back to semantic in LanceDB 0.33 (no FTS index support).",
                        "enum": ["hybrid", "semantic", "keyword"],
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results. Default 5 for search, 10 for review.",
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Filter results: minimum confidence threshold.",
                    },
                    "since": {
                        "type": "string",
                        "description": "Filter results: updated after YYYY-MM-DD.",
                    },
                    "format": {
                        "type": "string",
                        "description": "Export format: 'markdown' (default) or 'json'.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File path for export (otherwise returns inline).",
                    },
                    "parts": {
                        "type": "string",
                        "description": "For split: pipe-separated atomic facts, e.g. 'Part A | Part B'.",
                    },
                    "older_than_days": {
                        "type": "integer",
                        "description": "For vacuum: remove archived facts older than N days. Default 365.",
                    },
                },
                "required": ["action"],
            },
        },
        handler=kb_handler,
        check_fn=lambda: True,
    )

except ImportError:
    # Running standalone (not inside Hermes) — tool registration is skipped.
    pass
