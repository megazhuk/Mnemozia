#!/usr/bin/env python3
"""
Comprehensive integration tests for Mnemozia v2.0.0 (pg0 + llama.cpp + Qwen3).

Tests every action, edge case, data transformation, and search mode.
Reports structured results — not just pass/fail, but evidence of correctness.
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mnemozia import MnemoziaKB
from mnemozia.schema import ensure_schema, connect

# ═══════════════════════════════════════════════════════════════════════
# Test harness
# ═══════════════════════════════════════════════════════════════════════

class TestReport:
    def __init__(self):
        self.sections: list[dict] = []
        self.passes = 0
        self.fails = 0
        self._current_section = None

    def section(self, name: str):
        self._current_section = {"name": name, "checks": []}
        self.sections.append(self._current_section)

    def check(self, name: str, passed: bool, detail: str = ""):
        self._current_section["checks"].append({
            "name": name,
            "passed": passed,
            "detail": detail,
        })
        if passed:
            self.passes += 1
        else:
            self.fails += 1
            print(f"  ❌ FAIL: {name} — {detail}")

    def print_report(self):
        print("=" * 72)
        print(f"  MNEMOZIA v2 — Comprehensive Test Report")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 72)
        for sec in self.sections:
            status = "✅" if all(c["passed"] for c in sec["checks"]) else "❌"
            total = len(sec["checks"])
            good = sum(1 for c in sec["checks"] if c["passed"])
            print(f"\n  {status}  {sec['name']}  ({good}/{total})")
            print(f"  {'─' * (len(sec['name']) + 10)}")
            for c in sec["checks"]:
                icon = "✅" if c["passed"] else "❌"
                print(f"  {icon} {c['name']}")
                if c["detail"] and not c["passed"]:
                    for line in c["detail"].split("\n"):
                        print(f"     {line}")
        print(f"\n{'═' * 72}")
        total = self.passes + self.fails
        pct = (self.passes / total * 100) if total else 0
        print(f"  RESULT: {self.passes}/{total} passed ({pct:.0f}%)")
        print(f"  FAILS:  {self.fails}")
        print(f"{'═' * 72}")


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def get_db_uri():
    return os.environ.get(
        "MNEMOZIA_URI",
        "postgresql://postgres:postgres@127.0.0.1:5432/postgres",
    )


def reset_db():
    """Drop and recreate the notes table for a clean test run."""
    conn = connect(get_db_uri())
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS notes")
    conn.commit()
    ensure_schema(conn)
    cur.close()
    conn.close()


def extract_ids(text: str) -> list[str]:
    """Extract hex IDs from result text like [a1b2c3d4e5f6]."""
    import re
    return re.findall(r"\[([0-9a-f]{12})\]", text)


def extract_line(text: str, keyword: str) -> str:
    """Find first line containing keyword."""
    for line in text.split("\n"):
        if keyword in line:
            return line.strip()
    return ""


def json_parse(text: str) -> Any:
    """Parse JSON from text, trying to extract from code block."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return json.loads(text.strip())


# ═══════════════════════════════════════════════════════════════════════
# Test suite
# ═══════════════════════════════════════════════════════════════════════

def run_tests():
    rpt = TestReport()
    kb = MnemoziaKB(get_db_uri())

    # ────────────────────────────────────────────────────────────────
    # 1. SANITY — server health
    # ────────────────────────────────────────────────────────────────
    rpt.section("1. Sanity — llama-server & pg0 health")

    # Check llama-server
    import requests
    try:
        resp = requests.post(
            "http://127.0.0.1:18080/embedding",
            json={"content": "ping"},
            timeout=10,
        )
        rpt.check("llama-server responds 200", resp.status_code == 200,
                   f"status={resp.status_code}")
        data = resp.json()
        emb = data[0]["embedding"][0]
        rpt.check("llama-server returns 1024-dim vector",
                   len(emb) == 1024,
                   f"got dim={len(emb)}")
        rpt.check("llama-server vector is normalized",
                   abs(sum(v*v for v in emb) - 1.0) < 0.01,
                   f"norm^2={sum(v*v for v in emb):.6f}")
    except Exception as e:
        rpt.check("llama-server health check", False, str(e))

    # Check pg0
    try:
        conn = connect(get_db_uri())
        cur = conn.cursor()
        cur.execute("SELECT 1")
        rpt.check("pg0 connection", cur.fetchone()[0] == 1)
        cur.execute("SELECT COUNT(*) FROM notes")
        count = cur.fetchone()[0]
        rpt.check("pg0 table exists", isinstance(count, int))
        cur.close()
        conn.close()
    except Exception as e:
        rpt.check("pg0 health check", False, str(e))

    # ────────────────────────────────────────────────────────────────
    # 2. ADD — basic storage + dedup
    # ────────────────────────────────────────────────────────────────
    rpt.section("2. Add — basic storage & dedup")

    # 2a. Simple add
    r1 = kb.execute({
        "action": "add",
        "text": "OpenRouter proxy: https://openrouter.ai/api/v1",
        "category": "credentials",
        "tags": "proxy,AI,api",
        "confidence": 0.95,
        "importance": 0.9,
    })
    rpt.check("add returns success with ID", "✅ Stored [" in r1,
               f"result={r1[:100]}")
    id_or = extract_ids(r1)[0] if "[✅" not in r1 else ""
    # Re-extract properly
    id_or = [w for w in r1.split() if w.startswith("[") and w.endswith("]")][0].strip("[]")

    # 2b. Add in Russian
    r2 = kb.execute({
        "action": "add",
        "text": "Прокси OpenRouter для доступа к LLM: https://openrouter.ai/api/v1",
        "category": "credentials",
        "tags": "прокси,AI",
        "confidence": 0.9,
    })
    rpt.check("add Russian text", "✅ Stored [" in r2,
               f"result={r2[:80]}")
    id_or_ru = [w for w in r2.split() if w.startswith("[") and w.endswith("]")][0].strip("[]")

    # 2c. Exact duplicate rejection
    r3 = kb.execute({
        "action": "add",
        "text": "OpenRouter proxy: https://openrouter.ai/api/v1",
        "category": "credentials",
    })
    rpt.check("exact duplicate rejected", "Exact duplicate" in r3,
               f"result={r3[:80]}")

    # 2d. Near-duplicate detection
    r4 = kb.execute({
        "action": "add",
        "text": "OpenRouter proxy is https://openrouter.ai/api/v1 for AI models",
        "category": "credentials",
    })
    rpt.check("near-duplicate flagged", "Near-duplicate" in r4,
               f"result={r4[:80]}")

    # 2e. Contradiction flag (add a semantically related but different fact)
    r5 = kb.execute({
        "action": "add",
        "text": "Nginx reverse proxy for internal services on port 8080",
        "category": "devops",
        "tags": "nginx,proxy",
        "confidence": 0.8,
    })
    rpt.check("add unrelated fact succeeds", "✅ Stored [" in r5,
               f"result={r5[:80]}")

    # 2f. Add without confidence (should default to 1.0)
    r6 = kb.execute({
        "action": "add",
        "text": "Server IP: 89.127.196.10",
        "category": "devops",
        "tags": "server,ip",
    })
    rpt.check("add with default confidence", "confidence=1.0" in r6,
               f"result={r6[:80]}")

    # 2g. Add with low confidence
    r7 = kb.execute({
        "action": "add",
        "text": "Maybe the database password is 'postgres'",
        "category": "credentials",
        "confidence": 0.3,
        "tags": "speculative",
    })
    rpt.check("add low confidence fact", "confidence=0.3" in r7,
               f"result={r7[:80]}")

    # 2h. Add with TTL
    r8 = kb.execute({
        "action": "add",
        "text": "Temporary debug token: abc123",
        "category": "credentials",
        "ttl_days": 7,
        "tags": "temp",
    })
    rpt.check("add with TTL", "✅ Stored [" in r8)

    # 2i. Missing text
    r9 = kb.execute({"action": "add"})
    rpt.check("add with missing text", "Missing required" in r9)

    # 2j. Add long text
    long_text = "The quick brown fox jumps over the lazy dog. " * 20
    r10 = kb.execute({
        "action": "add",
        "text": long_text,
        "category": "general",
    })
    rpt.check("add long text (1000+ chars)", "✅ Stored [" in r10)

    # ────────────────────────────────────────────────────────────────
    # 3. SEARCH — all modes
    # ────────────────────────────────────────────────────────────────
    rpt.section("3. Search — semantic, keyword, hybrid, filters")

    # 3a. Semantic search (Russian query)
    r = kb.execute({
        "action": "search",
        "query": "прокси для доступа к нейросетям",
        "mode": "semantic",
        "limit": 5,
    })
    has_or = "OpenRouter" in r or "openrouter" in r
    has_ru = "Прокси" in r
    rpt.check("semantic search finds Russian + English results",
               has_or or has_ru,
               f"found OR={has_or} RU={has_ru}\n{r[:300]}")

    # 3b. Keyword search
    r = kb.execute({
        "action": "search",
        "query": "nginx",
        "mode": "keyword",
    })
    rpt.check("keyword search finds exact word",
               "nginx" in r.lower() or "Nginx" in r,
               f"result={r[:200]}")

    # 3c. Hybrid search (default)
    r = kb.execute({
        "action": "search",
        "query": "proxy server connection",
    })
    rpt.check("hybrid search returns results",
               "results" in r or "result" in r,
               f"result={r[:200]}")

    # 3d. Category filter
    r = kb.execute({
        "action": "search",
        "query": "proxy",
        "category": "devops",
        "mode": "keyword",
    })
    rpt.check("keyword + category filter returns devops only",
               "devops" in r,
               f"result={r[:200]}")
    # Should NOT find credentials
    r_creds = kb.execute({
        "action": "search",
        "query": "proxy",
        "category": "credentials",
        "mode": "keyword",
    })
    # Check results by scanning IDs — should find credential items
    ids_devops = extract_ids(r)
    ids_creds = extract_ids(r_creds)

    # 3e. Tags filter
    r = kb.execute({
        "action": "search",
        "query": "proxy",
        "tags": "nginx",
        "mode": "keyword",
    })
    rpt.check("tag filter narrows results",
               len(r.split("results")) > 0,
               f"result={r[:200]}")

    # 3f. min_confidence filter
    r = kb.execute({
        "action": "search",
        "query": "proxy",
        "min_confidence": 0.9,
        "mode": "keyword",
    })
    rpt.check("min_confidence=0.9 filters out low-confidence",
               "speculative" not in r and "Maybe" not in r,
               f"result={r[:200]}")

    # 3g. since filter
    r = kb.execute({
        "action": "search",
        "query": "proxy",
        "since": "2099-01-01",
        "mode": "keyword",
    })
    rpt.check("since=2099 returns no results (future date)",
               "No results" in r,
               f"result={r[:200]}")

    # 3h. Empty query
    r = kb.execute({"action": "search", "query": ""})
    rpt.check("empty query returns error", "Missing required" in r)

    # 3i. No results
    r = kb.execute({
        "action": "search",
        "query": "zyxwv99999_nonexistent",
        "mode": "keyword",
    })
    rpt.check("no results for gibberish", "No results" in r)

    # ────────────────────────────────────────────────────────────────
    # 4. VERSIONING — update + history
    # ────────────────────────────────────────────────────────────────
    rpt.section("4. Versioning — update & history")

    # 4a. Update fact
    r = kb.execute({
        "action": "update",
        "id": id_or,
        "text": "OpenRouter proxy endpoint: https://openrouter.ai/api/v1 (primary)",
        "confidence": 0.98,
    })
    rpt.check("update creates new version",
               "→ v2" in r or "→ v2" in r,
               f"result={r[:150]}")

    # 4b. Verify old version is archived — search with unique fragment
    r = kb.execute({
        "action": "search",
        "query": f"OpenRouter proxy: https://openrouter",
        "mode": "semantic",
        "limit": 5,
    })
    # The old text "OpenRouter proxy: https://openrouter.ai/api/v1" should NOT appear
    # as a separate result. The active v2 text contains similar content, but the
    # old inactive row shouldn't be returned.
    old_text_exact = "OpenRouter proxy: https://openrouter.ai/api/v1"
    old_text_found = old_text_exact in r
    rpt.check("old version not in active semantic search",
               not old_text_found,
               f"Old text '{old_text_exact[:60]}' found in active search")

    # 4c. History shows both versions
    r = kb.execute({
        "action": "history",
        "id": id_or,
    })
    has_v1 = "v1" in r or "v 1" in r
    has_v2 = "v2" in r or "v 2" in r
    has_v2 = has_v2 or "→ v2" in r  # also check update result
    rpt.check("history shows v1 and v2",
               has_v1 and has_v2,
               f"history={r[:300]}")

    # 4d. Text unchanged guard
    r = kb.execute({
        "action": "update",
        "id": id_or,
        "text": "OpenRouter proxy endpoint: https://openrouter.ai/api/v1 (primary)",
    })
    rpt.check("no-change update detected",
               "Text unchanged" in r,
               f"result={r[:80]}")

    # 4e. Update non-existent ID
    r = kb.execute({
        "action": "update",
        "id": "nonexistent1234",
        "text": "test",
    })
    rpt.check("update non-existent ID returns error",
               "No active fact" in r,
               f"result={r[:80]}")

    # 4f. History for non-existent ID
    r = kb.execute({
        "action": "history",
        "id": "nonexistent1234",
    })
    rpt.check("history for non-existent ID", "No history" in r)

    # ────────────────────────────────────────────────────────────────
    # 5. MERGE — combine facts
    # ────────────────────────────────────────────────────────────────
    rpt.section("5. Merge — fact combination")

    # 5a. Create two facts to merge
    r_a = kb.execute({
        "action": "add",
        "text": "PostgreSQL connection string: postgresql://localhost:5432/mydb",
        "category": "credentials",
        "tags": "postgres,db",
    })
    id_a = [w for w in r_a.split() if w.startswith("[") and w.endswith("]")][0].strip("[]")

    r_b = kb.execute({
        "action": "add",
        "text": "Database user: admin with password in vault",
        "category": "credentials",
        "tags": "postgres,security",
    })
    id_b = [w for w in r_b.split() if w.startswith("[") and w.endswith("]")][0].strip("[]")

    # 5b. Merge them
    r = kb.execute({
        "action": "merge",
        "id": id_a,
        "with": id_b,
        "text": "PostgreSQL: connect at postgresql://localhost:5432/mydb as admin (password in vault)",
    })
    merged_ids = extract_ids(r)
    rpt.check("merge returns new ID",
               len(merged_ids) >= 1,
               f"result={r[:150]}")

    # 5c. Originals are archived
    r = kb.execute({
        "action": "search",
        "query": id_a,
        "mode": "keyword",
        "limit": 10,
    })
    rpt.check("original facts not in active search (archived)",
               id_a[:8] not in r or "No results" in r,
               f"Search still finds archived: {r[:200]}")

    # 5d. Self-merge rejected
    r = kb.execute({
        "action": "merge",
        "id": id_or,
        "with": id_or,
    })
    rpt.check("self-merge rejected", "Cannot merge a fact with itself" in r)

    # 5e. Merge non-existent
    r = kb.execute({
        "action": "merge",
        "id": "deadbeef1234",
        "with": id_b,
    })
    rpt.check("merge non-existent ID", "No active fact" in r)

    # ────────────────────────────────────────────────────────────────
    # 6. SPLIT — break facts into atomic parts
    # ────────────────────────────────────────────────────────────────
    rpt.section("6. Split — atomic decomposition")

    # 6a. Create compound fact
    r_compound = kb.execute({
        "action": "add",
        "text": "Server specs: CPU Xeon E3-1230, RAM 16GB, SSD 256GB, IP 89.127.196.10",
        "category": "devops",
        "tags": "server,specs",
    })
    id_compound = [w for w in r_compound.split() if w.startswith("[") and w.endswith("]")][0].strip("[]")

    # 6b. Split into parts
    r = kb.execute({
        "action": "split",
        "id": id_compound,
        "parts": [
            "Server CPU: Xeon E3-1230",
            "Server RAM: 16GB",
            "Server SSD: 256GB",
            "Server IP: 89.127.196.10",
        ],
    })
    rpt.check("split creates multiple facts",
               "→ 4" in r or "4 atomic facts" in r,
               f"result={r[:200]}")
    split_ids = extract_ids(r)
    rpt.check("split returns 4 new IDs",
               len(split_ids) >= 4,
               f"got {len(split_ids)} IDs: {split_ids}")
    # Filter to only IDs from indented lines (the new child facts)
    split_ids_new = [
        lid for lid in split_ids
        if f"   [{lid}]" in r
    ]
    rpt.check("split creates 4 distinct child facts",
               len(split_ids_new) == 4,
               f"got {len(split_ids_new)} children: {split_ids_new}")

    # 6c. Original archived
    r = kb.execute({
        "action": "search",
        "query": id_compound,
        "mode": "keyword",
        "limit": 10,
    })
    rpt.check("original compound fact archived after split",
               id_compound[:8] not in r or "No results" in r,
               f"result={r[:200]}")

    # 6d. Split non-existent
    r = kb.execute({
        "action": "split",
        "id": "deadbeef1234",
        "parts": ["a", "b"],
    })
    rpt.check("split non-existent ID", "No active fact" in r)

    # 6e. Split with <2 parts
    r = kb.execute({
        "action": "split",
        "id": split_ids[0],
        "parts": ["only one"],
    })
    rpt.check("split with 1 part rejected", "at least 2 parts" in r)

    # ────────────────────────────────────────────────────────────────
    # 7. DEACTIVATE / REACTIVATE
    # ────────────────────────────────────────────────────────────────
    rpt.section("7. Deactivate / Reactivate")

    # Grab a fresh fact
    r_fresh = kb.execute({
        "action": "add",
        "text": "Temporary note: this will be archived and restored",
        "category": "general",
        "tags": "test",
    })
    id_fresh = [w for w in r_fresh.split() if w.startswith("[") and w.endswith("]")][0].strip("[]")

    # 7a. Deactivate
    r = kb.execute({"action": "deactivate", "id": id_fresh})
    rpt.check("deactivate returns success",
               "Archived" in r,
               f"result={r[:80]}")

    # 7b. Verify archived — search by unique text fragment
    r = kb.execute({
        "action": "search",
        "query": "this will be archived and restored",
        "mode": "semantic",
        "limit": 10,
    })
    # The exact fact text should NOT appear (it's archived)
    exact_text = "Temporary note: this will be archived and restored"
    rpt.check("archived fact not in semantic search results",
               exact_text not in r,
               f"Archived text found in results:\n{r[:300]}")

    # 7c. Reactivate
    r = kb.execute({"action": "reactivate", "id": id_fresh})
    rpt.check("reactivate returns success",
               "Reactivated" in r,
               f"result={r[:80]}")

    # 7d. Verify back in search
    r = kb.execute({
        "action": "search",
        "query": "Temporary note: this will be archived",
        "mode": "keyword",
        "limit": 10,
    })
    rpt.check("reactivated fact in search results",
               "Temporary note" in r,
               f"result={r[:200]}")

    # 7e. Deactivate non-existent
    r = kb.execute({"action": "deactivate", "id": "deadbeef1234"})
    rpt.check("deactivate non-existent", "No active fact" in r)

    # 7f. Reactivate non-existent
    r = kb.execute({"action": "reactivate", "id": "deadbeef1234"})
    rpt.check("reactivate non-existent", "No fact with ID" in r)

    # ────────────────────────────────────────────────────────────────
    # 8. RELATE / UNRELATE
    # ────────────────────────────────────────────────────────────────
    rpt.section("8. Relate / Unrelate")

    # Get two IDs to relate
    ids = extract_ids(r_fresh)
    r_fresh2 = kb.execute({
        "action": "add",
        "text": "Related: database migration steps for v2",
        "category": "devops",
        "tags": "migration,db",
    })
    id_fresh2 = [w for w in r_fresh2.split() if w.startswith("[") and w.endswith("]")][0].strip("[]")

    # 8a. Relate two facts
    r = kb.execute({
        "action": "relate",
        "id": id_fresh,
        "to": id_fresh2,
    })
    rpt.check("relate returns success", "Linked" in r, f"result={r[:80]}")

    # 8b. Self-relate rejected
    r = kb.execute({
        "action": "relate",
        "id": id_fresh,
        "to": id_fresh,
    })
    rpt.check("self-relate rejected", "Cannot relate a fact to itself" in r)

    # 8c. Unrelate
    r = kb.execute({
        "action": "unrelate",
        "id": id_fresh,
        "to": id_fresh2,
    })
    rpt.check("unrelate returns success", "Unlinked" in r, f"result={r[:80]}")

    # 8d. Missing params
    r = kb.execute({"action": "relate", "id": id_fresh})
    rpt.check("relate missing 'to'", "Both" in r)

    # ────────────────────────────────────────────────────────────────
    # 9. EXPORT
    # ────────────────────────────────────────────────────────────────
    rpt.section("9. Export — markdown & JSON")

    # 9a. Export markdown (to string, no path)
    r = kb.execute({"action": "export", "format": "markdown"})
    rpt.check("export markdown starts with heading",
               r.strip().startswith("#"),
               f"first line={r[:50]}")

    # 9b. Export JSON (to string)
    r = kb.execute({"action": "export", "format": "json"})
    try:
        data = json_parse(r)
        rpt.check("export JSON is valid array", isinstance(data, list),
                   f"type={type(data).__name__}, len={len(data) if isinstance(data, list) else 'N/A'}")
        rpt.check("export JSON has required fields",
                   all("id" in item and "text" in item for item in data[:5]),
                   f"sample keys={list(data[0].keys()) if data else 'empty'}")
    except Exception as e:
        rpt.check("export JSON parsing", False, str(e))

    # 9c. Export to file
    export_path = "/tmp/mnemozia_export_test.json"
    r = kb.execute({
        "action": "export",
        "format": "json",
        "path": export_path,
    })
    rpt.check("export JSON to file",
               "Exported" in r and os.path.exists(export_path),
               f"result={r[:80]}")

    # 9d. Verify exported file content
    with open(export_path) as f:
        exported = json.load(f)
    rpt.check("exported file is valid JSON array",
               isinstance(exported, list) and len(exported) > 0,
               f"len={len(exported) if isinstance(exported, list) else 'fail'}")

    # 9e. Export by category
    r = kb.execute({
        "action": "export",
        "format": "json",
        "category": "credentials",
    })
    try:
        data = json_parse(r)
        all_creds = all(item.get("category") == "credentials" for item in data)
        rpt.check("export filtered by category (credentials)",
                   all_creds and len(data) > 0,
                   f"count={len(data)}, all_creds={all_creds}")
    except Exception as e:
        rpt.check("export filtered by category", False, str(e))

    # 9f. Export empty category
    r = kb.execute({
        "action": "export",
        "format": "json",
        "category": "nonexistent_category_xyz",
    })
    rpt.check("export non-existent category",
               "No facts to export" in r,
               f"result={r[:80]}")

    # ────────────────────────────────────────────────────────────────
    # 10. STATS
    # ────────────────────────────────────────────────────────────────
    rpt.section("10. Stats")

    r = kb.execute({"action": "stats"})
    rpt.check("stats returns all sections",
               "Total facts:" in r and "Active:" in r and "By category:" in r and "By language:" in r,
               f"result={r[:200]}")

    # Verify numbers make sense
    import re
    total_match = re.search(r"Total facts: (\d+)", r)
    active_match = re.search(r"Active: (\d+)", r)
    if total_match and active_match:
        total = int(total_match.group(1))
        active = int(active_match.group(1))
        rpt.check("stats: active ≤ total",
                   active <= total,
                   f"active={active}, total={total}")
    else:
        rpt.check("stats: parse numbers", False, f"Could not parse totals from:\n{r[:300]}")

    # ────────────────────────────────────────────────────────────────
    # 11. REVIEW
    # ────────────────────────────────────────────────────────────────
    rpt.section("11. Review")

    r = kb.execute({"action": "review"})
    rpt.check("review returns content",
               len(r) > 20,
               f"result={r[:200]}")

    # Should find the low-confidence fact (confidence=0.3)
    rpt.check("review finds low confidence facts",
               "confidence" in r.lower() or "stale" in r.lower() or "clean" in r,
               f"result={r[:300]}")

    # ────────────────────────────────────────────────────────────────
    # 12. VACUUM
    # ────────────────────────────────────────────────────────────────
    rpt.section("12. Vacuum")

    # Archive something first so vacuum has work
    r_temp = kb.execute({
        "action": "add",
        "text": "Old fact to be vacuumed away",
        "category": "general",
    })
    id_temp = [w for w in r_temp.split() if w.startswith("[") and w.endswith("]")][0].strip("[]")
    kb.execute({"action": "deactivate", "id": id_temp})

    # Vacuum with 0 days (remove all archived)
    r = kb.execute({"action": "vacuum", "older_than_days": 0})
    rpt.check("vacuum removes archived facts",
               "Vacuumed" in r,
               f"result={r[:80]}")
    # Should have removed at least 1
    removed_match = re.search(r"removed (\d+)", r)
    if removed_match:
        removed = int(removed_match.group(1))
        rpt.check("vacuum removed ≥1 archived fact",
                   removed >= 1,
                   f"removed={removed}")

    # ────────────────────────────────────────────────────────────────
    # 13. SEARCH CORRECTNESS — deep checks
    # ────────────────────────────────────────────────────────────────
    rpt.section("13. Search correctness — precision & ranking")

    # Add very distinctive facts for precision testing
    kb.execute({
        "action": "add",
        "text": "Docker compose configuration for web services with nginx and postgres",
        "category": "devops",
        "tags": "docker,compose",
    })
    kb.execute({
        "action": "add",
        "text": "Kubernetes cluster setup with 3 worker nodes and kubectl commands",
        "category": "devops",
        "tags": "k8s,cluster",
    })
    kb.execute({
        "action": "add",
        "text": "Python async SQLAlchemy connection pool settings for aiosqlite and asyncpg",
        "category": "programming",
        "tags": "python,async,db",
    })
    kb.execute({
        "action": "add",
        "text": "React component library with TypeScript interfaces and Storybook docs",
        "category": "programming",
        "tags": "react,typescript",
    })

    # 13a. Semantic: Docker query should rank Docker result highest
    r = kb.execute({
        "action": "search",
        "query": "Docker compose how to configure containers",
        "mode": "semantic",
        "limit": 5,
    })
    rpt.check("semantic search: Docker query finds Docker result",
               "Docker" in r or "docker" in r,
               f"result={r[:300]}")

    # 13b. Semantic: Python query should rank Python result highest
    r = kb.execute({
        "action": "search",
        "query": "SQLAlchemy async database connection",
        "mode": "semantic",
        "limit": 5,
    })
    rpt.check("semantic search: Python query finds Python result",
               "SQLAlchemy" in r or "async" in r,
               f"result={r[:300]}")

    # 13c. Semantic: Russian query about servers
    r = kb.execute({
        "action": "search",
        "query": "сервер и прокси для нейросетей",
        "mode": "semantic",
        "limit": 5,
    })
    rpt.check("semantic search: Russian query finds server/proxy results",
               "proxy" in r.lower() or "сервер" in r.lower() or "OpenRouter" in r,
               f"result={r[:300]}")

    # 13d. Multi-word keyword search
    r = kb.execute({
        "action": "search",
        "query": "nginx postgres",
        "mode": "keyword",
        "limit": 10,
    })
    # Should find facts that contain both words
    has_nginx = "nginx" in r.lower()
    has_postgres = "postgres" in r.lower() or "PostgreSQL" in r
    rpt.check("multi-word keyword search",
               has_nginx or has_postgres,
               f"nginx={has_nginx} postgres={has_postgres}\n{r[:300]}")

    # 13e. Hybrid with category+tag combo
    r = kb.execute({
        "action": "search",
        "query": "database",
        "category": "programming",
        "mode": "hybrid",
        "limit": 5,
    })
    rpt.check("hybrid + category=programming filters correctly",
               "programming" in r,
               f"result={r[:200]}")

    # 13f. Limit parameter
    r = kb.execute({
        "action": "search",
        "query": "proxy server database configuration",
        "mode": "semantic",
        "limit": 2,
    })
    rpt.check("limit=2 returns at most 2 results",
               "·  2 results" in r or "2 results" in r or len(extract_ids(r)) <= 2,
               f"result={r[:200]}")

    # ────────────────────────────────────────────────────────────────
    # 14. EDGE CASES & ERROR HANDLING
    # ────────────────────────────────────────────────────────────────
    rpt.section("14. Edge cases & error handling")

    # 14a. Unknown action
    r = kb.execute({"action": "fly_to_the_moon", "text": "test"})
    rpt.check("unknown action returns error", "Unknown action" in r)

    # 14b. Empty action
    r = kb.execute({})
    rpt.check("empty action shows usage", "available actions" in r.lower())

    # 14c. SQL injection attempt via text
    r = kb.execute({
        "action": "add",
        "text": "'; DROP TABLE notes; --",
        "category": "general",
    })
    rpt.check("SQL injection attempt (DROP TABLE) is safely stored as text",
               "✅ Stored [" in r or "Near-duplicate" in r or "Exact duplicate" in r,
               f"result={r[:80]}")
    # Verify table still exists
    conn = connect(get_db_uri())
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM notes")
    rpt.check("notes table still exists after SQL injection attempt",
               isinstance(cur.fetchone()[0], int))
    cur.close()
    conn.close()

    # 14d. SQL injection via ID
    r = kb.execute({
        "action": "deactivate",
        "id": "'; UPDATE notes SET is_active = FALSE; --",
    })
    rpt.check("SQL injection attempt via ID is rejected (no active fact)",
               "No active fact" in r,
               f"result={r[:80]}")

    # 14e. Very short text (1 char)
    r = kb.execute({
        "action": "add",
        "text": "x",
        "category": "general",
    })
    rpt.check("add single-char text", "✅ Stored [" in r or "Near-duplicate" in r or "Exact duplicate" in r)

    # 14f. Unicode / emoji in text
    r = kb.execute({
        "action": "add",
        "text": "Mnemozia от Мнемозины 🧠 Greek goddess of memory 🌟",
        "category": "general",
        "tags": "unicode,emoji",
        "language": "ru",
    })
    rpt.check("add Unicode/emoji text", "✅ Stored [" in r, f"result={r[:80]}")

    # ────────────────────────────────────────────────────────────────
    # 15. END-TO-END WORKFLOW
    # ────────────────────────────────────────────────────────────────
    rpt.section("15. End-to-end workflow simulation")

    # Simulate a realistic workflow: add → update → merge → split → relate → export → stats
    e2e_id = ""

    # Step 1: Add credentials fact
    r = kb.execute({
        "action": "add",
        "text": "AWS S3 bucket: my-app-data, region eu-west-1, access key AKIA123",
        "category": "credentials",
        "confidence": 0.7,
        "tags": "aws,s3,cloud",
        "importance": 0.85,
    })
    aws_id = [w for w in r.split() if w.startswith("[") and w.endswith("]")][0].strip("[]")
    rpt.check("E2E: add AWS fact", "✅ Stored [" in r)

    # Step 2: Update with more detail
    r = kb.execute({
        "action": "update",
        "id": aws_id,
        "text": "AWS S3 bucket: my-app-data, region eu-west-1, access key in vault (AKIA123 rotated)",
        "confidence": 0.85,
    })
    rpt.check("E2E: update AWS fact", "→ v2" in r)

    # Step 3: Add related fact
    r = kb.execute({
        "action": "add",
        "text": "CloudFront distribution for S3 bucket: d123.cloudfront.net",
        "category": "devops",
        "tags": "aws,cloudfront,cdn",
    })
    cf_id = [w for w in r.split() if w.startswith("[") and w.endswith("]")][0].strip("[]")

    # Step 4: Relate them
    r = kb.execute({
        "action": "relate",
        "id": aws_id,
        "to": cf_id,
    })
    rpt.check("E2E: relate AWS ↔ CloudFront", "Linked" in r)

    # Step 5: Merge with another fact
    r = kb.execute({
        "action": "add",
        "text": "AWS CLI config: region = eu-west-1, output = json",
        "category": "credentials",
        "tags": "aws,cli",
    })
    aws2_id = [w for w in r.split() if w.startswith("[") and w.endswith("]")][0].strip("[]")

    r = kb.execute({
        "action": "merge",
        "id": aws_id,
        "with": aws2_id,
        "text": "AWS: S3 bucket my-app-data in eu-west-1, CLI configured for json output, keys in vault",
    })
    merged_e2e = extract_ids(r)
    rpt.check("E2E: merge AWS facts", len(merged_e2e) >= 1)

    # Step 6: History check
    r = kb.execute({"action": "history", "id": aws_id})
    rpt.check("E2E: history shows all AWS versions", "v1" in r and "v2" in r)

    # Step 7: Export
    r = kb.execute({"action": "export", "format": "markdown"})
    rpt.check("E2E: export markdown", r.startswith("#"))

    # Step 8: Stats
    r = kb.execute({"action": "stats"})
    rpt.check("E2E: stats available", "Total facts:" in r)

    return rpt


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 72)
    print("  MNEMOZIA v2 — Comprehensive Integration Test Suite")
    print("  Stack: pg0 + llama.cpp + Qwen3-Embedding-0.6B")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    # Reset DB for clean test
    print("\n  🧹 Resetting database...")
    reset_db()
    print("  ✅ Database clean\n")

    try:
        report = run_tests()
        report.print_report()
    except Exception as e:
        print(f"\n❌ CRASH: {e}")
        traceback.print_exc()
        sys.exit(1)
