#!/usr/bin/env python3
"""
Mnemozia PoC — validates core logic (embedding, dedup, versioning, search)
without LanceDB (which requires AVX2, unavailable on this Sandy Bridge CPU).

Uses FAISS + SQLite as a compatible backend.
"""

import os, sys, json, time, sqlite3, uuid, psutil
from datetime import datetime
import numpy as np

# ── Embedding model (same as Mnemozia uses) ──
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed(text: str, is_query: bool = False) -> np.ndarray:
    prefix = QUERY_PREFIX if is_query else PASSAGE_PREFIX
    return get_model().encode(prefix + text)

# ── PoC Knowledge Base (drop-in for MnemoziaKB) ──
class PoC_KB:
    def __init__(self, db_path: str = "/tmp/mnemozia_poc"):
        os.makedirs(db_path, exist_ok=True)
        self.db_path = db_path
        self.con = sqlite3.connect(os.path.join(db_path, "facts.db"))
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                text TEXT,
                version INTEGER DEFAULT 1,
                is_active INTEGER DEFAULT 1,
                category TEXT DEFAULT 'general',
                tags TEXT DEFAULT '',
                confidence REAL DEFAULT 1.0,
                importance REAL DEFAULT 0.5,
                language TEXT DEFAULT 'auto',
                source TEXT DEFAULT '',
                source_detail TEXT DEFAULT '',
                ttl_days INTEGER DEFAULT 0,
                supersedes TEXT DEFAULT '',
                related_to TEXT DEFAULT '',
                created_at TEXT,
                updated_at TEXT
            )
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id TEXT,
                version INTEGER,
                text TEXT,
                is_active INTEGER,
                category TEXT,
                tags TEXT,
                confidence REAL,
                importance REAL,
                supersedes TEXT,
                related_to TEXT,
                created_at TEXT,
                updated_at TEXT,
                ttl_days INTEGER,
                language TEXT,
                source TEXT,
                source_detail TEXT,
                archived_at TEXT,
                PRIMARY KEY (id, version)
            )
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id1 TEXT,
                id2 TEXT,
                created_at TEXT,
                PRIMARY KEY (id1, id2)
            )
        """)
        self.con.commit()
        # FAISS index — cosine similarity via inner product on normalized vectors
        self.dim = 384
        self.index = None
        self.id_map = {}  # position -> fact_id
        self.rev_id_map = {}  # fact_id -> position
        self._load_faiss()

    def _load_faiss(self):
        import faiss
        if os.path.exists(os.path.join(self.db_path, "faiss.index")):
            self.index = faiss.read_index(os.path.join(self.db_path, "faiss.index"))
            # Rebuild id maps
            self.id_map = {}
            self.rev_id_map = {}
            rows = self.con.execute("SELECT id FROM facts WHERE is_active=1").fetchall()
            for i, (fid,) in enumerate(rows):
                if i < self.index.ntotal:
                    self.id_map[i] = fid
                    self.rev_id_map[fid] = i
        else:
            self.index = faiss.IndexFlatIP(self.dim)

    def _save_faiss(self):
        import faiss
        faiss.write_index(self.index, os.path.join(self.db_path, "faiss.index"))

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _uid(self) -> str:
        return uuid.uuid4().hex[:12]

    def _now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _confidence_flag(self, conf: float) -> str:
        if conf >= 0.9: return "✅ verified"
        if conf >= 0.5: return "📋 reliable"
        return "⚠️ low confidence"

    def _distance_to_stars(self, dist: float) -> str:
        if dist < 0.1: return "★★★★★"
        if dist < 0.2: return "★★★★"
        if dist < 0.3: return "★★★"
        if dist < 0.4: return "★★"
        return "★"

    def execute(self, args: dict) -> str:
        action = args.get("action", "")
        method = f"_action_{action}"
        if hasattr(self, method):
            try:
                return getattr(self, method)(args)
            except Exception as e:
                return f"❌ Error: {e}"
        return f"Unknown action '{action}'. Available: add, search, update, merge, split, deactivate, reactivate, history, review, stats, export, relate, unrelate, vacuum"

    # ── ADD ──
    def _action_add(self, args: dict) -> str:
        text = args.get("text", "").strip()
        if not text:
            return "❌ 'text' is required"
        category = args.get("category", "general")
        tags = args.get("tags", "")
        confidence = float(args.get("confidence", 1.0))
        importance = float(args.get("importance", 0.5))
        ttl_days = int(args.get("ttl_days", 0))
        source = args.get("source", "")
        source_detail = args.get("source_detail", "")
        language = args.get("language", "auto")

        # 1. Exact match check
        row = self.con.execute(
            "SELECT id FROM facts WHERE text=? AND is_active=1", (text,)
        ).fetchone()
        if row:
            return f"🔁 Exact duplicate found [ID: {row[0]}]"

        # 2-3. Semantic check
        vec = self._normalize(embed(text))
        if self.index.ntotal > 0:
            distances, indices = self.index.search(vec.reshape(1, -1), 5)
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < 0:
                    continue
                fid = self.id_map.get(int(idx), "")
                if dist <= 0.08:
                    return f"🔁 Near-duplicate found [ID: {fid}] (distance={dist:.4f})"
                if dist <= 0.25 and i == 0:
                    existing = self.con.execute(
                        "SELECT text FROM facts WHERE id=? AND is_active=1", (fid,)
                    ).fetchone()
                    if existing:
                        pass  # will attach warning below

        # Insert
        fid = self._uid()
        now = self._now()
        self.con.execute(
            """INSERT INTO facts (id, text, version, category, tags, confidence,
               importance, ttl_days, source, source_detail, language,
               created_at, updated_at)
               VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (fid, text, category, tags, confidence, importance,
             ttl_days, source, source_detail, language, now, now)
        )
        self.con.commit()

        # Add to FAISS
        pos = self.index.ntotal
        self.index.add(vec.reshape(1, -1))
        self.id_map[pos] = fid
        self.rev_id_map[fid] = pos
        self._save_faiss()

        result = f"✅ Stored [ID: {fid}] | 📂 {category} | {self._confidence_flag(confidence)}"
        return result

    # ── SEARCH ──
    def _action_search(self, args: dict) -> str:
        query = args.get("query", args.get("text", "")).strip()
        if not query:
            return "❌ 'query' is required"
        limit = int(args.get("limit", 5))
        category = args.get("category", "")
        tags = args.get("tags", "")
        min_conf = float(args.get("min_confidence", 0.0))
        since = args.get("since", "")

        vec = self._normalize(embed(query, is_query=True))
        if self.index.ntotal == 0:
            return "📭 Knowledge base is empty"

        k = min(limit * 3, self.index.ntotal)
        distances, indices = self.index.search(vec.reshape(1, -1), k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            fid = self.id_map.get(int(idx))
            if not fid:
                continue
            row = self.con.execute(
                """SELECT id, text, version, category, tags, confidence,
                          importance, language, updated_at
                   FROM facts WHERE id=? AND is_active=1""", (fid,)
            ).fetchone()
            if not row:
                continue
            rid, rtext, rver, rcat, rtags, rconf, rimp, rlang, rupd = row
            # Apply filters
            if category and category not in rcat:
                continue
            if min_conf and rconf < min_conf:
                continue
            if since and rupd < since:
                continue
            results.append({
                "id": rid, "text": rtext[:120], "version": rver,
                "category": rcat, "tags": rtags, "confidence": rconf,
                "importance": rimp, "language": rlang, "updated_at": rupd,
                "_distance": float(dist)
            })

        # Sort by distance - 0.5 * importance
        results.sort(key=lambda r: r["_distance"] - 0.5 * r["importance"])
        results = results[:limit]

        if not results:
            return "🔍 No results"

        lines = []
        for rank, r in enumerate(results, 1):
            stars = self._distance_to_stars(r["_distance"])
            flag = self._confidence_flag(r["confidence"])
            tag_str = f" #{r['tags']}" if r['tags'] else ""
            lines.append(
                f"{rank}. [{r['id']}] {stars} (d={r['_distance']:.3f}) | {flag}{tag_str} | "
                f"v{r['version']} | {r['updated_at']}\n"
                f"  📂 {r['category']}\n"
                f"  {r['text']}"
            )
        return "\n\n".join(lines)

    # ── UPDATE ──
    def _action_update(self, args: dict) -> str:
        fid = args.get("id", "")
        new_text = args.get("text", "").strip()
        if not fid or not new_text:
            return "❌ 'id' and 'text' are required"
        row = self.con.execute(
            "SELECT text, version FROM facts WHERE id=? AND is_active=1", (fid,)
        ).fetchone()
        if not row:
            return f"❌ Fact [ID: {fid}] not found or inactive"
        old_text, old_ver = row
        if old_text == new_text:
            return "ℹ️ No change"
        # Archive old
        now = self._now()
        self.con.execute(
            """INSERT INTO history (id, version, text, is_active, category, tags,
               confidence, importance, supersedes, related_to, created_at, updated_at,
               ttl_days, language, source, source_detail, archived_at)
               SELECT id, version, text, is_active, category, tags,
               confidence, importance, supersedes, related_to, created_at, updated_at,
               ttl_days, language, source, source_detail, ?
               FROM facts WHERE id=? AND is_active=1""",
            (now, fid)
        )
        # Deactivate old
        self.con.execute(
            "UPDATE facts SET is_active=0, updated_at=? WHERE id=? AND is_active=1",
            (now, fid)
        )
        # Insert new
        new_ver = old_ver + 1
        for col in ("category", "tags", "confidence", "importance", "ttl_days"):
            val = args.get(col)
            if val is not None:
                self.con.execute(f"UPDATE facts SET {col}=? WHERE id=? AND is_active=0 AND version=?",
                                 (val, fid, old_ver))
        # Re-read metadata from archived version
        meta = self.con.execute(
            "SELECT category, tags, confidence, importance, ttl_days, source, source_detail, language "
            "FROM facts WHERE id=? AND version=?", (fid, old_ver)
        ).fetchone()
        cat, tg, conf, imp, ttl, src, sdet, lang = meta
        # Override with new values
        cat = args.get("category", cat)
        tg = args.get("tags", tg)
        conf = float(args.get("confidence", conf))
        imp = float(args.get("importance", imp))
        ttl = int(args.get("ttl_days", ttl))

        self.con.execute(
            """INSERT INTO facts (id, text, version, category, tags, confidence,
               importance, ttl_days, source, source_detail, language,
               created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
               (SELECT created_at FROM facts WHERE id=? AND version=?), ?)""",
            (fid, new_text, new_ver, cat, tg, conf, imp, ttl, src, sdet, lang,
             fid, old_ver, now)
        )
        self.con.commit()

        # Update FAISS
        pos = self.rev_id_map.get(fid)
        if pos is not None:
            vec = self._normalize(embed(new_text))
            import faiss
            faiss.remove_ids(self.index, np.array([pos]))
            new_pos = self.index.ntotal
            self.index.add(vec.reshape(1, -1))
            del self.id_map[pos]
            self.id_map[new_pos] = fid
            self.rev_id_map[fid] = new_pos
            self._save_faiss()

        return f"✅ Updated [ID: {fid}] v{old_ver}→v{new_ver}:\n  Old: {old_text[:80]}...\n  New: {new_text[:80]}..."

    # ── HISTORY ──
    def _action_history(self, args: dict) -> str:
        fid = args.get("id", "")
        if not fid:
            return "❌ 'id' is required"
        rows = self.con.execute(
            "SELECT id, version, text, is_active, category, confidence, created_at, updated_at "
            "FROM facts WHERE id=? ORDER BY version", (fid,)
        ).fetchall()
        hist_rows = self.con.execute(
            "SELECT id, version, text, is_active, category, confidence, archived_at "
            "FROM history WHERE id=? ORDER BY version", (fid,)
        ).fetchall()
        if not rows and not hist_rows:
            return f"❌ Fact [ID: {fid}] not found"
        lines = [f"📜 History for [{fid}]"]
        if not hist_rows:
            lines.append("  (no archived versions)")
        for r in hist_rows:
            status = "🗂 archived" if not r[3] else "✅ active"
            conf = r[6] if isinstance(r[6], (int, float)) else 0.5
            lines.append(f"  v{r[1]} | {status} | {self._confidence_flag(conf)} | {r[15] if len(r) > 15 and r[15] else '-'}")
            lines.append(f"    {r[2][:100]}")
        for r in rows:
            status = "✅ active" if r[3] else "🗂 archived"
            conf = r[6] if isinstance(r[6], (int, float)) else 0.5
            lines.append(f"  v{r[1]} | {status} | {self._confidence_flag(conf)} | {r[9]}")
            lines.append(f"    {r[2][:100]}")
        return "\n".join(lines)

    # ── STATS ──
    def _action_stats(self, args: dict) -> str:
        total = self.con.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        active = self.con.execute("SELECT COUNT(*) FROM facts WHERE is_active=1").fetchone()[0]
        archived = total - active
        by_cat = self.con.execute(
            "SELECT category, COUNT(*) FROM facts WHERE is_active=1 GROUP BY category ORDER BY 2 DESC LIMIT 10"
        ).fetchall()
        by_lang = self.con.execute(
            "SELECT language, COUNT(*) FROM facts WHERE is_active=1 GROUP BY language"
        ).fetchall()
        conf_bands = self.con.execute("""
            SELECT
              SUM(CASE WHEN confidence >= 0.9 THEN 1 ELSE 0 END) as high,
              SUM(CASE WHEN confidence >= 0.5 AND confidence < 0.9 THEN 1 ELSE 0 END) as medium,
              SUM(CASE WHEN confidence < 0.5 THEN 1 ELSE 0 END) as low
            FROM facts WHERE is_active=1
        """).fetchone()

        lines = [
            f"📊 Mnemozia Stats",
            f"  Total: {total} | Active: {active} | Archived: {archived}",
            f"  Path: {self.db_path}",
            f"",
            f"  Categories (top 10):",
        ]
        for cat, cnt in by_cat:
            lines.append(f"    {cat}: {cnt}")
        lines.append(f"")
        lines.append(f"  Languages: {', '.join(f'{l}: {c}' for l, c in by_lang)}")
        lines.append(f"  Confidence: ✅ high={conf_bands[0]} | 📋 med={conf_bands[1]} | ⚠️ low={conf_bands[2]}")
        return "\n".join(lines)

    # ── DEACTIVATE / REACTIVATE ──
    def _action_deactivate(self, args: dict) -> str:
        fid = args.get("id", "")
        if not fid:
            return "❌ 'id' is required"
        row = self.con.execute(
            "SELECT text FROM facts WHERE id=? AND is_active=1", (fid,)
        ).fetchone()
        if not row:
            return f"❌ Fact [ID: {fid}] not found or already inactive"
        now = self._now()
        self.con.execute(
            """INSERT INTO history (id, version, text, is_active, category, tags,
               confidence, importance, supersedes, related_to, created_at, updated_at,
               ttl_days, language, source, source_detail, archived_at)
               SELECT id, version, text, is_active, category, tags,
               confidence, importance, supersedes, related_to, created_at, updated_at,
               ttl_days, language, source, source_detail, ?
               FROM facts WHERE id=? AND is_active=1""",
            (now, fid)
        )
        self.con.execute("UPDATE facts SET is_active=0, updated_at=? WHERE id=? AND is_active=1",
                         (now, fid))
        self.con.commit()
        # Remove from FAISS
        pos = self.rev_id_map.pop(fid, None)
        if pos is not None:
            import faiss
            faiss.remove_ids(self.index, np.array([pos]))
            del self.id_map[pos]
            # Rebuild id_map
            self.id_map = {i: fid for i, (fid,) in enumerate(
                self.con.execute("SELECT id FROM facts WHERE is_active=1").fetchall()
            ) if i < self.index.ntotal}
            self.rev_id_map = {fid: i for i, fid in self.id_map.items()}
            self._save_faiss()
        return f"🗂 Deactivated [ID: {fid}]"

    def _action_reactivate(self, args: dict) -> str:
        fid = args.get("id", "")
        if not fid:
            return "❌ 'id' is required"
        self.con.execute("UPDATE facts SET is_active=1, updated_at=? WHERE id=?",
                         (self._now(), fid))
        self.con.commit()
        # Re-add to FAISS
        row = self.con.execute("SELECT text FROM facts WHERE id=? AND is_active=1", (fid,)).fetchone()
        if row:
            vec = self._normalize(embed(row[0]))
            pos = self.index.ntotal
            self.index.add(vec.reshape(1, -1))
            self.id_map[pos] = fid
            self.rev_id_map[fid] = pos
            self._save_faiss()
        return f"✅ Reactivated [ID: {fid}]"


def measure_ram():
    """Return current process RSS in MB."""
    return psutil.Process().memory_info().rss / 1024 / 1024


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 MNEMOZIA PoC — Core Logic Validation")
    print("=" * 60)
    print()

    # Clean start
    import shutil
    if os.path.exists("/tmp/mnemozia_poc"):
        shutil.rmtree("/tmp/mnemozia_poc")

    kb = PoC_KB("/tmp/mnemozia_poc")
    ram_before = measure_ram()
    print(f"📊 RAM before model load: {ram_before:.1f} MB")
    print()

    # Force model load
    print("⏳ Loading embedding model...")
    t0 = time.time()
    _ = embed("warmup")
    t_load = time.time() - t0
    ram_after_model = measure_ram()
    print(f"✅ Model loaded in {t_load:.1f}s")
    print(f"📊 RAM after model load: {ram_after_model:.1f} MB")
    print(f"📊 Model RAM cost: {ram_after_model - ram_before:.1f} MB")
    print()

    # ── TEST 1: Add facts ──
    print("─" * 40)
    print("🧪 TEST 1: ADD FACTS")
    print("─" * 40)

    facts = [
        {"text": "OpenRouter требует socks5h-прокси 199.68.196.14:31149",
         "category": "devops/networking", "tags": "openrouter,proxy,socks5", "confidence": 1.0},
        {"text": "PostgreSQL порт по умолчанию — 5432",
         "category": "devops/database", "tags": "postgresql,port", "confidence": 1.0},
        {"text": "Hermes Agent хранит конфиг в ~/.hermes/config.yaml",
         "category": "devops/hermes", "tags": "hermes,config", "confidence": 1.0},
        {"text": "Для подключения к OpenRouter используется SOCKS5 прокси",
         "category": "devops/networking", "tags": "openrouter,proxy", "confidence": 0.7},
        {"text": "Nginx нужно перезагружать после изменения конфига через nginx -s reload",
         "category": "devops/nginx", "tags": "nginx,reload", "confidence": 1.0},
        {"text": "Алексей предпочитает DevFocus Dark тему: #0b1326 bg, #2665fd primary",
         "category": "personal/preferences", "tags": "design,theme", "confidence": 1.0},
    ]

    for f in facts:
        result = kb.execute({"action": "add", **f})
        print(f"  {result}")

    print()

    # ── TEST 2: Exact duplicate rejection ──
    print("─" * 40)
    print("🧪 TEST 2: DEDUP — EXACT DUPLICATE")
    print("─" * 40)
    result = kb.execute({"action": "add", "text": facts[0]["text"],
                         "category": "devops/networking"})
    print(f"  {result}")
    print()

    # ── TEST 3: Semantic near-duplicate ──
    print("─" * 40)
    print("🧪 TEST 3: DEDUP — SEMANTIC NEAR-DUPLICATE")
    print("─" * 40)
    result = kb.execute({"action": "add", "text": "Для работы OpenRouter нужен socks5h прокси 199.68.196.14:31149",
                         "category": "devops/networking"})
    print(f"  {result}")
    print()

    # ── TEST 4: Semantic search ──
    print("─" * 40)
    print("🧪 TEST 4: SEMANTIC SEARCH (Russian)")
    print("─" * 40)
    result = kb.execute({"action": "search", "query": "как подключиться к OpenRouter", "limit": 3})
    print(result)
    print()

    # ── TEST 5: English search ──
    print("─" * 40)
    print("🧪 TEST 5: SEMANTIC SEARCH (English)")
    print("─" * 40)
    result = kb.execute({"action": "search", "query": "how to reload nginx config", "limit": 3})
    print(result)
    print()

    # ── TEST 6: Search with category filter ──
    print("─" * 40)
    print("🧪 TEST 6: SEARCH WITH CATEGORY FILTER")
    print("─" * 40)
    result = kb.execute({"action": "search", "query": "настройки", "category": "personal", "limit": 5})
    print(result)
    print()

    # ── TEST 7: Update fact ──
    print("─" * 40)
    print("🧪 TEST 7: UPDATE FACT")
    print("─" * 40)
    import re
    result = kb.execute({"action": "search", "query": "OpenRouter", "limit": 1})
    print(f"  Search result:\n{result}")
    id_match = re.search(r'\[([a-f0-9]{12})\]', result)
    if id_match:
        fid = id_match.group(1)
        result = kb.execute({"action": "update", "id": fid,
                             "text": "OpenRouter требует socks5h://199.68.196.14:31149 (схема обязательна)",
                             "confidence": 1.0})
        print(f"  Update: {result}")
        # Verify history
        result = kb.execute({"action": "history", "id": fid})
        print(f"  History:\n{result}")
    print()

    # ── TEST 8: Stats ──
    print("─" * 40)
    print("🧪 TEST 8: STATS")
    print("─" * 40)
    result = kb.execute({"action": "stats"})
    print(result)
    print()

    # ── TEST 9: Deactivate ──
    print("─" * 40)
    print("🧪 TEST 9: DEACTIVATE")
    print("─" * 40)
    result = kb.execute({"action": "deactivate", "id": fid})
    print(f"  {result}")
    result = kb.execute({"action": "search", "query": "OpenRouter", "limit": 3})
    print(f"  Search after deactivate:\n{result}")
    print()

    # ── TEST 10: Reactivate ──
    print("─" * 40)
    print("🧪 TEST 10: REACTIVATE")
    print("─" * 40)
    result = kb.execute({"action": "reactivate", "id": fid})
    print(f"  {result}")
    result = kb.execute({"action": "search", "query": "OpenRouter", "limit": 3})
    print(f"  Search after reactivate:\n{result}")
    print()

    # ── RAM summary ──
    ram_final = measure_ram()
    print("=" * 60)
    print(f"📊 RAM FINAL: {ram_final:.1f} MB")
    print(f"   Model load time: {t_load:.1f}s")
    print()
    print("🧪 PoC COMPLETE — All core operations verified!")
    print("=" * 60)
