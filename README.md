# Mnemozia

> 🇷🇺 [Русская версия](README_ru.md)

**Semantic knowledge base for [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

Named after Mnemosyne (Μνημοσύνη), the Greek goddess of memory. Stores facts, retrieves them
by meaning (not exact keywords), and tracks how knowledge evolves with full version history.

**Stack:**
- **Storage:** [pg0](https://github.com/vectorize-io/pg0) (PostgreSQL 18 + pgvector) — zero-config vector DB
- **Embeddings:** [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) via [llama.cpp](https://github.com/ggml-org/llama.cpp) — 1024-dim, 100+ languages, 4-bit quantized (378 MB)
- **Inference:** `llama-server` as a systemd service, communicates via HTTP API — no SIGILL on older CPUs

Created with the help of [Hermes Agent](https://github.com/NousResearch/hermes-agent) on the DeepSeek model.

## Features

- **14 operations:** add, search, update, merge, split, deactivate, reactivate, history, review, stats, export, relate, unrelate, vacuum
- **3-stage dedup:** exact text → semantic near-duplicate → contradiction flag — no duplicates sneak in
- **Semantic search:** vector search via multilingual-e5-small — find by meaning in Russian or English
- **Full versioning:** every update creates a new version; `history` shows the complete audit trail
- **Merge & split:** combine overlapping facts or break complex ones into atomic pieces
- **Confidence scoring:** 0.0 (hypothesis) → 1.0 (verified fact) — the agent knows when to trust
- **LLM-optimised output:** search results include distance scores, confidence flags, merge tips
- **Lazy loading:** the embedding model only loads on first use — zero RAM until you need it
- **Hermes-native:** install as a plugin (git clone to `~/.hermes/plugins/`) or standalone

## Install

```bash
# Option A: Hermes plugin (recommended)
git clone https://github.com/megazhuk/Mnemozia.git ~/.hermes/plugins/mnemozia
pip install -r ~/.hermes/plugins/mnemozia/requirements.txt
hermes tools enable mnemozia
# Then type /reset in your Hermes chat, or restart Hermes

# Option B: Standalone Python (install from git)
pip install git+https://github.com/megazhuk/Mnemozia.git
```

## Quick Start

```python
from mnemozia import MnemoziaKB

kb = MnemoziaKB("~/.hermes/knowledge_base")

# Store facts
kb.execute({"action": "add", "text": "OpenRouter требует socks5h-прокси 199.68.196.14:31149", "category": "devops/networking", "tags": "openrouter,proxy,socks5", "confidence": 1.0})

# Find by meaning
kb.execute({"action": "search", "query": "как подключиться к OpenRouter", "limit": 3})

# Update with version tracking
kb.execute({"action": "update", "id": "a1b2c3d4e5f6", "text": "OpenRouter прокси: socks5h://199.68.196.14:31149 (схема обязательна)", "confidence": 1.0})

# Merge duplicates
kb.execute({"action": "merge", "id": "a1b2c3", "with": "d4e5f6", "text": "объединённый текст"})
```

## Operations

| Action | Description |
|--------|-------------|
| `add` | Store a fact (auto-dedup: exact → near-duplicate → contradiction flag) |
| `search` | Semantic vector search (LanceDB 0.33: semantic-only; keyword mode planned for future LanceDB) |
| `update` | New version of an existing fact (preserves history) |
| `merge` | Combine two facts into one (originals archived) |
| `split` | Break a complex fact into atomic parts |
| `deactivate` | Soft-delete (archived, recoverable) |
| `reactivate` | Restore an archived fact |
| `history` | Full version history of a fact |
| `review` | Show facts needing attention (low confidence, stale) |
| `stats` | Totals, by category, confidence distribution |
| `export` | Export as Markdown or JSON |
| `relate` | Link two facts |
| `unrelate` | Remove link between facts |
| `vacuum` | Hard-delete old archived rows |

## Categories

`general`, `work`, `personal`, `finance`, `credentials`, `ideas`, `tech`, `devops`, `programming`, `schedule`, `contacts`, `health`, `travel`, `learning` — plus hierarchical via `/` (e.g. `devops/networking/proxy`).

## Confidence Levels

| Level | Meaning |
|-------|---------|
| 1.0 | Verified fact |
| 0.7–0.9 | Reliable |
| 0.5–0.7 | Likely |
| 0.3–0.5 | Hypothesis |
| 0.0–0.3 | Speculation |

## License

MIT
