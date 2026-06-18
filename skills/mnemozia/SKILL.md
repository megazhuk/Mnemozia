---
name: mnemozia
description: Semantic knowledge base — store facts, retrieve by meaning, track evolution over time.
version: 1.0.0
author: Hermes Community
license: MIT
tags: [memory, knowledge, semantic-search, versioning, facts]
metadata:
  hermes:
    tool: kb
    toolset: mnemozia
---

# Mnemozia — Semantic Knowledge Base

Mnemozia (Μνημοσία, from Μνημοσύνη — the Greek goddess of memory) is your **long-term semantic memory**.
It stores facts, retrieves them by meaning (not exact keywords), and tracks how knowledge evolves
with full version history. Built on LanceDB + multilingual embeddings — works in Russian and English.

**When to use `kb` instead of built-in `memory`:**
- You need to find facts by *meaning*, not by exact ID or keyword
- A fact has evolved over time and you need the audit trail
- You want to merge two overlapping facts, or split a complex one into atomic pieces
- You need to export your knowledge base as Markdown

## Core Rules

### 1. Dedup is automatic — but you must read the response

The `add` action runs three checks: exact text match → semantic near-duplicate → contradiction flag.
**If add returns a near-duplicate warning**, extract the ID and use `update` or `merge` — do NOT retry add.

### 2. Search before you answer

When the user asks a factual question, search Mnemozia before answering:
```
kb(action="search", query="<user's question rephrased as keywords>", limit=3)
```
If a fact exists with `confidence > 0.8`, cite it with its ID and timestamp:
> According to your note [a1b2c3] from June 15: «OpenRouter requires socks5h-proxy…»

### 3. Confidence levels

| Level | Meaning | When to assign |
|-------|---------|----------------|
| 1.0 | Verified fact | User explicitly confirmed; tested in practice; from official docs |
| 0.7–0.9 | Reliable | Observed in practice but not explicitly confirmed; from a trusted source |
| 0.5–0.7 | Likely | Inferred from context; user mentioned indirectly |
| 0.3–0.5 | Hypothesis | Your best guess; needs verification |
| 0.0–0.3 | Speculation | Wild guess — avoid storing these |

**Lower confidence on new facts.** Default to 0.7 unless the user explicitly confirmed.
When you later verify a fact, `update` it with `confidence=1.0`.

### 4. Temporal awareness

When retrieving facts, incorporate timestamps into your response:
> Your note from January 12 said the deadline was March. But the update from February 5 moved it to May.

### 5. Auto-categorisation

Classify every fact into one of these categories (lowercase):
- `general` — uncategorised
- `work` — projects, tasks, professional
- `personal` — private information
- `finance` — money, invoices, payments
- `credentials` — logins, API keys, tokens (⚠️ high sensitivity — never expose in public)
- `ideas` — brainstorming, future plans
- `tech` — general technical knowledge
- `devops` — deployment, infrastructure, servers, proxies
- `programming` — code patterns, libraries, frameworks
- `schedule` — deadlines, appointments
- `contacts` — people, organisations
- `health` — medical, wellness
- `travel` — trips, locations
- `learning` — tutorials, courses, study notes

Use hierarchical categories with `/`: `devops/networking/proxy`, `programming/python/fastapi`.

### 6. Update vs Merge

| Situation | Action |
|-----------|--------|
| Fact is wrong → correct it | `update` |
| Fact is incomplete → add detail | `update` |
| Two facts say the same thing → combine | `merge` |
| Complex fact → break apart | `split` |
| Fact no longer true → archive | `deactivate` |

### 7. Tags are your friend

Always add 2–4 tags. They enable category-filtered search and keep your knowledge base organised.
Example for «OpenRouter proxy 199.68.196.14:31149»:
```
tags="openrouter,proxy,socks5,networking"
```

### 8. Export periodically

Every few sessions, or when the user reviews their knowledge, offer to export:
```
kb(action="export", format="markdown", path="~/knowledge_base.md")
```

## Quick Reference

```
kb(action="add", text="...", category="...", tags="...", confidence=0.7)
kb(action="search", query="...", mode="semantic", limit=5, category="...")
kb(action="update", id="...", text="...")
kb(action="merge", id="primary_id", with="secondary_id", text="merged text")
kb(action="split", id="...", parts="A | B | C")
kb(action="history", id="...")
kb(action="review", limit=10)
kb(action="stats")
kb(action="export", format="markdown", path="~/kb.md")
kb(action="relate", id="a1", to="b2")
kb(action="unrelate", id="a1", to="b2")
kb(action="deactivate", id="...")
kb(action="reactivate", id="...")
```

## Pitfalls

- **Don't store secrets in `text` of `credentials` category** — the knowledge base is local but unencrypted.
  Store *pointers*: «API key for X is in ~/.env under X_API_KEY».
- **Don't call `add` without `category`** — auto-classify based on context.
- **`merge` is irreversible for the originals** (they get archived). Review both facts before merging.
- **`vacuum` hard-deletes archived rows.** Default threshold is 365 days — safe for most use cases.
- **The embedding model loads on first use** (lazy). First `search` or `add` will take ~2 seconds.
