"""
Mnemozia — semantic knowledge base for Hermes Agent.

Named after Mnemosyne (Μνημοσύνη), the Greek goddess of memory.
Built on pg0 (PostgreSQL 18 + pgvector) + llama.cpp + Qwen3-Embedding-0.6B.
Embeddings are computed by llama-server via HTTP API — no Python ML deps.
"""

from .core import MnemoziaKB

__version__ = "2.0.0"
__all__ = ["MnemoziaKB"]
