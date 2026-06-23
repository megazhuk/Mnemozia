"""
Mnemozia — semantic knowledge base for Hermes Agent.

Named after Mnemosyne (Μνημοσύνη), the Greek goddess of memory.
Built on PostgreSQL + pgvector + intfloat/multilingual-e5-small.
Lazy-loads the embedding model — zero RAM cost until first query.
"""

from .core import MnemoziaKB

__version__ = "1.0.0"
__all__ = ["MnemoziaKB"]
