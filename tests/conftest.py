"""
Pytest configuration — mocks lancedb to prevent SIGILL on pre-Haswell CPUs.
psycopg2 mocking is done per-test in test_core.py via fixture.
"""

import os
import sys
from unittest.mock import MagicMock
from types import ModuleType

import pytest

# Always mock lancedb (its Rust binary crashes with SIGILL on Sandy Bridge)
_lancedb = ModuleType("lancedb")
_lancedb.__version__ = "0.0.0"
_emb = ModuleType("lancedb.embeddings")
_emb.EmbeddingFunctionRegistry = MagicMock()
_lancedb.embeddings = _emb
sys.modules["lancedb"] = _lancedb
sys.modules["lancedb.embeddings"] = _emb
