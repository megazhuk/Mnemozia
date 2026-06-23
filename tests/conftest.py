"""
Pytest configuration — mocks lancedb before test collection.
LanceDB's Rust binary crashes with SIGILL on this Sandy Bridge CPU,
so we provide a thin mock that satisfies schema.py imports.
"""

import sys
from unittest.mock import MagicMock
from types import ModuleType

# ── Build lancedb mock module tree ──

_lancedb = ModuleType("lancedb")
_lancedb.__version__ = "0.4.4"
_lancedb.__file__ = "<mocked>"

# lancedb.embeddings
_embeddings = ModuleType("lancedb.embeddings")


class MockRegistry:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get(self, name):
        mock_ef = MagicMock()
        mock_ef.create.return_value = MagicMock()
        return mock_ef


_embeddings.EmbeddingFunctionRegistry = MockRegistry

# lancedb.pydantic
_pydantic = ModuleType("lancedb.pydantic")


class MockLanceModel:
    pass


class MockVector:
    def __new__(cls, dims):
        return list


_pydantic.LanceModel = MockLanceModel
_pydantic.Vector = MockVector

# lancedb.connect
def mock_connect(uri):
    mock_db = MagicMock()
    mock_db.table_names.return_value = []
    mock_db.create_table.return_value = MagicMock()
    mock_db.open_table.return_value = MagicMock()
    return mock_db


_lancedb.connect = mock_connect
_lancedb.embeddings = _embeddings
_lancedb.pydantic = _pydantic

# Register before test collection
sys.modules["lancedb"] = _lancedb
sys.modules["lancedb.embeddings"] = _embeddings
sys.modules["lancedb.pydantic"] = _pydantic
