"""
Pytest Fixtures for Isolated RAG Chatbot Test Suite
Location: rag-chatbot/tests/conftest.py
"""

import os
import sys
from pathlib import Path
from types import ModuleType
import pytest

# Paths
_TESTS_DIR = Path(__file__).resolve().parent
_RAG_ROOT = _TESTS_DIR.parent
_PROJECT_ROOT = _RAG_ROOT.parent

# Ensure project root is first in sys.path so 'backend' resolves to main app
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
else:
    sys.path.remove(str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT))

# Register packages in sys.modules under 'rag_chatbot' namespace
rag_root_str = str(_RAG_ROOT)
for mod_name, rel_p in [
    ("rag_chatbot", ""),
    ("rag_chatbot.backend", "backend"),
    ("rag_chatbot.backend.retrieval", "backend/retrieval"),
    ("rag_chatbot.ingestion", "ingestion"),
]:
    if mod_name not in sys.modules:
        full_p = os.path.join(rag_root_str, rel_p) if rel_p else rag_root_str
        pkg = ModuleType(mod_name)
        pkg.__path__ = [full_p]
        pkg.__file__ = os.path.join(full_p, "__init__.py")
        sys.modules[mod_name] = pkg

from backend.db.session import SessionLocal
from backend.main import app
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def db_session():
    """Provides a database session for test execution."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)
