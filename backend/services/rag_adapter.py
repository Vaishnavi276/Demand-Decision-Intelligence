"""
RAG Adapter — Bridge between Existing Application and Isolated RAG Module
Location: backend/services/rag_adapter.py
Project: Demand-Decision-Intelligence

Provides a clean, decoupled adapter to invoke the isolated RAG engine
located in the dedicated `rag-chatbot/` directory without polluting the
main application backend.
"""

import sys
import os
import logging
from types import ModuleType
from pathlib import Path
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Resolve path to rag-chatbot root
_CURRENT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _CURRENT_DIR.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
_RAG_ROOT = _PROJECT_ROOT / "rag-chatbot"

# Register rag_chatbot in sys.modules to handle hyphenated directory name
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

from rag_chatbot.backend.decision_rag_synthesizer import (
    DecisionRAGSynthesizer,
    decision_rag_synthesizer as isolated_synthesizer,
)
from rag_chatbot.backend.knowledge_base_service import (
    KnowledgeBaseService,
    knowledge_base_service as isolated_knowledge_base,
)


class RAGServiceAdapter:
    """
    Thin integration adapter exposing the isolated RAG synthesizer
    to the FastAPI application router.
    """

    def __init__(self, synthesizer: Any = isolated_synthesizer, knowledge_base: Any = isolated_knowledge_base):
        self.synthesizer = synthesizer
        self.knowledge_base = knowledge_base

    def synthesize(
        self,
        db: Session,
        query: str,
        dataset_id: Optional[int] = None,
        user_id: Optional[int] = None,
        session_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Delegates query synthesis to the isolated RAG engine."""
        return self.synthesizer.synthesize(
            db=db,
            query=query,
            dataset_id=dataset_id,
            user_id=user_id,
            session_id=session_id,
        )

    def route_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Exposes query routing classification."""
        return self.synthesizer.route_query(query=query, context=context)

    def classify_query_type(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Exposes query type classification."""
        return self.synthesizer.classify_query_type(query=query, context=context)

    def get_document_stats(self) -> Dict[str, Any]:
        """Returns statistics from the knowledge base."""
        return self.knowledge_base.get_document_stats()

    def search(self, query: str, top_k: int = 5) -> list:
        """Searches indexed documentation."""
        return self.knowledge_base.search(query=query, top_k=top_k)

    def search_documents(self, query: str, top_k: int = 5) -> list:
        """Searches indexed documentation."""
        return self.knowledge_base.search(query=query, top_k=top_k)

    @property
    def groq_api_key(self):
        return self.synthesizer.groq_api_key

    @groq_api_key.setter
    def groq_api_key(self, val):
        self.synthesizer.groq_api_key = val

    @property
    def openai_api_key(self):
        return self.synthesizer.openai_api_key

    @openai_api_key.setter
    def openai_api_key(self, val):
        self.synthesizer.openai_api_key = val


# Singleton adapter instance
rag_adapter = RAGServiceAdapter()
rag_synthesizer = rag_adapter
