"""
Knowledge Base Service Shim
Location: backend/services/knowledge_base_service.py
Project: Demand-Decision-Intelligence

Thin backward-compatibility adapter delegating to the isolated RAG engine
in `rag-chatbot/backend/knowledge_base_service.py` via `backend/services/rag_adapter.py`.
"""

from backend.services.rag_adapter import isolated_knowledge_base, KnowledgeBaseService

# Expose singleton instance for backward compatibility
knowledge_base_service = isolated_knowledge_base

__all__ = ["knowledge_base_service", "KnowledgeBaseService"]
