"""
Decision RAG Synthesizer Shim
Location: backend/services/decision_rag_synthesizer.py
Project: Demand-Decision-Intelligence

Thin backward-compatibility adapter delegating to the isolated RAG engine
in `rag-chatbot/backend/decision_rag_synthesizer.py` via `backend/services/rag_adapter.py`.
"""

from backend.services.rag_adapter import rag_adapter, isolated_synthesizer, DecisionRAGSynthesizer

# Expose singleton instance for backward compatibility
decision_rag_synthesizer = rag_adapter

__all__ = ["decision_rag_synthesizer", "DecisionRAGSynthesizer"]
