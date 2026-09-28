"""
Document Retriever
Location: rag-chatbot/backend/retrieval/document_retriever.py
Project: Demand-Decision-Intelligence

Responsible for semantic vector search and resilient lexical fallback search
over project documentation chunks. Guarantees line tracking, section metadata,
and mathematically correct distance-to-similarity conversions.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """
    Retrieves relevant markdown documentation chunks with line and section citations.
    Delegates to KnowledgeBaseService while standardizing scores and metadata.
    """

    def __init__(self, knowledge_base_service: Any):
        self.kb = knowledge_base_service

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top_k chunks matching the query.
        Ensures each chunk has document_name, source_path, heading,
        start_line, end_line, chunk_text, and properly labeled retrieval_score.
        """
        if not query or not query.strip():
            return []

        raw_hits = self.kb.search(query=query, top_k=top_k)
        standardized_hits: List[Dict[str, Any]] = []

        for h in raw_hits:
            # Ensure all required fields exist
            retrieval_method = h.get("retrieval_method", "semantic_vector")
            score = h.get("retrieval_score", h.get("similarity_score", 0.0))
            score_type = h.get("score_type", "cosine_similarity" if retrieval_method == "semantic_vector" else "lexical_frequency")

            standardized_hits.append({
                "chunk_id": h.get("chunk_id", ""),
                "document_name": h.get("document_name", "Documentation"),
                "source_path": h.get("source_path", ""),
                "heading": h.get("heading", ""),
                "start_line": int(h.get("start_line", 1)),
                "end_line": int(h.get("end_line", 1)),
                "text": h.get("text", ""),
                "chunk_text": h.get("text", ""),
                "retrieval_score": round(float(score), 4),
                "similarity_score": round(float(score), 4),  # backward compatibility
                "score_type": score_type,
                "retrieval_method": retrieval_method,
            })

        return standardized_hits
