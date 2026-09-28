"""
Tests for Document Retrieval, Ingestion, and Similarity Scoring
Location: rag-chatbot/tests/test_document_retrieval.py
Project: Demand-Decision-Intelligence

Covers:
- Topic A: DOCS queries
- Topic I: Chroma similarity score correctness and metric-awareness
- Document chunk metadata: doc name, source path, heading, start line, end line, chunk text, retrieval score
- Lexical fallback labeling as retrieval_score (never falsely labeled as semantic similarity)
- Missing document reporting
"""

import pytest
from rag_chatbot.backend.knowledge_base_service import (
    knowledge_base_service,
    TARGET_DOCUMENTS,
)
from rag_chatbot.backend.retrieval.document_retriever import DocumentRetriever


@pytest.fixture(scope="module")
def retriever():
    return DocumentRetriever(knowledge_base_service)


def test_target_documents_and_ingestion_stats():
    """Verifies target documents are explicitly listed and reported."""
    assert len(TARGET_DOCUMENTS) >= 10, "Target documents list should contain all core project documents."
    assert "docs/ARCHITECTURE.md" in TARGET_DOCUMENTS
    assert "docs/PRD.md" in TARGET_DOCUMENTS
    assert "docs/SRS.md" in TARGET_DOCUMENTS

    stats = knowledge_base_service.get_document_stats()
    assert stats["total_chunks"] > 0, "Vector store should have indexed chunks."
    assert stats["total_documents"] > 0, "Indexed documents count should be > 0."
    assert "indexed_files" in stats
    assert "missing_files" in stats
    assert isinstance(stats["missing_files"], list)


def test_chunk_metadata_structure(retriever):
    """Verifies that all chunks contain required metadata fields."""
    hits = retriever.retrieve("What is the demand forecasting methodology?", top_k=3)
    assert len(hits) > 0, "Should retrieve at least one chunk."

    for hit in hits:
        assert "document_name" in hit
        assert "source_path" in hit
        assert "heading" in hit
        assert "start_line" in hit
        assert "end_line" in hit
        assert "text" in hit
        assert "chunk_text" in hit
        assert "retrieval_score" in hit
        assert isinstance(hit["start_line"], int)
        assert isinstance(hit["end_line"], int)
        assert hit["end_line"] >= hit["start_line"]
        assert len(hit["text"]) > 10


def test_chroma_similarity_score_correctness(retriever):
    """
    Topic I: Verifies that ChromaDB similarity scores are mathematically valid.
    For cosine distance, similarity must be 1 - distance, bounded in [0.0, 1.0].
    """
    hits = retriever.retrieve("Croston intermittent demand method", top_k=3)
    assert len(hits) > 0

    for hit in hits:
        score = hit["retrieval_score"]
        # Scores must be bounded between 0.0 and 1.0
        assert 0.0 <= score <= 1.0, f"Score {score} is not bounded in [0.0, 1.0]"
        assert hit["score_type"] in ["cosine_similarity", "l2_similarity", "lexical_frequency"]


def test_lexical_fallback_score_labeling():
    """
    Topic I: If lexical fallback is used, score must NOT be labeled as semantic similarity.
    It must use retrieval_score and indicate score_type='lexical_frequency'.
    """
    # Execute lexical search directly
    lexical_hits = knowledge_base_service._lexical_search("forecasting methodology croston", top_k=3)
    assert len(lexical_hits) > 0

    for hit in lexical_hits:
        assert "retrieval_score" in hit
        assert hit["retrieval_method"] == "lexical"
        assert hit["score_type"] == "lexical_frequency"
        # Score must be bounded in [0, 1]
        assert 0.0 <= hit["retrieval_score"] <= 1.0


def test_missing_document_and_gibberish_handling(retriever):
    """Verifies graceful handling of non-existent query terms without exceptions."""
    gibberish = "xyzzy99999 non_existent_token_quantum_flux_antigravity"
    hits = retriever.retrieve(gibberish, top_k=3)
    assert isinstance(hits, list)
    # Either empty or low similarity, but must never crash
    for h in hits:
        assert "document_name" in h
        assert "retrieval_score" in h
