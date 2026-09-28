"""
Tests for Hybrid Query Routing (DOCS vs DATA vs HYBRID)
Location: rag-chatbot/tests/test_hybrid_routing.py
Project: Demand-Decision-Intelligence

Covers:
- Topic A: DOCS query classification and execution
- Topic B: DATA query classification and guarded SQL execution
- Topic C: HYBRID query classification and blended synthesis
- Topic H: Deterministic HYBRID fallback uses real document text, no hardcoded rules
"""

import pytest
from rag_chatbot.backend.decision_rag_synthesizer import decision_rag_synthesizer


def test_docs_query_routing():
    """Topic A: Verifies documentation questions route to DOCS."""
    doc_queries = [
        "What is the demand forecasting methodology?",
        "What does the PRD say about system architecture?",
        "Explain the safety stock formula in SRS.md",
        "What models are used for forecasting?",
    ]
    for q in doc_queries:
        route = decision_rag_synthesizer.route_query(q)
        assert route == "DOCS", f"Expected DOCS route for '{q}', got {route}"


def test_data_query_routing():
    """Topic B: Verifies operational database questions route to DATA."""
    data_queries = [
        "Which products are below ROP?",
        "Show me the top 10 SKUs by sales volume",
        "What is my total capital tied up in dead stock?",
        "Which SKUs will stock out before Diwali?",
    ]
    for q in data_queries:
        route = decision_rag_synthesizer.route_query(q)
        assert route == "DATA", f"Expected DATA route for '{q}', got {route}"


def test_hybrid_query_routing():
    """Topic C: Verifies policy/explanation questions route to HYBRID."""
    hybrid_queries = [
        "Why is SKU 19512 recommended for reorder?",
        "Why are these items below ROP?",
        "Explain why this SKU is marked critical and what policy applies",
    ]
    for q in hybrid_queries:
        route = decision_rag_synthesizer.route_query(q)
        assert route == "HYBRID", f"Expected HYBRID route for '{q}', got {route}"


def test_data_query_execution(db_session):
    """Topic B: Verifies DATA queries execute via guarded SQL adapter and return tabular data."""
    res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="Which products are below ROP?"
    )
    assert res["status"] == "success"
    assert res["query_type"] == "DATA"
    assert res["template_name"] == "items_below_rop"
    assert "table" in res
    assert "prose" in res
    assert "Answer" in res["prose"]
    assert len(res["sources"]) > 0


def test_docs_query_execution(db_session):
    """Topic A: Verifies DOCS queries return document citations and 6-part markdown."""
    res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="What is the demand forecasting methodology?"
    )
    assert res["status"] == "success"
    assert res["query_type"] == "DOCS"
    assert len(res["document_sources"]) > 0
    assert any("ML_ENGINEERING" in s or "ARCHITECTURE" in s or "PRD" in s for s in res["sources"])

    prose = res["prose"]
    assert "### Answer" in prose
    assert "### What this means" in prose
    assert "### Key numbers" in prose
    assert "### Recommended action" in prose
    assert "### Evidence" in prose
    assert "### Source" in prose


def test_hybrid_query_execution_and_no_fake_rules(db_session):
    """
    Topic C & H: Verifies HYBRID queries blend data and docs without inventing
    hardcoded business claims like '95% cycle service level'.
    """
    res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="Why is SKU 19512 recommended for reorder?"
    )
    assert res["status"] == "success"
    assert res["query_type"] == "HYBRID"
    assert len(res["sources"]) > 0

    prose = res["prose"]
    assert "### Answer" in prose
    assert "### What this means" in prose
    # Critical Topic H assertion: Must NOT claim 'cycle service levels (95%)' unless in retrieved excerpt
    if "95%" in prose:
        # If 95% is in prose, it must have come from document sources
        doc_texts = " ".join([d.get("text", "") for d in res.get("document_sources", [])])
        assert "95%" in doc_texts, "95% service level claim appeared in prose without being in document evidence!"
