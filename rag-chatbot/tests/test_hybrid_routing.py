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
from rag_chatbot.backend.decision_rag_synthesizer import (
    decision_rag_synthesizer,
    build_document_search_query,
)


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


def test_fix_1_docs_query_classification_positive_and_negative():
    """
    FIX 1: Verifies pattern/token-based recognition for conceptual/documentation questions,
    ensuring all required DOCS examples route to DOCS without breaking DATA routing.
    """
    must_route_to_docs = [
        "How is safety stock and reorder point calculated?",
        "What are the business rules for dead stock clearance?",
        "Explain the recommendation logic.",
        "How does demand forecasting work?",
        "What formula is used for safety stock?",
        # Pattern variations
        "How are safety stocks calculated?",
        "How does lead time forecasting work?",
        "Explain the formula for ROP",
        "What is the formula for reorder point?",
        "What are the rules for liquidation?",
        "Why does the system use Croston method?",
        "How is reorder point determined?",
    ]
    for q in must_route_to_docs:
        route = decision_rag_synthesizer.route_query(q)
        assert route == "DOCS", f"Expected DOCS route for '{q}', got {route}"

    must_remain_data = [
        "Show me the top 10 SKUs by sales volume",
        "Which items are below ROP?",
        "Which SKUs will stock out before Diwali?",
        "Show me supplier purchase orders",
    ]
    for q in must_remain_data:
        route = decision_rag_synthesizer.route_query(q)
        assert route == "DATA", f"Expected DATA route for '{q}', got {route}"


def test_fix_2_hybrid_document_search_query_expansion():
    """
    FIX 2: Verifies build_document_search_query() normalizes HYBRID queries:
    1. Removes numeric SKU/product IDs so they don't dominate vector search
    2. Keeps domain terms (reorder, inventory, policy, safety stock, lead time)
    """
    # Reorder question with SKU 19512
    q = "Why is SKU 19512 recommended for reorder?"
    doc_q = build_document_search_query(q, query_type="HYBRID")
    assert "19512" not in doc_q, f"Numeric SKU was not removed: {doc_q}"
    assert "reorder" in doc_q.lower()
    assert "inventory" in doc_q.lower()
    assert "policy" in doc_q.lower()
    assert "safety stock" in doc_q.lower()
    assert "lead time" in doc_q.lower()

    # Stockout question with SKU
    q_stock = "Why is product 98765 facing stockout risk?"
    doc_stock = build_document_search_query(q_stock, query_type="HYBRID")
    assert "98765" not in doc_stock
    assert "stockout" in doc_stock.lower()
    assert "safety stock" in doc_stock.lower()

    # Demand forecasting question
    q_fc = "Explain forecast accuracy changes for SKU 44102"
    doc_fc = build_document_search_query(q_fc, query_type="HYBRID")
    assert "44102" not in doc_fc
    assert "forecasting" in doc_fc.lower() or "forecast" in doc_fc.lower()
    assert "methodology" in doc_fc.lower() or "accuracy" in doc_fc.lower()

    # Non-calculation DOCS query should be returned unchanged
    docs_q = "What are the business rules for dead stock clearance?"
    assert build_document_search_query(docs_q, query_type="DOCS") == docs_q

    # Calculation/formula DOCS query should be enriched with formula terms
    calc_q = "How is safety stock calculated?"
    calc_enriched = build_document_search_query(calc_q, query_type="DOCS")
    assert any(t in calc_enriched for t in ["calculation", "formula", "methodology", "equation", "inputs"])


def test_fix_2_hybrid_retrieval_sku_retention_in_sql(db_session, monkeypatch):
    """
    FIX 2: Verifies that during hybrid synthesis:
    - Document retriever receives the normalized query without the SKU
    - SQL retriever still receives the query containing the numeric SKU
    """
    captured_doc_queries = []
    captured_sql_queries = []

    orig_doc_retrieve = decision_rag_synthesizer.document_retriever.retrieve
    orig_sql_execute = decision_rag_synthesizer.sql_retriever.execute_query

    def mock_doc_retrieve(search_query, top_k=3):
        captured_doc_queries.append(search_query)
        return orig_doc_retrieve(search_query, top_k=top_k)

    def mock_sql_execute(db, query, **kwargs):
        captured_sql_queries.append(query)
        return orig_sql_execute(db, query, **kwargs)

    monkeypatch.setattr(decision_rag_synthesizer.document_retriever, "retrieve", mock_doc_retrieve)
    monkeypatch.setattr(decision_rag_synthesizer.sql_retriever, "execute_query", mock_sql_execute)

    query = "Why is SKU 19512 recommended for reorder?"
    res = decision_rag_synthesizer.synthesize(db=db_session, query=query)
    assert res["status"] == "success"

    # Verify document search query had SKU removed and domain concepts added
    assert len(captured_doc_queries) > 0
    doc_q = captured_doc_queries[0]
    assert "19512" not in doc_q
    assert "reorder" in doc_q.lower()
    assert "inventory" in doc_q.lower()
    assert "safety stock" in doc_q.lower()

    # Verify SQL query still received SKU 19512
    assert len(captured_sql_queries) > 0
    assert "19512" in captured_sql_queries[0]
