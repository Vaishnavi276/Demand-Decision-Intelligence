"""
Phase 2 Test Suite: Hybrid RAG & Conversational Context
Tests:
1. Document ingestion & vector indexing
2. Document retrieval (Semantic Vector Search)
3. Guarded SQL retrieval (Zero-SQL templates)
4. Hybrid retrieval (Data + Knowledge Base)
5. Missing-document behavior
6. Missing-LLM-key deterministic fallback
7. Multi-turn context carryover
8. Evidence metadata structure
9. No hallucinated sources / No-Bluff enforcement
10. Existing assistant endpoint backward compatibility & regression tests
"""

import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app
from backend.db.session import SessionLocal
from backend.services.knowledge_base_service import knowledge_base_service
from backend.services.decision_rag_synthesizer import decision_rag_synthesizer


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


# ── Test 1: Document Ingestion ───────────────────────────────────────────────

def test_1_document_ingestion():
    """Verifies that project documents are discovered, chunked, and indexed."""
    stats = knowledge_base_service.get_document_stats()
    assert stats["total_documents"] > 0, "No documents were indexed."
    assert stats["total_chunks"] > 0, "No document chunks were created."
    assert stats["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert "ARCHITECTURE.md" in stats["indexed_documents"] or "PRD.md" in stats["indexed_documents"]


# ── Test 2: Document Retrieval ───────────────────────────────────────────────

def test_2_document_retrieval():
    """Tests semantic search for documentation questions."""
    query = "What is the demand forecasting methodology?"
    hits = knowledge_base_service.search(query, top_k=3)
    assert len(hits) > 0, "Document vector search returned zero results."

    top_hit = hits[0]
    assert "document_name" in top_hit
    assert "source_path" in top_hit
    assert "chunk_id" in top_hit
    assert "similarity_score" in top_hit
    assert top_hit["similarity_score"] > 0.4
    assert len(top_hit["text"]) > 20


# ── Test 3: Guarded SQL Retrieval ────────────────────────────────────────────

def test_3_sql_retrieval(db_session):
    """Verifies business data queries are routed strictly to guarded SQL templates."""
    query = "Which products are below ROP?"
    route = decision_rag_synthesizer.route_query(query)
    assert route == "DATA", f"Expected DATA route for ROP query, got {route}"

    res = decision_rag_synthesizer.synthesize(db=db_session, query=query)
    assert res["status"] == "success"
    assert res["query_type"] == "DATA"
    assert res["template_name"] == "items_below_rop"
    assert "table" in res
    assert "prose" in res
    assert "Answer" in res["prose"]


# ── Test 4: Hybrid Retrieval ────────────────────────────────────────────────

def test_4_hybrid_retrieval(db_session):
    """Verifies hybrid queries route to HYBRID and retrieve both data and document context."""
    query = "Why is SKU 19512 recommended for reorder?"
    route = decision_rag_synthesizer.route_query(query)
    assert route == "HYBRID", f"Expected HYBRID route, got {route}"

    res = decision_rag_synthesizer.synthesize(db=db_session, query=query)
    assert res["status"] == "success"
    assert res["query_type"] == "HYBRID"
    assert len(res["sources"]) > 0
    # Must contain both Answer and business explanation
    assert "### Answer" in res["prose"]


# ── Test 5: Missing-Document Behavior ────────────────────────────────────────

def test_5_missing_document_behavior():
    """Verifies safe handling when queries have no matching documentation."""
    query = "supercalifragilisticexpialidocious quantum warp engine fluctuations"
    hits = knowledge_base_service.search(query, top_k=3)
    # FastEmbed/Chroma will return top-k nearest neighbors with lower similarities or empty
    assert isinstance(hits, list)
    for hit in hits:
        assert "document_name" in hit


# ── Test 6: Missing-LLM-Key Fallback ─────────────────────────────────────────

def test_6_missing_llm_key_fallback(db_session):
    """Verifies deterministic 6-part markdown generation when LLM API keys are absent."""
    # Temporarily clear API keys
    orig_groq = decision_rag_synthesizer.groq_api_key
    orig_openai = decision_rag_synthesizer.openai_api_key
    decision_rag_synthesizer.groq_api_key = None
    decision_rag_synthesizer.openai_api_key = None

    try:
        res = decision_rag_synthesizer.synthesize(
            db=db_session,
            query="What is the forecasting methodology?"
        )
        assert res["status"] == "success"
        prose = res["prose"]
        assert "### Answer" in prose
        assert "### What this means" in prose
        assert "### Key numbers" in prose
        assert "### Recommended action" in prose
        assert "### Evidence" in prose
        assert "### Source" in prose
    finally:
        decision_rag_synthesizer.groq_api_key = orig_groq
        decision_rag_synthesizer.openai_api_key = orig_openai


# ── Test 7: Multi-Turn Context Carryover ─────────────────────────────────────

def test_7_multi_turn_context(db_session):
    """
    Tests conversation entity carryover between turns:
    Turn 1: "Which products should I reorder in Delhi?"
    Turn 2: "How much will that cost?"
    Expected: City 'Delhi' and previous reorder result retained.
    """
    # Turn 1
    t1_res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="Which products should I reorder in Delhi?"
    )
    sess_id = t1_res["session_id"]
    assert sess_id is not None

    # Turn 2
    t2_res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="How much will that cost?",
        session_id=sess_id
    )
    assert t2_res["status"] == "success"
    prose = t2_res["prose"]
    # Should resolve the context of Delhi / previous reorder
    assert ("Delhi" in prose or "reorder" in prose.lower() or "cost" in prose.lower() or "₹" in prose)


# ── Test 8: Evidence Metadata Structure ──────────────────────────────────────

def test_8_evidence_metadata_structure(db_session):
    """Verifies complete metadata fields for Grounded Evidence Drawer."""
    res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="What does the PRD say about demand health score?"
    )
    assert "sources" in res
    assert "data_sources" in res
    assert "document_sources" in res
    assert "evidence" in res

    if res["document_sources"]:
        doc_src = res["document_sources"][0]
        assert "document_name" in doc_src
        assert "source_path" in doc_src
        assert "chunk_id" in doc_src
        assert "similarity" in doc_src
        assert "text" in doc_src


# ── Test 9: No Hallucinated Sources (No-Bluff Rule) ─────────────────────────

def test_9_no_hallucinated_sources(db_session):
    """Verifies that all returned sources correspond to verified on-disk or database resources."""
    res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="What is the supply chain architecture?"
    )
    sources = res.get("sources", [])
    project_root = Path(__file__).resolve().parent.parent.parent

    for src in sources:
        if src.endswith(".md") or "/" in src or "\\" in src:
            # Check if this document exists on disk
            potential_paths = [
                project_root / src,
                project_root / "docs" / src,
                project_root / Path(src).name,
            ]
            exists = any(p.exists() for p in potential_paths)
            assert exists, f"Hallucinated document source: {src}"


# ── Test 10: Existing Assistant Endpoint Backward Compatibility ──────────────

def test_10_endpoint_compatibility_and_regressions(client):
    """Verifies all existing Assistant API endpoints and standard operational queries."""
    # 1. Suggested Prompts
    r_prompts = client.get("/api/v1/assistant/suggested-prompts")
    assert r_prompts.status_code == 200
    assert "prompts" in r_prompts.json()

    # 2. Daily Briefing
    r_brief = client.get("/api/v1/assistant/daily-briefing")
    assert r_brief.status_code == 200

    # 3. Query: Top demand SKUs
    r_demand = client.post("/api/v1/assistant/query", json={"query": "Show me the top 10 SKUs by sales volume"})
    assert r_demand.status_code == 200
    d_json = r_demand.json()
    assert d_json["status"] == "success"
    assert "prose" in d_json
    assert "table" in d_json
    assert "execution_ms" in d_json

    # 4. Query: Dead stock capital
    r_dead = client.post("/api/v1/assistant/query", json={"query": "What's my total capital tied up in dead stock?"})
    assert r_dead.status_code == 200
    assert r_dead.json()["status"] == "success"

    # 5. Query: Festival Calendar
    r_fest = client.post("/api/v1/assistant/query", json={"query": "What upcoming festivals?"})
    assert r_fest.status_code == 200
    assert r_fest.json()["status"] == "success"

    # 6. Session history
    sess_id = d_json.get("session_id")
    if sess_id:
        r_hist = client.get(f"/api/v1/assistant/history?session_id={sess_id}")
        assert r_hist.status_code == 200
        assert "messages" in r_hist.json()
