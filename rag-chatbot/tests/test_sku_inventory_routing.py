"""
Tests for SKU-Specific Inventory Query Routing & Verification
Location: rag-chatbot/tests/test_sku_inventory_routing.py
Project: Demand-Decision-Intelligence

Verifies:
1. SKU-specific reorder query ("Why is SKU 19512 recommended for reorder?") -> HYBRID, sku_inventory_recommendation
2. SKU-specific inventory recommendation -> DATA, sku_inventory_recommendation
3. SKU-specific reorder point -> DATA, sku_inventory_recommendation
4. SKU-specific current stock -> DATA, sku_inventory_recommendation
5. SKU-specific recommended order quantity -> DATA, sku_inventory_recommendation
6. SKU-specific missing-product clean behavior (no whole-catalog fallback)
7. HYBRID SKU query preserving product_id into SQL retrieval layer
8. Broad "Which items are below ROP?" still using items_below_rop
9. Broad "Show me the top 10 SKUs by sales volume" still using top_n_by
10. Conceptual "How is safety stock calculated?" still routing to DOCS
11. Zero hardcoded product IDs, quantities, prices, or business values
"""

import datetime
import pytest
from unittest.mock import patch

from backend.models.inventory import InventoryRecommendation
from backend.models.product import Product
from rag_chatbot.backend.decision_rag_synthesizer import decision_rag_synthesizer
from rag_chatbot.backend.retrieval.sql_retriever import SQLRetriever


def test_sku_reorder_query_routing_and_synthesis(db_session):
    """
    Asserts:
    - "Why is SKU 19512 recommended for reorder?" -> HYBRID
    - product_id 19512 reaches the SQL retrieval layer
    - SQL retrieval does NOT become items_below_rop
    - Document retrieval still happens
    - Final template is NOT items_below_rop (is sku_inventory_recommendation)
    - Response contains actual verified database information when DB has the record
    """
    query = "Why is SKU 19512 recommended for reorder?"

    # 1. Test with missing record in DB (clean missing data behavior)
    res_missing = decision_rag_synthesizer.synthesize(db=db_session, query=query)
    assert res_missing["query_type"] == "HYBRID"
    assert res_missing["template_name"] == "sku_inventory_recommendation"
    assert res_missing["template_name"] != "items_below_rop"
    assert len(res_missing["document_sources"]) > 0
    assert "No verified inventory recommendation record" in res_missing["prose"]
    assert "Verified database:" in res_missing["prose"]
    assert "Documentation:" in res_missing["prose"]

    # 2. Test with temporary DB record to verify real values are used without hardcoding
    temp_rec = None
    try:
        temp_rec = InventoryRecommendation(
            dataset_id=1,
            product_id="19512",
            city_name="ALL",
            calculation_date=datetime.date.today(),
            current_stock=120.0,
            avg_daily_demand=25.0,
            lead_time_days=3,
            safety_stock=45.0,
            reorder_point=150.0,
            recommended_order_qty=200.0,
            risk_status="HIGH_STOCKOUT_RISK",
            priority="HIGH",
        )
        db_session.add(temp_rec)
        db_session.commit()

        res_found = decision_rag_synthesizer.synthesize(db=db_session, query=query)
        assert res_found["status"] == "success"
        assert res_found["query_type"] == "HYBRID"
        assert res_found["template_name"] == "sku_inventory_recommendation"
        assert res_found["template_name"] != "items_below_rop"
        assert len(res_found["document_sources"]) > 0

        # Assert verified DB values are present
        assert len(res_found["table"]["rows"]) == 1
        row = res_found["table"]["rows"][0]
        assert row["current_stock"] == 120.0
        assert row["reorder_point"] == 150.0
        assert row["safety_stock"] == 45.0
        assert row["recommended_order_quantity"] == 200.0
        assert row["risk_status"] == "HIGH_STOCKOUT_RISK"

        # Assert prose formatting distinguishes DB data from docs
        prose = res_found["prose"]
        assert "Verified database:" in prose
        assert "120" in prose
        assert "150" in prose
        assert "45" in prose
        assert "200" in prose
        assert "HIGH_STOCKOUT_RISK" in prose
        assert "Documentation:" in prose
    finally:
        if temp_rec:
            db_session.delete(temp_rec)
            db_session.commit()


def test_product_id_reaches_sql_retriever_context(db_session):
    """Verifies that resolved product_id (19512) is explicitly passed in context to SQLRetriever."""
    called_context = {}

    orig_execute = decision_rag_synthesizer.sql_retriever.execute_query

    def spy_execute(db, query, dataset_id=None, user_id=None, session_id=None, context=None):
        nonlocal called_context
        called_context = context or {}
        return orig_execute(db, query, dataset_id=dataset_id, user_id=user_id, session_id=session_id, context=context)

    with patch.object(decision_rag_synthesizer.sql_retriever, "execute_query", side_effect=spy_execute):
        decision_rag_synthesizer.synthesize(db=db_session, query="Why is SKU 19512 recommended for reorder?")

    assert called_context.get("product_id") == 19512, f"Expected product_id=19512 in context, got {called_context}"


def test_sku_inventory_recommendation_query(db_session):
    """'Show me the inventory recommendation for SKU 19512' -> DATA, sku_inventory_recommendation."""
    res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="Show me the inventory recommendation for SKU 19512"
    )
    assert res["status"] == "success"
    assert res["query_type"] == "DATA"
    assert res["template_name"] == "sku_inventory_recommendation"
    assert res["template_name"] != "general_business_advisory"
    assert res["template_name"] != "items_below_rop"


def test_sku_reorder_point_query(db_session):
    """'What is the reorder point of SKU 19512?' -> DATA, sku_inventory_recommendation."""
    res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="What is the reorder point of SKU 19512?"
    )
    assert res["status"] == "success"
    assert res["query_type"] == "DATA"
    assert res["template_name"] == "sku_inventory_recommendation"
    assert res["template_name"] != "items_below_rop"


def test_sku_current_stock_query(db_session):
    """'What is the current stock of SKU 19512?' -> DATA, sku_inventory_recommendation."""
    res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="What is the current stock of SKU 19512?"
    )
    assert res["status"] == "success"
    assert res["query_type"] == "DATA"
    assert res["template_name"] == "sku_inventory_recommendation"
    assert res["template_name"] != "items_below_rop"


def test_sku_recommended_order_quantity_query(db_session):
    """'What is the recommended order quantity for SKU 19512?' -> DATA, sku_inventory_recommendation."""
    res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="What is the recommended order quantity for SKU 19512?"
    )
    assert res["status"] == "success"
    assert res["query_type"] == "DATA"
    assert res["template_name"] == "sku_inventory_recommendation"
    assert res["template_name"] != "items_below_rop"


def test_sku_missing_product_behavior(db_session):
    """Non-existent SKU returns clean verified-missing response, never falls back to whole catalog."""
    res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="What is the reorder point of SKU 999999?"
    )
    assert res["status"] == "success"
    assert res["query_type"] == "DATA"
    assert res["template_name"] == "sku_inventory_recommendation"
    assert res["template_name"] != "items_below_rop"
    assert len(res["table"]["rows"]) == 0
    assert "No verified inventory recommendation record" in res["prose"]


def test_broad_items_below_rop_still_works(db_session):
    """Broad query 'Which items are below ROP?' still routes to items_below_rop without SKU filter."""
    res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="Which items are below ROP?"
    )
    assert res["status"] == "success"
    assert res["query_type"] == "DATA"
    assert res["template_name"] == "items_below_rop"


def test_top_n_by_still_works(db_session):
    """'Show me the top 10 SKUs by sales volume' still routes to top_n_by with rows."""
    res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="Show me the top 10 SKUs by sales volume"
    )
    assert res["status"] == "success"
    assert res["query_type"] == "DATA"
    assert res["template_name"] == "top_n_by"
    assert len(res["table"]["rows"]) > 0


def test_docs_query_still_works(db_session):
    """'How is safety stock calculated?' still routes to DOCS and retrieves docs."""
    res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="How is safety stock calculated?"
    )
    assert res["status"] == "success"
    assert res["query_type"] == "DOCS"
    assert res["template_name"] == "document_rag"
    assert len(res["document_sources"]) > 0


def test_no_hardcoded_sku_values(db_session):
    """Verifies that dynamic SKU IDs other than 19512 behave dynamically without hardcoding."""
    dynamic_sku = "88421"
    res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query=f"What is the reorder point of SKU {dynamic_sku}?"
    )
    assert res["template_name"] == "sku_inventory_recommendation"
    assert dynamic_sku in res["prose"]
    assert "19512" not in res["prose"]


def test_api_endpoint_all_eight_queries(client):
    """Verifies POST /api/v1/assistant/query returns expected routing for all 8 queries."""
    expected = [
        ("Why is SKU 19512 recommended for reorder?", "HYBRID", "sku_inventory_recommendation"),
        ("Show me the inventory recommendation for SKU 19512", "DATA", "sku_inventory_recommendation"),
        ("What is the reorder point of SKU 19512?", "DATA", "sku_inventory_recommendation"),
        ("What is the current stock of SKU 19512?", "DATA", "sku_inventory_recommendation"),
        ("What is the recommended order quantity for SKU 19512?", "DATA", "sku_inventory_recommendation"),
        ("Which items are below ROP?", "DATA", "items_below_rop"),
        ("Show me the top 10 SKUs by sales volume", "DATA", "top_n_by"),
        ("How is safety stock calculated?", "DOCS", "document_rag"),
    ]

    for q, exp_type, exp_template in expected:
        resp = client.post("/api/v1/assistant/query", json={"query": q})
        assert resp.status_code == 200, f"Failed for '{q}': {resp.text}"
        data = resp.json()
        assert data["status"] == "success"
        assert data["query_type"] == exp_type, f"Query '{q}' expected {exp_type}, got {data['query_type']}"
        assert data["template_name"] == exp_template, f"Query '{q}' expected {exp_template}, got {data['template_name']}"


def test_items_below_rop_zero_rows_non_contradictory(db_session):
    """
    FIX 1: When 0 items are below ROP, the response must state 0 items are below ROP
    and no immediate procurement action is identified. It must NOT recommend urgent procurement.
    """
    res = decision_rag_synthesizer.synthesize(db=db_session, query="Which items are below ROP?")
    assert res["status"] == "success"
    assert res["query_type"] == "DATA"
    assert res["template_name"] == "items_below_rop"
    assert len(res["table"]["rows"]) == 0

    prose = res["prose"]
    assert "0 SKUs are currently below ROP in this dataset." in prose
    assert "No immediate ROP-based procurement action is identified from this query." in prose
    assert "urgent procurement" not in prose.lower()
    assert "trigger procurement" not in prose.lower()


def test_items_below_rop_one_or_more_rows_recommends_replenishment(db_session):
    """
    FIX 1: When 1+ items are below ROP, the response recommends reviewing and replenishing them.
    """
    temp_rec = None
    try:
        temp_rec = InventoryRecommendation(
            dataset_id=1,
            product_id="19512",
            city_name="ALL",
            calculation_date=datetime.date.today(),
            current_stock=50.0,
            avg_daily_demand=25.0,
            lead_time_days=3,
            safety_stock=45.0,
            reorder_point=150.0,
            recommended_order_qty=200.0,
            risk_status="HIGH_STOCKOUT_RISK",
            priority="HIGH",
        )
        db_session.add(temp_rec)
        db_session.commit()

        res = decision_rag_synthesizer.synthesize(db=db_session, query="Which items are below ROP?")
        assert res["status"] == "success"
        assert res["query_type"] == "DATA"
        assert res["template_name"] == "items_below_rop"
        assert len(res["table"]["rows"]) == 1

        prose = res["prose"]
        assert "Found 1 SKUs currently operating below their Reorder Point (ROP)." in prose
        assert "Immediate replenishment review is recommended for these items." in prose
        assert "Review the 1 below-ROP items in the table below and initiate replenishment" in prose
    finally:
        if temp_rec:
            db_session.delete(temp_rec)
            db_session.commit()


def test_safety_stock_formula_grounding_and_inputs(db_session):
    """
    FIX 2: 'How is safety stock calculated?' retrieves the documented mathematical formula
    from system_architecture_hld_lld.md / ML_ENGINEERING_AND_PIPELINE_GUIDE.md and explains its inputs.
    """
    res = decision_rag_synthesizer.synthesize(db=db_session, query="How is safety stock calculated?")
    assert res["status"] == "success"
    assert res["query_type"] == "DOCS"
    assert res["template_name"] == "document_rag"

    prose = res["prose"]
    assert "King's Formula Safety Stock" in prose or "\\text{SS}" in prose or "SS =" in prose
    assert "Z" in prose
    assert "sigma_d" in prose
    assert "system_architecture_hld_lld.md" in "".join(res["sources"]) or "ML_ENGINEERING_AND_PIPELINE_GUIDE.md" in "".join(res["sources"])
    assert "The current project documentation does not specify a complete safety-stock" not in prose


def test_formula_for_safety_stock_query(db_session):
    """
    FIX 2: 'What is the formula for safety stock?' returns grounded formula with documented variables.
    """
    res = decision_rag_synthesizer.synthesize(db=db_session, query="What is the formula for safety stock?")
    assert res["status"] == "success"
    assert res["query_type"] == "DOCS"
    assert res["template_name"] == "document_rag"

    prose = res["prose"]
    assert "King's Formula" in prose or "\\text{SS}" in prose or "SS =" in prose
    assert "system_architecture_hld_lld.md" in "".join(res["sources"]) or "ML_ENGINEERING_AND_PIPELINE_GUIDE.md" in "".join(res["sources"])


def test_reorder_point_calculation_grounding_and_inputs(db_session):
    """
    FIX 2: 'How is reorder point calculated?' returns documented ROP = LTD + SS methodology and inputs.
    """
    res = decision_rag_synthesizer.synthesize(db=db_session, query="How is reorder point calculated?")
    assert res["status"] == "success"
    assert res["query_type"] == "DOCS"
    assert res["template_name"] == "document_rag"

    prose = res["prose"]
    assert "ROP" in prose
    assert "LTD" in prose
    assert "Lead Time Demand" in prose
    assert "system_architecture_hld_lld.md" in "".join(res["sources"]) or "ML_ENGINEERING_AND_PIPELINE_GUIDE.md" in "".join(res["sources"])


def test_unspecified_calculation_formula_clean_rejection(db_session):
    """
    FIX 2: When user asks for a formula not in project documentation, returns explicit missing message.
    """
    res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="What is the formula for warehouse cooling shelf-life degradation?"
    )
    assert res["status"] == "success"
    assert res["query_type"] == "DOCS"
    assert "The current project documentation does not specify a complete calculation formula" in res["prose"]

