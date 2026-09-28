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


# ==============================================================================
# DATA QA ROUND FOCUSED TESTS (Issues 1 - 4)
# ==============================================================================

def test_top_n_demand_sales_response_neutral_wording(db_session):
    """
    QA Issue 1: Top-N sales/demand response must keep factual ranking
    and use neutral wording without unsupported procurement claims ('hamesha', 'always').
    """
    queries = [
        "Show me the top 10 products by demand.",
        "Show me the top 10 SKUs by sales volume."
    ]
    for q in queries:
        res = decision_rag_synthesizer.synthesize(db=db_session, query=q)
        assert res["status"] == "success"
        assert res["query_type"] == "DATA"
        assert res["template_name"] == "top_n_by"
        assert len(res["table"]["rows"]) == 10

        prose = res["prose"]
        assert "These are the top 10 products by recorded sales/demand volume." in prose
        assert "Review their current stock, ROP, and inventory recommendations before making procurement decisions." in prose
        assert "hamesha" not in prose.lower()
        assert "always" not in prose.lower()


def test_zero_below_rop_exact_wording(db_session):
    """
    QA Issue 2: Zero below-ROP query must state 0 SKUs below ROP and 'No ROP breach detected',
    without falsely claiming inventory is 'optimal'.
    """
    res = decision_rag_synthesizer.synthesize(db=db_session, query="Which items are currently below ROP?")
    assert res["status"] == "success"
    assert res["query_type"] == "DATA"
    assert res["template_name"] == "items_below_rop"
    assert len(res["table"]["rows"]) == 0

    prose = res["prose"]
    assert "0 SKUs are currently below ROP in this dataset. No immediate ROP-based procurement action is identified from this query." in prose
    assert "No evaluated SKU was found below its recorded reorder point." in prose
    assert "Items Below ROP" in prose and "0 SKUs" in prose
    assert "Replenishment Status" in prose and "No ROP breach detected" in prose
    assert "No ROP-based reorder action is indicated by this query. Continue routine inventory monitoring." in prose
    assert "optimal" not in prose.lower()
    assert "urgent" not in prose.lower()


def test_natural_language_highest_demand_routing_variations(db_session):
    """
    QA Issue 3: Natural language demand-ranking variations without explicit numeric SKUs
    must all route through the existing top_n_by template without errors.
    """
    variations = [
        "Which products have the highest demand?",
        "Which products have the highest sales?",
        "Show me the top products by demand.",
        "Show me the top 10 products by demand.",
        "What are the highest-demand products?",
        "Which SKUs have the highest demand?",
    ]
    for q in variations:
        res = decision_rag_synthesizer.synthesize(db=db_session, query=q)
        assert res["status"] == "success", f"Query '{q}' failed: {res}"
        assert res["query_type"] == "DATA", f"Query '{q}' had wrong query_type: {res.get('query_type')}"
        assert res["template_name"] == "top_n_by", f"Query '{q}' had wrong template: {res.get('template_name')}"
        assert len(res["table"]["rows"]) > 0, f"Query '{q}' returned 0 rows"
        assert "These are the top" in res["prose"]
        assert "error" not in res["prose"].lower()


def test_zero_purchase_orders_response_wording(db_session):
    """
    QA Issue 4: Zero purchase order result must state no recent POs were found,
    Total PO Value: ₹0, and must NOT tell user to review an empty table.
    """
    res = decision_rag_synthesizer.synthesize(db=db_session, query="Show me recent purchase orders.")
    assert res["status"] == "success"
    assert res["query_type"] == "DATA"
    assert res["template_name"] == "supplier_po_summary"
    assert len(res["table"]["rows"]) == 0

    prose = res["prose"]
    assert "No recent purchase orders were found in the database." in prose
    assert "No recent PO records are available for the requested query." in prose
    assert "Records Found" in prose and ": 0" in prose
    assert "Total PO Value" in prose and "₹0" in prose
    assert "No PO records are available for review. Check procurement requirements separately if needed." in prose
    assert "table below" not in prose.lower()


# ==============================================================================
# STEP 3 QA REGRESSION TESTS (A through G)
# ==============================================================================

def test_step3_issue1_general_query_after_sku_context(db_session):
    """
    Test A: General query after SKU context must NOT inherit SKU 19512.
    Previous: "Why is SKU 19512 recommended for reorder?"
    Current:  "Which items are currently below ROP?"
    Expected: items_below_rop (NOT sku_inventory_recommendation)
    """
    t1 = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="Why is SKU 19512 recommended for reorder?"
    )
    assert t1["status"] == "success"
    assert t1["query_type"] == "HYBRID"
    assert t1["template_name"] == "sku_inventory_recommendation"
    sess_id = t1["session_id"]

    t2 = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="Which items are currently below ROP?",
        session_id=sess_id
    )
    assert t2["status"] == "success"
    assert t2["query_type"] == "DATA"
    assert t2["template_name"] == "items_below_rop"
    assert t2["template_name"] != "sku_inventory_recommendation"
    assert "19512" not in t2["prose"]


def test_step3_issue2_dead_stock_docs_query(db_session):
    """
    Test B: Dead-stock DOCS query.
    Query: "What are the rules for identifying dead stock?"
    Expected:
    - Route = DOCS
    - Template = document_rag
    - Relevant dead-stock documentation retrieved (system_architecture_hld_lld.md Line 225)
    - Cites documented thresholds (90 days inactivity, 180 days excess cover)
    - Does NOT invent generic inventory rules
    """
    res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="What are the rules for identifying dead stock?"
    )
    assert res["status"] == "success"
    assert res["query_type"] == "DOCS"
    assert res["template_name"] == "document_rag"
    assert res["template_name"] != "sku_inventory_recommendation"

    doc_names = [d.get("document_name") for d in res.get("document_sources", [])]
    assert any("system_architecture_hld_lld.md" in d for d in doc_names)

    prose = res["prose"]
    assert "90" in prose  # 90-day inactivity threshold
    assert "180" in prose  # 180-day excess cover threshold
    assert "system_architecture_hld_lld.md" in prose
    assert "19512" not in prose


def test_step3_issue3_forecasting_methodology_docs_query(db_session):
    """
    Test C: Forecasting methodology.
    Query: "Explain the demand forecasting methodology used in this project."
    Expected:
    - Route = DOCS
    - Template = document_rag
    - Relevant methodology/pipeline evidence across documents
    - Anti-leakage lags and chronological backtesting documented
    """
    res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="Explain the demand forecasting methodology used in this project."
    )
    assert res["status"] == "success"
    assert res["query_type"] == "DOCS"
    assert res["template_name"] == "document_rag"

    prose = res["prose"]
    assert "anti-leakage" in prose.lower() or "leakage" in prose.lower()
    assert "backtesting" in prose.lower() or "chronological" in prose.lower()
    assert any(h in prose for h in ["7", "14", "30"])  # forecast horizons
    assert "19512" not in prose


def test_step3_issue4_forecasting_models_docs_query(db_session):
    """
    Test D: Forecasting models.
    Query: "What models are used for demand forecasting?"
    Expected:
    - Route = DOCS
    - Template = document_rag
    - Actual documented model names: Ridge Regression, Naive, Seasonal Naive, HistGradientBoosting, Prophet, Croston
    """
    res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="What models are used for demand forecasting?"
    )
    assert res["status"] == "success"
    assert res["query_type"] == "DOCS"
    assert res["template_name"] == "document_rag"

    prose = res["prose"]
    assert "Ridge" in prose
    assert "Naive" in prose
    assert "Prophet" in prose
    assert "HistGradientBoosting" in prose or "GBT" in prose


def test_step3_issue5_inventory_recommendation_generation_docs_query(db_session):
    """
    Test E: Inventory recommendation methodology.
    Query: "How are inventory recommendations generated?"
    Expected:
    - Route = DOCS
    - Template = document_rag
    - Relevant inventory decision documentation (SS, ROP, TSL, Lead Time)
    - Does NOT return SKU-specific database information unless explicitly asked
    """
    res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="How are inventory recommendations generated?"
    )
    assert res["status"] == "success"
    assert res["query_type"] == "DOCS"
    assert res["template_name"] == "document_rag"
    assert res["template_name"] != "sku_inventory_recommendation"

    prose = res["prose"]
    assert "ROP" in prose or "Reorder Point" in prose
    assert "Safety Stock" in prose or "SS" in prose
    assert "19512" not in prose


def test_step3_genuine_followup_preserves_sku_context(db_session):
    """
    Test F: Genuine follow-up after SKU context must preserve the SKU.
    Previous: "Tell me about SKU 19512."
    Current:  "What is its reorder point?"
    Expected: SKU 19512 context preserved and addressed.
    """
    t1 = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="Tell me about SKU 19512."
    )
    sess_id = t1["session_id"]
    assert sess_id is not None

    t2 = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="What is its reorder point?",
        session_id=sess_id
    )
    assert t2["status"] == "success"
    assert t2["template_name"] == "sku_inventory_recommendation"
    assert "19512" in t2["prose"]


def test_step3_existing_working_docs_and_hybrid_queries(db_session):
    """
    Test G: Existing working queries must continue passing:
    - "How is safety stock calculated?" -> DOCS, document_rag
    - "What is the formula for safety stock?" -> DOCS, document_rag
    - "How is reorder point calculated?" -> DOCS, document_rag
    - "Why is SKU 19512 recommended for reorder?" -> HYBRID, sku_inventory_recommendation
    """
    # 1. How is safety stock calculated?
    r1 = decision_rag_synthesizer.synthesize(db=db_session, query="How is safety stock calculated?")
    assert r1["query_type"] == "DOCS"
    assert r1["template_name"] == "document_rag"
    assert "King's Formula" in r1["prose"]

    # 2. What is the formula for safety stock?
    r2 = decision_rag_synthesizer.synthesize(db=db_session, query="What is the formula for safety stock?")
    assert r2["query_type"] == "DOCS"
    assert r2["template_name"] == "document_rag"
    assert "King's Formula" in r2["prose"]

    # 3. How is reorder point calculated?
    r3 = decision_rag_synthesizer.synthesize(db=db_session, query="How is reorder point calculated?")
    assert r3["query_type"] == "DOCS"
    assert r3["template_name"] == "document_rag"
    assert "ROP" in r3["prose"]

    # 4. Why is SKU 19512 recommended for reorder?
    r4 = decision_rag_synthesizer.synthesize(db=db_session, query="Why is SKU 19512 recommended for reorder?")
    assert r4["query_type"] == "HYBRID"
    assert r4["template_name"] == "sku_inventory_recommendation"
    assert "19512" in r4["prose"]



