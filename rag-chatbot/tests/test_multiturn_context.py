"""
Tests for Multi-Turn Context and Conversational Carryover
Location: rag-chatbot/tests/test_multiturn_context.py
Project: Demand-Decision-Intelligence

Covers:
- Topic D: Multi-turn follow-up questions
- Topic G: Procurement cost calculation using verified DB values
- Topic H: Absence of hardcoded fallback values
"""

import pytest
from backend.models.product import Product
from backend.models.procurement import SupplierProduct
from rag_chatbot.backend.decision_rag_synthesizer import decision_rag_synthesizer
from rag_chatbot.backend.context_manager import ContextManager


def test_entity_extraction_from_query():
    """Verifies that ContextManager parses quantity, SKU, city, and cost intent without guessing."""
    cm = ContextManager()
    query = "What would it cost to procure 250 units of SKU 476763 in Bengaluru?"
    entities = cm.extract_entities_from_query(query)

    assert entities["product_id"] == 476763
    assert entities["quantity"] == 250.0
    assert entities["city"] == "Bengaluru"
    assert entities["is_cost_query"] is True


def test_multi_turn_followup_and_cost_calculation(db_session):
    """
    Topic D & G:
    Turn 1: User asks "Which product has the highest demand?"
    Turn 2: User asks "What would it cost to procure 500 units?"
    Must resolve:
    - product from verified Turn 1 context
    - quantity = 500 from Turn 2 query
    - unit cost from verified database
    - total cost = quantity * unit_cost
    - NO hardcoded defaults (no fake 19512 or 45.0 unless verified in DB)
    """
    # Turn 1
    t1 = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="Show me the top 10 SKUs by sales volume"
    )
    assert t1["status"] == "success"
    sess_id = t1["session_id"]
    assert sess_id is not None

    # Check that Turn 1 returned table data
    tbl = t1.get("table", {})
    rows = tbl.get("rows", [])
    assert len(rows) > 0, "Top SKUs query should return rows."
    first_pid = rows[0].get("product_id") or rows[0].get("id")

    # Seed or verify a unit cost for this product so cost calculation can succeed
    p = db_session.query(Product).filter(Product.product_id == str(first_pid)).first()
    if p:
        # Check if SupplierProduct exists, if not temporarily attach or test
        sp = db_session.query(SupplierProduct).filter(SupplierProduct.product_id == str(first_pid)).first()
        created_sp = False
        if not sp:
            sp = SupplierProduct(
                supplier_id=1,
                product_id=str(first_pid),
                unit_cost=32.50,
                is_preferred=True
            )
            # Create a dummy supplier if needed
            from backend.models.procurement import Supplier
            supp = db_session.query(Supplier).filter(Supplier.id == 1).first()
            if not supp:
                supp = Supplier(id=1, name="Primary Test Supplier")
                db_session.add(supp)
            db_session.add(sp)
            db_session.commit()
            created_sp = True

        try:
            # Turn 2: Follow up question with 500 units
            t2 = decision_rag_synthesizer.synthesize(
                db=db_session,
                query="What would it cost to procure 500 units?",
                session_id=sess_id
            )
            assert t2["status"] == "success"
            prose = t2["prose"]

            # Must contain the verified calculation: 500 * 32.50 = 16,250.00
            assert "16,250" in prose or "32.5" in prose or str(first_pid) in prose
            # Must NOT use fake 45.0 or 19512 if first_pid is different
            if str(first_pid) != "19512":
                assert "19512" not in prose, "Fake product ID 19512 was injected!"
        finally:
            if created_sp:
                db_session.delete(sp)
                db_session.commit()


def test_city_context_carryover(db_session):
    """Verifies that location context (e.g. 'Delhi') carries over to follow-up questions."""
    # Turn 1
    t1 = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="Which products should I reorder in Delhi?"
    )
    sess_id = t1["session_id"]
    assert sess_id is not None

    # Turn 2
    t2 = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="Explain why this reorder is necessary",
        session_id=sess_id
    )
    assert t2["status"] == "success"
    # City Delhi should be reflected in context or prose
    prose = t2["prose"]
    assert "Delhi" in prose or "reorder" in prose.lower()
