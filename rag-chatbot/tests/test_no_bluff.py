"""
Tests for Strict No-Bluff Rule and Absence of Hardcoded Fallbacks
Location: rag-chatbot/tests/test_no_bluff.py
Project: Demand-Decision-Intelligence

Covers:
- Topic E: Missing-context questions
- Topic F: No-bluff behavior and source integrity
- Topic H: Absence of hardcoded fallback values (19512, Delhi, 500, 45.0, 95% rule)
"""

from pathlib import Path
import pytest
from rag_chatbot.backend.decision_rag_synthesizer import decision_rag_synthesizer


def test_missing_context_procurement_cost_rejection(db_session):
    """
    Topic E & F:
    When a user asks "What is the total procurement cost?" without prior context,
    the system must NOT invent values (e.g. 500 units of 19512 at 45.0).
    It must clearly state that it lacks verified data to calculate the cost.
    """
    res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="What is the total procurement cost?"
    )
    assert res["status"] == "success"
    prose = res["prose"]

    # Must refuse to calculate with clear message
    assert "I don't have enough verified data" in prose or "cannot verify" in prose
    # Critical Topic H: Must NEVER inject hardcoded values
    assert "₹45.00" not in prose, "Hardcoded unit_cost 45.0 was injected!"
    assert "₹22,500" not in prose, "Hardcoded fake calculation (500 * 45) was injected!"


def test_zero_hardcoded_defaults_across_queries(db_session):
    """
    Topic H: Verifies that arbitrary unverified queries do not inject
    hardcoded business values:
    product_id = 19512
    city = "Delhi"
    quantity = 500
    unit_cost = 45.0
    """
    queries = [
        "What would it cost to procure inventory?",
        "How much should I spend on replenishments?",
        "What is my supplier order expense?",
    ]
    for q in queries:
        res = decision_rag_synthesizer.synthesize(db=db_session, query=q)
        prose = res["prose"]
        assert "₹45.00" not in prose, f"Fake unit_cost 45.0 found for query '{q}'"
        assert "₹22,500" not in prose, f"Fake total cost found for query '{q}'"


def test_source_verification_and_no_hallucinated_files(db_session):
    """
    Topic F: All document citations in sources must correspond to real files on disk.
    """
    res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="What is the supply chain architecture and system design?"
    )
    sources = res.get("sources", [])
    project_root = Path(__file__).resolve().parent.parent.parent

    for src in sources:
        if src.endswith(".md") or "/" in src or "\\" in src:
            potential_paths = [
                project_root / src,
                project_root / "docs" / src,
                project_root / Path(src).name,
            ]
            exists = any(p.exists() for p in potential_paths)
            assert exists, f"Hallucinated non-existent document source returned: {src}"


def test_standardized_six_part_markdown_structure(db_session):
    """
    Verifies that responses adhere strictly to the 6-part standardized executive markdown:
    ### Answer
    ### What this means
    ### Key numbers
    ### Recommended action
    ### Evidence
    ### Source
    """
    res = decision_rag_synthesizer.synthesize(
        db=db_session,
        query="What algorithms are used in forecasting?"
    )
    prose = res["prose"]
    assert "### Answer" in prose
    assert "### What this means" in prose
    assert "### Key numbers" in prose
    assert "### Recommended action" in prose
    assert "### Evidence" in prose
    assert "### Source" in prose


def test_fix_3_procurement_cost_case_a_quantity_known_unit_cost_missing(db_session):
    """
    FIX 3 Case A: Quantity is known (500 units) but unit cost is missing in DB.
    Response must display actual quantity (500 units), not 'Unknown',
    must say unit cost is not available, and must reject calculation without inventing costs.
    """
    query = "What would it cost to procure 500 units of SKU 19512?"
    res = decision_rag_synthesizer.synthesize(db=db_session, query=query)
    assert res["status"] == "success"
    prose = res["prose"]

    # Rejection behavior
    assert "I don't have enough verified data to calculate the procurement cost." in prose

    # Quantity known assertion (FIX 3)
    assert "500 units" in prose
    assert "Verified Quantity: Unknown" not in prose
    assert "Quantity: Unknown" not in prose

    # Unit cost missing assertion
    assert "Unit Cost: Not available" in prose or "Not available in verified database" in prose

    # Section structure
    assert "### Verified Data" in prose
    assert "### Result" in prose

    # No bluff assertion
    assert "₹45.00" not in prose
    assert "₹22,500" not in prose


def test_fix_3_procurement_cost_case_b_quantity_missing_unit_cost_missing(db_session):
    """
    FIX 3 Case B: Quantity is missing and unit cost is missing.
    Response must show 'Quantity: Not provided' and 'Unit Cost: Not available'.
    """
    query = "What would it cost to procure SKU 19512?"
    res = decision_rag_synthesizer.synthesize(db=db_session, query=query)
    assert res["status"] == "success"
    prose = res["prose"]

    assert "I don't have enough verified data to calculate the procurement cost." in prose
    assert "Quantity: Not provided" in prose
    assert "Unit Cost: Not available" in prose
    assert "### Verified Data" in prose
    assert "### Result" in prose


def test_fix_3_procurement_cost_case_c_quantity_known_unit_cost_available(db_session):
    """
    FIX 3 Case C: Quantity is known (500 units) and verified unit cost is available in DB.
    Response must calculate the procurement cost accurately (500 * unit_cost) and NOT reject.
    """
    from backend.models.procurement import Supplier, SupplierProduct
    supp = db_session.query(Supplier).filter(Supplier.id == 1).first()
    created_supp = False
    if not supp:
        supp = Supplier(id=1, name="Primary Supplier")
        db_session.add(supp)
        db_session.commit()
        created_supp = True

    # Temporarily attach unit_cost = 25.00 to SKU 19512
    sp = SupplierProduct(supplier_id=1, product_id="19512", unit_cost=25.00, is_preferred=True)
    db_session.add(sp)
    db_session.commit()

    try:
        query = "What would it cost to procure 500 units of SKU 19512?"
        res = decision_rag_synthesizer.synthesize(db=db_session, query=query)
        assert res["status"] == "success"
        prose = res["prose"]

        # Calculation verified: 500 * 25 = 12,500
        assert "12,500" in prose
        assert "25.00" in prose
        assert "500 units" in prose
        # Must NOT reject since verified data is available
        assert "I don't have enough verified data" not in prose
    finally:
        db_session.delete(sp)
        db_session.commit()
        if created_supp:
            db_session.delete(supp)
            db_session.commit()
