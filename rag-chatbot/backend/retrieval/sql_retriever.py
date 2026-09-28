"""
SQL Retriever Adapter
Location: rag-chatbot/backend/retrieval/sql_retriever.py
Project: Demand-Decision-Intelligence

Thin adapter interfacing with the existing guarded SQL intelligence layer
(backend.services.nl_query_service). Does NOT duplicate SQL query templates,
calculations, or database persistence.
"""

import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

# Re-use existing backend services without modification
from backend.services.nl_query_service import (
    handle_user_natural_language_query,
    detect_query_language,
    parse_natural_language_intent,
    execute_guarded_template,
)
from backend.models.product import Product

logger = logging.getLogger(__name__)


class SQLRetriever:
    """
    Thin adapter that reuses the existing backend/services/nl_query_service.py
    for guarded, parameterized PostgreSQL query execution.
    """

    def execute_query(
        self,
        db: Session,
        query: str,
        dataset_id: Optional[int] = None,
        user_id: Optional[int] = None,
        session_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes a guarded natural language data query using the existing engine.
        Returns tabular data, template name, execution time, and prose.
        """
        return handle_user_natural_language_query(
            db=db,
            query_text=query,
            dataset_id=dataset_id,
            user_id=user_id,
            session_id=session_id,
        )

    def get_verified_product(self, db: Session, product_id: Any) -> Optional[Dict[str, Any]]:
        """
        Fetches verified product information (name, cost_price/unit_cost) from PostgreSQL.
        Prevents hallucinated or invented unit costs.
        """
        if not product_id:
            return None
        try:
            pid_str = str(product_id).strip()
            p = db.query(Product).filter(
                (Product.product_id == pid_str) | (Product.product_id.ilike(f"%{pid_str}%"))
            ).first()
            if p:
                name = getattr(p, "product_name", getattr(p, "name", f"Product #{pid_str}"))
                unit_cost = None

                # 1. Check SupplierProduct table
                try:
                    from backend.models.procurement import SupplierProduct
                    sp = db.query(SupplierProduct).filter(
                        SupplierProduct.product_id == p.product_id
                    ).order_by(SupplierProduct.is_preferred.desc()).first()
                    if sp and sp.unit_cost is not None:
                        unit_cost = float(sp.unit_cost)
                except Exception:
                    pass

                # 2. Check DeadStockRecord table
                if unit_cost is None:
                    try:
                        from backend.models.dead_stock import DeadStockRecord
                        ds = db.query(DeadStockRecord).filter(DeadStockRecord.product_id == p.product_id).first()
                        if ds and ds.unit_cost is not None:
                            unit_cost = float(ds.unit_cost)
                    except Exception:
                        pass

                # 3. Check cost_price attribute on Product if present
                if unit_cost is None and hasattr(p, "cost_price") and getattr(p, "cost_price") is not None:
                    unit_cost = float(getattr(p, "cost_price"))

                return {
                    "id": p.product_id,
                    "product_id": p.product_id,
                    "name": name,
                    "cost_price": unit_cost,
                    "unit_cost": unit_cost,
                    "category": getattr(p, "l0_category", None),
                }
        except Exception as e:
            logger.warning(f"[SQLRetriever] Could not fetch verified product {product_id}: {e}")
        return None

    def map_template_to_source(self, template_name: str) -> str:
        """
        Maps a guarded template name to its primary underlying database table.
        """
        mapping = {
            "items_below_rop": "inventory_recommendations",
            "stockout_risk_before": "inventory_recommendations",
            "demand_for": "daily_product_demand",
            "top_n_by": "daily_product_demand",
            "compare_periods": "daily_product_demand",
            "dead_stock": "dead_stock_records",
            "supplier_po_summary": "purchase_orders",
            "price_movers": "price_observations",
            "upcoming_festivals": "calendar_events",
            "forecast_accuracy_for": "forecast_evaluations",
            "model_benchmarks": "forecast_runs",
            "inventory_health_overview": "inventory_states",
        }
        return mapping.get(template_name, "PostgreSQL Database")
