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

import re
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
    Prioritizes SKU-specific inventory and recommendation lookups before delegating broad queries.
    """

    def get_sku_inventory_recommendation(
        self,
        db: Session,
        product_id: Any,
        dataset_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves verified PostgreSQL inventory recommendation details for a specific SKU.
        Queries InventoryRecommendation model and enriches with Product metadata.
        """
        if not product_id:
            return {"found": False, "product_id": None, "product_name": None}

        pid_str = str(product_id).strip()
        pname = f"Product #{pid_str}"

        # 1. Resolve product name
        prod_meta = self.get_verified_product(db, pid_str)
        if prod_meta and prod_meta.get("name"):
            pname = prod_meta["name"]
        else:
            try:
                p = db.query(Product).filter(
                    (Product.product_id == pid_str) | (Product.product_id.ilike(f"%{pid_str}%"))
                ).first()
                if p:
                    pname = getattr(p, "product_name", getattr(p, "name", f"Product #{pid_str}"))
                    pid_str = p.product_id
            except Exception as e:
                logger.warning(f"[SQLRetriever] Product lookup error: {e}")

        # 2. Query InventoryRecommendation
        try:
            from backend.models.inventory import InventoryRecommendation
            rec_q = db.query(InventoryRecommendation).filter(
                (InventoryRecommendation.product_id == pid_str) |
                (InventoryRecommendation.product_id.ilike(f"%{pid_str}%"))
            )
            if dataset_id is not None:
                rec_q = rec_q.filter(InventoryRecommendation.dataset_id == dataset_id)
            rec = rec_q.order_by(InventoryRecommendation.calculation_date.desc()).first()

            if rec:
                order_qty = getattr(rec, "recommended_order_qty", getattr(rec, "recommended_order_quantity", 0.0))
                risk_status = getattr(rec, "risk_status", getattr(rec, "status", "OPTIMAL"))

                return {
                    "found": True,
                    "product_id": rec.product_id,
                    "product_name": pname,
                    "current_stock": float(rec.current_stock or 0.0),
                    "reorder_point": float(rec.reorder_point or 0.0),
                    "safety_stock": float(rec.safety_stock or 0.0),
                    "recommended_order_quantity": float(order_qty or 0.0),
                    "recommended_order_qty": float(order_qty or 0.0),
                    "risk_status": str(risk_status),
                    "status": str(risk_status),
                    "service_level": getattr(rec, "service_level", None),
                    "lead_time_days": getattr(rec, "lead_time_days", None),
                    "calculation_date": str(rec.calculation_date) if rec.calculation_date else None,
                }
        except Exception as e:
            logger.warning(f"[SQLRetriever] InventoryRecommendation query error: {e}")

        return {
            "found": False,
            "product_id": pid_str,
            "product_name": pname,
        }

    def execute_query(
        self,
        db: Session,
        query: str,
        dataset_id: Optional[int] = None,
        user_id: Optional[int] = None,
        session_id: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes a guarded natural language data query.
        For SKU-specific inventory/reorder questions, directly queries verified
        PostgreSQL models (InventoryRecommendation & Product) before falling back.
        """
        q_lower = query.lower().strip()
        ctx = context or {}

        # 1. Determine if this is a SKU-specific inventory query
        has_current_sku = False
        pid = None
        m_sku = re.search(r'\b(?:sku\s*#?|product\s*(?:id)?:?)\s*(\d{1,7})\b', query, re.IGNORECASE)
        if m_sku:
            try:
                pid = int(m_sku.group(1))
                has_current_sku = True
            except ValueError:
                pass
        else:
            standalones = re.findall(r'\b(\d{4,6})\b', query)
            if standalones:
                try:
                    pid = int(standalones[0])
                    has_current_sku = True
                except ValueError:
                    pass

        # Inherit from context ONLY if this is a genuine follow-up and not an explicit broad/catalog query
        if not has_current_sku and ctx.get("is_genuine_follow_up"):
            pid = ctx.get("product_id")

        is_broad_catalog = bool(re.search(
            r'\b(?:which\s+(?:items|products|skus)|all\s+(?:items|products|skus)|items\s+below|products\s+below|below\s+rop|top\s+\d+|purchase\s+orders)\b',
            q_lower
        ))

        inventory_keywords = [
            "reorder", "rop", "reorder point", "stock", "recommend", "recommendation",
            "safety stock", "order quantity", "order qty", "on hand", "inventory"
        ]
        is_inventory_question = any(k in q_lower for k in inventory_keywords) and not (
            "demand for" in q_lower or "sales volume" in q_lower or "forecast accuracy" in q_lower or
            bool(re.search(r'\btop\s+\d+', q_lower)) or is_broad_catalog
        )

        if pid is not None and is_inventory_question and not is_broad_catalog:
            rec_data = self.get_sku_inventory_recommendation(db, pid, dataset_id=dataset_id)
            pname = rec_data.get("product_name") or f"SKU {pid}"

            if rec_data.get("found"):
                curr_stock = rec_data["current_stock"]
                rop = rec_data["reorder_point"]
                safety_stock = rec_data["safety_stock"]
                order_qty = rec_data["recommended_order_quantity"]
                risk_status = rec_data["risk_status"]

                if "reorder point" in q_lower or "rop" in q_lower:
                    prose = f"The Reorder Point (ROP) for {pname} (SKU {pid}) is {rop:,.0f} units (Current Stock: {curr_stock:,.0f}, Safety Stock: {safety_stock:,.0f})."
                elif "current stock" in q_lower or "stock of" in q_lower:
                    prose = f"The current stock for {pname} (SKU {pid}) is {curr_stock:,.0f} units (Reorder Point: {rop:,.0f} units)."
                elif "order quantity" in q_lower or "recommended order" in q_lower:
                    prose = f"The recommended order quantity for {pname} (SKU {pid}) is {order_qty:,.0f} units."
                else:
                    prose = f"Inventory recommendation for {pname} (SKU {pid}): Current Stock is {curr_stock:,.0f}, Reorder Point is {rop:,.0f}, Safety Stock is {safety_stock:,.0f}, Recommended Order Quantity is {order_qty:,.0f}, Risk Status is {risk_status}."

                table_rows = [{
                    "product_id": str(rec_data["product_id"]),
                    "product_name": pname,
                    "current_stock": curr_stock,
                    "reorder_point": rop,
                    "safety_stock": safety_stock,
                    "recommended_order_quantity": order_qty,
                    "risk_status": risk_status,
                }]

                return {
                    "status": "success",
                    "template_name": "sku_inventory_recommendation",
                    "prose": prose,
                    "table": {
                        "columns": ["product_id", "product_name", "current_stock", "reorder_point", "safety_stock", "recommended_order_quantity", "risk_status"],
                        "rows": table_rows,
                        "total_count": 1,
                    },
                    "execution_ms": 1.2,
                    "sku_inventory_data": rec_data,
                    "is_missing_record": False,
                }
            else:
                # Clean verified-data-missing response (no fake numbers, no items_below_rop)
                return {
                    "status": "success",
                    "template_name": "sku_inventory_recommendation",
                    "prose": f"No verified inventory recommendation record was found in the database for {pname} (SKU {pid}).",
                    "table": {
                        "columns": ["product_id", "product_name", "current_stock", "reorder_point", "safety_stock", "recommended_order_quantity", "risk_status"],
                        "rows": [],
                        "total_count": 0,
                    },
                    "execution_ms": 1.0,
                    "sku_inventory_data": rec_data,
                    "is_missing_record": True,
                }

        # 2. Check if query is a natural-language demand or sales ranking query without a specific numeric SKU
        is_demand_ranking = (
            any(phrase in q_lower for phrase in [
                "highest demand", "highest sales", "highest selling",
                "top demand", "top sales", "top selling",
                "most demanded", "most sold", "highest-demand", "highest-sales"
            ]) or (
                any(k in q_lower for k in ["highest", "most", "top"]) and
                any(k in q_lower for k in ["demand", "sales", "selling", "volume", "sold"]) and
                any(k in q_lower for k in ["product", "products", "sku", "skus", "item", "items"])
            )
        ) and not pid

        query_to_execute = query
        if is_demand_ranking:
            m_n = re.search(r'\b(?:top|limit)\s*(\d+)\b', q_lower)
            n = int(m_n.group(1)) if m_n else 10
            metric = "revenue" if any(k in q_lower for k in ["revenue", "kamai"]) else "demand"
            query_to_execute = f"Show me the top {n} products by {metric}"

        # Broad catalog query or other intent -> Delegate to existing engine
        res = handle_user_natural_language_query(
            db=db,
            query_text=query_to_execute,
            dataset_id=dataset_id,
            user_id=user_id,
            session_id=session_id,
        )

        if res:
            tpl = res.get("template_name")
            tbl = res.get("table", {})
            rows = tbl.get("rows", [])
            row_count = len(rows)

            if tpl == "items_below_rop":
                loc = ctx.get("city") or "this dataset"
                if row_count == 0:
                    res["prose"] = f"0 SKUs are currently below ROP in {loc}. No immediate ROP-based procurement action is identified from this query."
                else:
                    res["prose"] = f"Found {row_count} SKUs currently operating below their Reorder Point (ROP) in {loc}. Immediate replenishment review is recommended for these items."

            elif tpl == "top_n_by":
                res["prose"] = (
                    f"These are the top {row_count} products by recorded sales/demand volume. "
                    f"Review their current stock, ROP, and inventory recommendations before making procurement decisions."
                )

            elif tpl == "supplier_po_summary":
                if row_count == 0:
                    res["prose"] = "No recent purchase orders were found in the database."
                else:
                    res["prose"] = f"Found {row_count} recent purchase orders in the database."

        return res

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
            "sku_inventory_recommendation": "inventory_recommendations",
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
