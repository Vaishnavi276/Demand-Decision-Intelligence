"""
Context Manager — Multi-Turn Conversation Entity Tracking
Location: rag-chatbot/backend/context_manager.py
Project: Demand-Decision-Intelligence

Handles multi-turn conversational context resolution (SKU, City, Order Quantity carryover)
STRICT NO-BLUFF RULE:
- Never injects fake defaults (19512, "Delhi", 500, 45.0).
- Verifies entity data strictly against PostgreSQL or active session messages.
- Flags missing context fields so the synthesizer can decline answering when unverified.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from backend.models.chat import ChatMessage
from backend.models.product import Product

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Extracts, resolves, and tracks conversational entities across multiple turns.
    Strictly avoids hardcoded fallback values.
    """

    SUPPORTED_CITIES = [
        "Delhi", "Mumbai", "Bengaluru", "Bangalore", "Kolkata",
        "Chennai", "Pune", "Hyderabad", "HR-NCR", "Ahmedabad", "Jaipur"
    ]

    def extract_entities_from_query(self, query: str) -> Dict[str, Any]:
        """
        Parses explicit SKU/Product ID, quantity, city, and follow-up intent from the current query.
        """
        entities: Dict[str, Any] = {
            "product_id": None,
            "quantity": None,
            "city": None,
            "is_cost_query": False,
            "is_follow_up": False,
        }
        q_lower = query.lower().strip()

        # 1. Cost intent detection
        cost_keywords = [
            "cost", "kharcha", "paise", "procure", "procurement",
            "purchase cost", "how much will that cost", "what would it cost",
            "calculate cost", "total cost"
        ]
        if any(ck in q_lower for ck in cost_keywords):
            entities["is_cost_query"] = True

        # 2. Follow-up intent detection
        follow_up_cues = [
            "how much", "what will that cost", "cost kya", "kitna kharcha", "kitne paise",
            "why", "why?", "kyu", "kyun", "explain why", "why should i",
            "which one is the most critical", "sabse critical", "sabse zyada",
            "what about mumbai", "and in delhi", "what about delhi", "bengaluru me",
            "same for", "for that", "is it expensive", "can we save money", "procure",
            "what would it cost", "procure 500", "order 500"
        ]
        if any(cue in q_lower for cue in follow_up_cues) or len(q_lower.split()) <= 4:
            entities["is_follow_up"] = True

        # 3. Explicit quantity in current query: e.g. "procure 500 units", "500 units", "order 100"
        qty_patterns = [
            r'(?:procure|order|buy)\s+(\d[\d,]*)\s*(?:units)?',
            r'(\d[\d,]*)\s*(?:units|pcs|pieces|items|qty)\b',
            r'quantity\s*(?:of|=|:)?\s*(\d[\d,]*)',
        ]
        for pat in qty_patterns:
            m = re.search(pat, q_lower)
            if m:
                try:
                    entities["quantity"] = float(m.group(1).replace(",", ""))
                    break
                except ValueError:
                    pass

        # 4. Explicit SKU/Product ID in current query: e.g. "product 19512", "SKU 19512", "sku #19512"
        pid_matches = re.findall(r'(?:product\s*id:?|sku\s*#?|sku|product)\s*(\d{1,7})\b', query, re.IGNORECASE)
        if pid_matches:
            entities["product_id"] = int(pid_matches[0])
        else:
            # Standalone 4-6 digit numbers if not already captured as quantity
            standalones = re.findall(r'\b(\d{4,6})\b', query)
            for s in standalones:
                val = int(s)
                if entities["quantity"] is None or val != int(entities["quantity"]):
                    entities["product_id"] = val
                    break

        # 5. Explicit City
        for c in self.SUPPORTED_CITIES:
            if c.lower() in q_lower:
                entities["city"] = "Bengaluru" if c.lower() == "bangalore" else c
                break

        return entities

    def get_conversation_context(
        self,
        db: Session,
        session_id: Optional[int],
        current_query: str,
        sql_retriever: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Resolves conversational context across the active session and current query.
        ZERO fake data: Does not invent 19512, Delhi, 500, or 45.0.
        """
        curr = self.extract_entities_from_query(current_query)

        ctx: Dict[str, Any] = {
            "product_id": curr["product_id"],
            "product_name": None,
            "city": curr["city"],
            "quantity": curr["quantity"],
            "unit_cost": None,
            "total_cost": None,
            "previous_template": None,
            "previous_query_type": None,
            "previous_table_sample": [],
            "is_follow_up": curr["is_follow_up"],
            "is_cost_query": curr["is_cost_query"],
            "has_verified_cost_inputs": False,
            "missing_cost_fields": [],
        }

        # Inspect session history if session_id is provided
        if session_id:
            try:
                recent_msgs = (
                    db.query(ChatMessage)
                    .filter(ChatMessage.session_id == session_id)
                    .order_by(ChatMessage.created_at.desc())
                    .limit(6)
                    .all()
                )
                recent_msgs.reverse()

                for msg in recent_msgs:
                    text = msg.message or ""
                    rc_data: Dict[str, Any] = {}
                    if msg.retrieved_context:
                        try:
                            rc_data = json.loads(msg.retrieved_context)
                        except Exception:
                            pass

                    # 1. Product carryover if missing
                    if ctx["product_id"] is None:
                        if rc_data.get("product_id"):
                            try:
                                ctx["product_id"] = int(rc_data["product_id"])
                            except (ValueError, TypeError):
                                pass
                        elif rc_data.get("table_sample") and isinstance(rc_data["table_sample"], list):
                            first_row = rc_data["table_sample"][0] if rc_data["table_sample"] else {}
                            if isinstance(first_row, dict):
                                raw_pid = first_row.get("product_id") or first_row.get("id")
                                if raw_pid:
                                    try:
                                        ctx["product_id"] = int(raw_pid)
                                    except (ValueError, TypeError):
                                        pass

                        # If still not found, search in assistant / user message text
                        if ctx["product_id"] is None:
                            pid_matches = re.findall(
                                r'(?:product\s*id:?|sku\s*#?|sku|product)\s*(\d{1,7})\b',
                                text,
                                re.IGNORECASE
                            )
                            if pid_matches:
                                ctx["product_id"] = int(pid_matches[0])
                            else:
                                standalones = re.findall(r'\b(\d{4,6})\b', text)
                                if standalones:
                                    ctx["product_id"] = int(standalones[0])

                    # 2. City carryover if missing
                    if ctx["city"] is None:
                        if rc_data.get("city"):
                            ctx["city"] = rc_data["city"]
                        else:
                            for c in self.SUPPORTED_CITIES:
                                if c.lower() in text.lower():
                                    ctx["city"] = "Bengaluru" if c.lower() == "bangalore" else c
                                    break

                    # 3. Quantity carryover ONLY if not specified in current query
                    if ctx["quantity"] is None:
                        qty_match = re.findall(
                            r'(\d[\d,]*)\s*(?:units|reorder\s*qty|order\s*qty|recommended_qty)',
                            text,
                            re.IGNORECASE
                        )
                        if qty_match:
                            try:
                                ctx["quantity"] = float(qty_match[0].replace(",", ""))
                            except ValueError:
                                pass

                    # 4. Previous template and query type
                    if msg.query_template and not ctx["previous_template"]:
                        ctx["previous_template"] = msg.query_template
                    if rc_data.get("query_type") and not ctx["previous_query_type"]:
                        ctx["previous_query_type"] = rc_data["query_type"]
                    if rc_data.get("table_sample") and not ctx["previous_table_sample"]:
                        ctx["previous_table_sample"] = rc_data["table_sample"]

            except Exception as e:
                logger.warning(f"[ContextManager] Could not read chat history: {e}")

        # Resolve product name and unit cost from verified database
        if ctx["product_id"] is not None:
            if sql_retriever:
                prod = sql_retriever.get_verified_product(db, ctx["product_id"])
                if prod:
                    ctx["product_name"] = prod.get("name")
                    ctx["unit_cost"] = prod.get("cost_price")
            else:
                try:
                    p = db.query(Product).filter(
                        Product.product_id == str(ctx["product_id"])
                    ).first()
                    if p:
                        ctx["product_name"] = getattr(p, "product_name", getattr(p, "name", f"Product #{ctx['product_id']}"))
                        ctx["unit_cost"] = float(p.cost_price) if hasattr(p, "cost_price") and p.cost_price is not None else None
                except Exception as e:
                    logger.warning(f"[ContextManager] Product query error: {e}")

        # Evaluate readiness for procurement cost calculation
        missing = []
        if ctx["product_id"] is None:
            missing.append("product")
        if ctx["quantity"] is None:
            missing.append("quantity")
        if ctx["unit_cost"] is None:
            missing.append("unit cost")

        ctx["missing_cost_fields"] = missing

        if not missing and ctx["quantity"] is not None and ctx["unit_cost"] is not None:
            ctx["has_verified_cost_inputs"] = True
            ctx["total_cost"] = round(ctx["quantity"] * ctx["unit_cost"], 2)

        return ctx
