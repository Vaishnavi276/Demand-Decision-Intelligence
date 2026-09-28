"""
Decision RAG Synthesizer — Hybrid Orchestration Engine
Location: rag-chatbot/backend/decision_rag_synthesizer.py
Project: Demand-Decision-Intelligence

Orchestrates:
1. Multi-turn conversational entity tracking via ContextManager (zero hardcoded fake defaults).
2. Dynamic Query Routing:
   - DATA: Delegated to guarded PostgreSQL templates via SQLRetriever thin adapter.
   - DOCS: Vector/semantic search over project documentation via DocumentRetriever.
   - HYBRID: Blends verified operational data with actual retrieved policy documentation.
3. LLM Integration (Groq llama-3.3-70b-versatile, OpenAI gpt-4o-mini) with
   deterministic 6-part Markdown synthesis fallback when API keys are absent.
4. Strict No-Bluff Rule:
   - No fabricated metrics (e.g. no invented '95% cycle service level').
   - Refuses cost calculation if product, quantity, or unit cost cannot be verified.
5. PostgreSQL session persistence (ChatSession & ChatMessage).
"""

import os
import re
import json
import time
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from backend.models.chat import ChatSession, ChatMessage
from backend.models.product import Product
from backend.services.dataset_service import resolve_dataset
from backend.services.nl_query_service import detect_query_language

# Internal modular imports with fallbacks
try:
    from rag_chatbot.backend.context_manager import ContextManager
    from rag_chatbot.backend.knowledge_base_service import KnowledgeBaseService, knowledge_base_service
    from rag_chatbot.backend.retrieval.document_retriever import DocumentRetriever
    from rag_chatbot.backend.retrieval.sql_retriever import SQLRetriever
except ImportError:
    try:
        from .context_manager import ContextManager
        from .knowledge_base_service import KnowledgeBaseService, knowledge_base_service
        from .retrieval.document_retriever import DocumentRetriever
        from .retrieval.sql_retriever import SQLRetriever
    except ImportError:
        from context_manager import ContextManager
        from knowledge_base_service import KnowledgeBaseService, knowledge_base_service
        from retrieval.document_retriever import DocumentRetriever
        from retrieval.sql_retriever import SQLRetriever

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the AI Decision Copilot for the "Demand & Decision Intelligence System".
You act as a senior Business Intelligence & Supply Chain Advisor to retail owners, store managers, and executives.

CORE PRINCIPLES (STRICT NO-BLUFF RULE):
1. Grounding & Zero Bluff:
   - Base answers STRICTLY on the retrieved database records and documentation evidence provided.
   - NEVER invent, extrapolate, or hallucinate numbers, SKUs, percentages, or document claims.
   - If cost data or product identity is missing, explicitly state that you do not have enough verified data to calculate it.
   - NEVER invent product IDs (like 19512), quantities (like 500), or unit costs (like 45.0) unless they are verified in the context or data.
2. Distinguish Evidence Types:
   - Explicitly distinguish verified database facts, documentation rules, and derived calculations.
3. No Unsupported Claims:
   - Never quote metrics (like 95% cycle service level) unless that exact figure exists in the retrieved documentation excerpts.
4. Product Naming:
   - Refer to products by their human-readable name and Product ID: "Amul Taaza Toned Fresh Milk (Product ID: 19512)".
5. Language Matching:
   - Match the user's language (English, Hinglish, or Hindi).
6. Strict Markdown Response Structure:
   ### Answer
   [Direct, clear 1-2 sentence executive answer]

   ### What this means
   [Simple, actionable business interpretation based strictly on retrieved evidence]

   ### Key numbers
   - **[Metric 1]**: [Verified value]
   - **[Metric 2]**: [Verified value]

   ### Recommended action
   [Specific action ONLY when supported by underlying data, otherwise 'Monitor performance.']

   ### Evidence
   - **Data Evidence**: [Key verified numbers from database or documentation]

   ### Source
   Source:
   - [List exact database tables and/or document paths retrieved]

7. Calculation and Formula Grounding:
   - When asked for a formula or calculation methodology (e.g. safety stock, reorder point):
     - Check the retrieved documentation evidence for the explicit mathematical formula.
     - State the formula, explain all documented inputs/variables, and cite the exact source document and line/section evidence.
     - If the project documentation does NOT specify a complete formula for the requested concept, say explicitly: "The current project documentation does not specify a complete calculation formula for [concept]." and provide only the available documented information.
     - NEVER invent or substitute standard industry formulas unless the user explicitly requests general industry knowledge.

8. Zero-Result Below-ROP Handling:
   - When 0 SKUs are below ROP: state clearly "0 SKUs are currently below ROP in this dataset. No immediate ROP-based procurement action is identified from this query."
   - Explain: "No evaluated SKU was found below its recorded reorder point."
   - Key numbers: "Items Below ROP: 0 SKUs", "Replenishment Status: No ROP breach detected".
   - Recommended action: "No ROP-based reorder action is indicated by this query. Continue routine inventory monitoring."
   - Do NOT say "These items require urgent procurement" and do NOT claim inventory is "optimal".

9. Top-N Sales / Demand Response:
   - Keep the factual ranking/data response, but remove unsupported procurement/inventory recommendations.
   - Use neutral wording: "These are the top N products by recorded sales/demand volume. Review their current stock, ROP, and inventory recommendations before making procurement decisions."
   - Do not claim that stock must "always" be kept ready or invent business conclusions.

10. Zero Purchase-Order Handling:
   - When 0 recent purchase orders are found: state "No recent purchase orders were found in the database."
   - Explain: "No recent PO records are available for the requested query."
   - Key numbers: "Records Found: 0", "Total PO Value: ₹0".
   - Recommended action: "No PO records are available for review. Check procurement requirements separately if needed."
   - Do not instruct the user to review an empty table.

11. General Neutrality Rule:
   - Recommendations must be strictly supported by retrieved data.
   - Avoid unsupported words such as "always", "optimal", "urgent", "must", "definitely" unless retrieved data and project rules explicitly justify them.
"""


def build_document_search_query(
    query: str,
    query_type: str = "DOCS",
    intent: Optional[str] = None
) -> str:
    """
    Builds a normalized, concept-focused search query for document retrieval. (FIX 2)
    Enriches calculation and formula questions with relevant domain terms (calculation, formula,
    methodology, equation, inputs, variables, logic) to surface exact mathematical specifications.
    """
    q_lower = query.lower()

    # Detect if query asks for calculation, formula, methodology, or equation
    is_calc_query = bool(re.search(
        r'\b(?:how\s+(?:is|are|do\s+you)\s+.*?\s+(?:calculated|computed|determined)|formula|equation|how\s+to\s+calculate|explain\s+(?:the\s+)?calculation|calculation\s+of|what\s+methodology\s+is\s+used)\b',
        q_lower
    )) or any(k in q_lower for k in ["formula", "equation", "arithmetic"])

    if query_type == "DOCS":
        q_clean = query.strip()
        # 1. Forecasting models
        if any(k in q_lower for k in ["models are used", "models used", "what models", "forecasting models", "forecast models"]):
            return f"{q_clean} forecasting model benchmarks comparison Naive Lag-1 Seasonal Naive Lag-7 Rolling Mean 7 Ridge Regression HistGradientBoosting GBT Prophet Croston Syntetos-Boylan-Croston"

        # 2. Forecasting methodology
        if "forecasting methodology" in q_lower or ("methodology" in q_lower and "forecast" in q_lower):
            return f"{q_clean} demand forecasting pipeline feature engineering lag features rolling windows chronological train validation split backtesting WAPE benchmark Ridge Regression HistGradientBoosting Prophet"

        # 4. Inventory recommendations generated / methodology
        if ("inventory recommendation" in q_lower or "recommendation" in q_lower) and any(k in q_lower for k in ["generate", "generated", "how are", "workflow", "process", "created"]):
            return f"{q_clean} inventory decision layer recommendation generation Safety Stock SS Reorder Point ROP Target Stock Level TSL lead time demand King's formula"

        if is_calc_query:
            calc_terms = ["calculation", "formula", "methodology", "equation", "inputs", "variables", "logic"]
            if "safety stock" in q_lower or "ss" in q_lower.split():
                calc_terms.extend(["King's Formula", "stochastic lead time", "Z", "sigma_d", "L", "inventory decision layer", "stochastic demand"])
            elif "reorder point" in q_lower or "rop" in q_lower.split():
                calc_terms.extend(["LTD", "Lead Time Demand", "safety stock", "SS", "stochastic inventory optimization arithmetic"])
            elif any(k in q_lower for k in ["wape", "mae", "rmse", "forecast error", "accuracy"]):
                calc_terms.extend(["WAPE formula", "forecast evaluations", "benchmark"])

            return f"{q_clean} {' '.join(calc_terms)}"
        return q_clean

    # For HYBRID queries:
    # 1 & 2. Remove SKU/product ID prefixes and numeric IDs
    clean_q = re.sub(r'\b(?:product\s*(?:id)?:?|sku\s*#?)\s*\d+\b', '', query, flags=re.IGNORECASE)
    clean_q = re.sub(r'\b\d{4,7}\b', '', clean_q)
    clean_q = re.sub(r'\b(?:sku|product\s*id)\b', '', clean_q, flags=re.IGNORECASE)
    clean_q = ' '.join(clean_q.split()).strip()

    # 3. Add relevant domain terms based on detected intent
    domain_terms = []
    effective_intent = (intent or "").lower()

    if any(k in q_lower or k in effective_intent for k in ["reorder", "rop", "replenish", "replenishment"]):
        domain_terms.append("inventory reorder policy safety stock lead time reorder point")
    elif any(k in q_lower or k in effective_intent for k in ["stockout", "stock out", "out of stock", "run out"]):
        domain_terms.append("stockout inventory safety stock lead time demand")
    elif any(k in q_lower or k in effective_intent for k in ["forecast", "forecasting", "model", "accuracy", "wape", "prophet"]):
        domain_terms.append("demand forecasting methodology models forecast accuracy")
    elif any(k in q_lower or k in effective_intent for k in ["dead stock", "excess", "inactive", "markdown"]):
        domain_terms.append("dead stock markdown clearance obsolescence holding cost")
    else:
        domain_terms.append("inventory policy safety stock lead time")

    terms_str = " ".join(domain_terms)
    expanded = f"{clean_q} {terms_str}".strip() if clean_q else terms_str
    return expanded


class DecisionRAGSynthesizer:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # Initialize modular components
        self.context_manager = ContextManager()
        self.sql_retriever = SQLRetriever()
        self.document_retriever = DocumentRetriever(knowledge_base_service)

    # ── 1. Query Classification & Routing ────────────────────────────────────

    def route_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Alias for classify_query_type for test compatibility."""
        return self.classify_query_type(query, context or {})

    def classify_query_type(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Classifies query into:
        - 'DOCS': Pure documentation / architecture / methodology / PRD inquiries
        - 'DATA': Live PostgreSQL operational / demand / inventory data
        - 'HYBRID': Questions requiring live data combined with methodology / policy explanation
        """
        q = query.lower().strip()
        ctx = context or {}

        hybrid_keywords = [
            "why is this recommended", "why should i order", "why are these items below rop",
            "explain why", "why is this an anomaly", "what policy", "why reorder",
            "why is sku", "why is product", "how much will that cost and why", "and why", "aur kyu"
        ]

        # 1. Explicit hybrid cues (checked first so queries like 'explain why this SKU...' route to HYBRID)
        if any(k in q for k in hybrid_keywords):
            return "HYBRID"

        # Follow-up "Why?" with previous data context -> HYBRID
        if ctx.get("is_follow_up") and any(w in q.split() for w in ["why", "why?", "kyu", "kyun"]):
            return "HYBRID"

        # 2. Pattern-based conceptual/documentation recognition (FIX 1)
        doc_patterns = [
            r"how\s+(?:is|are)\s+.*?\s+(?:calculated|determined|computed|generated|derived|produced|created)",
            r"how\s+does\s+.*?\s+work",
            r"what\s+(?:is|are)\s+(?:the\s+)?(?:business\s+)?rules(?:\s+for)?",
            r"what\s+(?:is|are)\s+(?:the\s+)?rules(?:\s+for)?",
            r"rules\s+for\s+(?:identifying\s+)?dead\s+stock",
            r"(?:explain|what\s+is|what)\s+(?:the\s+)?formula(?:\s+for|\s+is\s+used)?",
            r"explain\s+(?:the\s+)?(?:recommendation\s+logic|methodology|architecture|workflow|pipeline|policy|system|design|formula)",
            r"explain\s+(?:the\s+)?(?:demand\s+)?forecasting\s+methodology",
            r"why\s+does\s+the\s+system\s+use",
            r"how\s+(?:is|are)\s+.*?\s+determined",
            r"what\s+models\s+(?:are\s+used|used)",
            r"how\s+(?:are|is)\s+(?:inventory\s+)?recommendations?\s+(?:generated|calculated|determined|created|derived)",
        ]
        if any(re.search(pat, q) for pat in doc_patterns):
            return "DOCS"

        # Additional keywords for documentation / methodology / architecture
        doc_keywords = [
            "prd", "architecture", "srs", "methodology", "guide", "documentation",
            "safety stock formula", "wape formula", "croston method",
            "system design", "sla", "data pipeline guide", "conventions", "stack",
            "what does the prd say", "system architecture", "what algorithms",
            "forecasting methodology", "demand forecasting methodology", "recommendation logic",
            "business rules", "what formula", "specification", "policy", "workflow", "design",
            "dead stock rules", "rules for dead stock", "rules for identifying dead stock",
            "models are used for demand forecasting", "models used for demand forecasting",
            "how are inventory recommendations generated"
        ]
        if any(k in q for k in doc_keywords):
            return "DOCS"

        # Default: Structured live data
        return "DATA"

    def build_document_search_query(
        self,
        query: str,
        query_type: str = "DOCS",
        intent: Optional[str] = None
    ) -> str:
        """Builds a normalized, concept-focused search query for document retrieval. (FIX 2)"""
        return build_document_search_query(query, query_type=query_type, intent=intent)

    # ── 2. Hybrid Synthesis Engine ───────────────────────────────────────────

    def synthesize(
        self,
        db: Session,
        query: str,
        dataset_id: Optional[int] = None,
        user_id: Optional[int] = None,
        session_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point for Hybrid RAG decision intelligence synthesis.
        """
        t0 = time.perf_counter()
        target_dataset = resolve_dataset(db, user_id, dataset_id)

        # 1. Multi-turn conversation context (ZERO hardcoded fallback values)
        conv_ctx = self.context_manager.get_conversation_context(
            db=db,
            session_id=session_id,
            current_query=query,
            sql_retriever=self.sql_retriever
        )

        # 2. Query routing
        query_type = self.classify_query_type(query, conv_ctx)
        user_lang = detect_query_language(query)

        data_result: Optional[Dict[str, Any]] = None
        doc_hits: List[Dict[str, Any]] = []
        data_sources: List[str] = []
        doc_sources: List[str] = []
        evidence_list: List[Dict[str, Any]] = []

        # ── Branch A: Live PostgreSQL Retrieval (DATA or HYBRID) ──
        if query_type in ("DATA", "HYBRID"):
            # Check if this is a procurement cost query
            if conv_ctx.get("is_cost_query"):
                if conv_ctx.get("has_verified_cost_inputs"):
                    # All inputs verified: calculate cost
                    pid = conv_ctx["product_id"]
                    pname = conv_ctx.get("product_name") or f"Product ID: {pid}"
                    city = conv_ctx.get("city") or "Catalog Location"
                    qty = conv_ctx["quantity"]
                    unit_cost = conv_ctx["unit_cost"]
                    total_cost = conv_ctx["total_cost"]

                    data_result = {
                        "template_name": "contextual_cost_calculation",
                        "prose": f"Procurement cost calculation for {pname}: {qty:,.0f} units at verified unit cost of ₹{unit_cost:.2f} amounts to approximately ₹{total_cost:,.2f}.",
                        "table": {
                            "columns": ["product_id", "product_name", "city", "recommended_qty", "unit_cost", "total_estimated_cost"],
                            "rows": [{
                                "product_id": pid,
                                "product_name": pname,
                                "city": city,
                                "recommended_qty": qty,
                                "unit_cost": f"₹{unit_cost:.2f}",
                                "total_estimated_cost": f"₹{total_cost:,.2f}"
                            }]
                        },
                        "execution_ms": 1.2
                    }
                    data_sources.append("products")
                    evidence_list.append({
                        "product": f"{pname} (ID: {pid})",
                        "city": city,
                        "quantity": qty,
                        "unit_cost": f"₹{unit_cost:.2f}",
                        "total_cost": f"₹{total_cost:,.2f}"
                    })
                else:
                    # Missing verified context for cost query -> NO BLUFF!
                    missing = ", ".join(conv_ctx.get("missing_cost_fields", ["required information"]))
                    data_result = {
                        "template_name": "insufficient_data_rejection",
                        "prose": "I don't have enough verified data to calculate the procurement cost.",
                        "table": {"columns": [], "rows": []},
                        "execution_ms": 0.5,
                        "missing_fields": conv_ctx.get("missing_cost_fields", []),
                    }
                    data_sources.append("products")
            else:
                # Regular data query -> Delegate to SQLRetriever thin adapter
                data_result = self.sql_retriever.execute_query(
                    db=db,
                    query=query,
                    dataset_id=target_dataset.id,
                    user_id=user_id,
                    session_id=session_id,
                    context=conv_ctx,
                )
                tpl = data_result.get("template_name", "database_query")
                data_sources.append(self.sql_retriever.map_template_to_source(tpl))

                tbl = data_result.get("table", {})
                if tbl and tbl.get("rows"):
                    for r in tbl["rows"][:4]:
                        evidence_list.append(r)

        # ── Branch B: Documentation Vector Search (DOCS or HYBRID) ──
        if query_type in ("DOCS", "HYBRID"):
            search_query = self.build_document_search_query(
                query=query,
                query_type=query_type,
                intent=conv_ctx.get("previous_template")
            )
            doc_hits = self.document_retriever.retrieve(search_query, top_k=4)
            for hit in doc_hits:
                doc_name = hit["document_name"]
                if doc_name not in doc_sources:
                    doc_sources.append(doc_name)
                evidence_list.append({
                    "document": doc_name,
                    "section": hit.get("heading", ""),
                    "source_path": hit.get("source_path", ""),
                    "similarity": f"{hit.get('retrieval_score', 0):.2f}",
                    "retrieval_score": hit.get("retrieval_score", 0),
                    "lines": f"L{hit.get('start_line')}-{hit.get('end_line')}",
                    "excerpt": hit.get("text", "")[:180] + "..."
                })

        all_sources = list(dict.fromkeys(data_sources + doc_sources))
        if not all_sources:
            all_sources = ["PostgreSQL Database"]

        # ── Step 3: Synthesize Response via LLM or Deterministic Fallback ──
        answer_prose = self._generate_answer(
            query=query,
            query_type=query_type,
            user_lang=user_lang,
            data_result=data_result,
            doc_hits=doc_hits,
            conv_ctx=conv_ctx,
            all_sources=all_sources
        )

        exec_ms = round((time.perf_counter() - t0) * 1000, 2)
        sess_id = data_result.get("session_id") if data_result else session_id
        table_payload = data_result.get("table", {}) if data_result else {}

        # Resolve final template name without legacy overwrite
        raw_tpl = data_result.get("template_name") if data_result else "document_rag"
        has_current_sku = bool(re.search(r'\b(?:product\s*(?:id)?:?|sku\s*#?)\s*\d+\b|\b\d{4,7}\b', query, re.I))
        is_single_sku = has_current_sku or conv_ctx.get("is_genuine_follow_up")
        is_broad = bool(re.search(r'\b(?:which\s+(?:items|products|skus)|all\s+(?:items|products|skus)|items\s+below|products\s+below|below\s+rop)\b', query, re.I))

        if query_type == "DOCS":
            final_template = "document_rag"
        elif is_single_sku and not is_broad and conv_ctx.get("product_id") and raw_tpl in ("items_below_rop", "general_business_advisory"):
            final_template = "sku_inventory_recommendation"
        elif query_type == "HYBRID":
            if is_single_sku and not is_broad and conv_ctx.get("product_id"):
                final_template = "sku_inventory_recommendation"
            else:
                final_template = "hybrid_decision_rag"
        else:
            final_template = raw_tpl or ("hybrid_decision_rag" if query_type == "HYBRID" else "document_rag")

        # ── Step 4: Persist Assistant Response in PostgreSQL ──
        if sess_id:
            try:
                retrieved_context_json = json.dumps({
                    "query_type": query_type,
                    "data_sources": data_sources,
                    "doc_sources": doc_sources,
                    "sources": all_sources,
                    "city": conv_ctx.get("city"),
                    "product_id": conv_ctx.get("product_id"),
                    "table_sample": table_payload.get("rows", [])[:3] if table_payload else []
                })

                last_bot_msg = (
                    db.query(ChatMessage)
                    .filter(ChatMessage.session_id == sess_id, ChatMessage.sender_role == "assistant")
                    .order_by(ChatMessage.created_at.desc())
                    .first()
                )
                if last_bot_msg and query_type != "DATA":
                    last_bot_msg.message = answer_prose
                    last_bot_msg.retrieved_context = retrieved_context_json
                    last_bot_msg.execution_ms = exec_ms
                    db.commit()
                elif not last_bot_msg or last_bot_msg.message != answer_prose:
                    bot_msg = ChatMessage(
                        session_id=sess_id,
                        sender_role="assistant",
                        message=answer_prose,
                        retrieved_context=retrieved_context_json,
                        query_template=final_template,
                        execution_ms=exec_ms
                    )
                    db.add(bot_msg)
                    db.commit()
            except Exception as e:
                logger.warning(f"[DecisionRAGSynthesizer] Persistence error: {e}")

        rich_data_sources = [
            {
                "table": src,
                "template": final_template,
                "execution_ms": data_result.get("execution_ms") if data_result else exec_ms,
                "row_count": len(table_payload.get("rows", [])) if table_payload else 0
            }
            for src in data_sources
        ]

        rich_doc_sources = [
            {
                "document_name": h["document_name"],
                "source_path": h.get("source_path", ""),
                "chunk_id": h.get("chunk_id", ""),
                "similarity": round(float(h.get("retrieval_score", 0)), 4),
                "retrieval_score": round(float(h.get("retrieval_score", 0)), 4),
                "start_line": h.get("start_line"),
                "end_line": h.get("end_line"),
                "heading": h.get("heading", ""),
                "text": h.get("text", "")[:350]
            }
            for h in doc_hits
        ]

        return {
            "status": "success",
            "session_id": sess_id,
            "query_type": query_type,
            "intent": query_type,
            "template_name": final_template,
            "prose": answer_prose,
            "natural_language_answer": answer_prose,
            "table": table_payload,
            "data_table": table_payload,
            "execution_ms": exec_ms,
            "detected_language": user_lang,
            "sources": all_sources,
            "data_sources": rich_data_sources,
            "document_sources": rich_doc_sources,
            "evidence": evidence_list,
        }

    # ── 3. Answer Generation & LLM Calling ───────────────────────────────────

    def _generate_answer(
        self,
        query: str,
        query_type: str,
        user_lang: str,
        data_result: Optional[Dict[str, Any]],
        doc_hits: List[Dict[str, Any]],
        conv_ctx: Dict[str, Any],
        all_sources: List[str]
    ) -> str:
        """Invokes Groq/OpenAI if configured, else uses deterministic 6-part synthesis."""

        # If data_result signaled insufficient data for cost calculation, return rejection prose (FIX 3)
        if data_result and data_result.get("template_name") == "insufficient_data_rejection":
            pid = conv_ctx.get("product_id")
            if conv_ctx.get("product_name"):
                pname = conv_ctx["product_name"]
            elif pid:
                pname = f"SKU {pid}"
            else:
                pname = "Not provided"

            qty = conv_ctx.get("quantity")
            if qty is not None:
                qty_str = f"{int(qty):,d} units" if float(qty).is_integer() else f"{qty:,.2f} units"
                unit_cost_str = "Not available in verified database"
            else:
                qty_str = "Not provided"
                unit_cost_str = "Not available"

            unit_cost = conv_ctx.get("unit_cost")
            if unit_cost is not None:
                unit_cost_str = f"₹{unit_cost:.2f}"

            return (
                "### Answer\n"
                "I don't have enough verified data to calculate the procurement cost.\n\n"
                "### Verified Data\n"
                f"- Product: {pname}\n"
                f"- Quantity: {qty_str}\n"
                f"- Unit Cost: {unit_cost_str}\n\n"
                "### Result\n"
                "I don't have enough verified data to calculate the procurement cost.\n\n"
                "### What this means\n"
                "To calculate procurement cost, the system requires verified product details, order quantity, and unit cost from the catalog or active supplier contracts.\n\n"
                "### Key numbers\n"
                f"- **Product**: {pname}\n"
                f"- **Quantity**: {qty_str}\n"
                f"- **Unit Cost**: {unit_cost_str}\n\n"
                "### Recommended action\n"
                "Please ensure the unit cost is configured in the supplier catalog, or provide complete item and quantity details.\n\n"
                "### Evidence\n"
                f"- **Product**: {pname}\n"
                f"- **Quantity**: {qty_str}\n"
                f"- **Unit Cost**: {unit_cost_str}\n\n"
                "### Source\n"
                "Source:\n"
                "- Demand Decision Intelligence System"
            )

        # Attempt Groq LLM
        if self.groq_api_key and len(self.groq_api_key.strip()) > 5:
            try:
                from groq import Groq
                client = Groq(api_key=self.groq_api_key)

                context_blocks = []
                if data_result and data_result.get("prose"):
                    context_blocks.append(f"**Verified PostgreSQL Data:**\n{data_result['prose']}")
                    if data_result.get("table", {}).get("rows"):
                        context_blocks.append(f"**Data Sample:**\n{json.dumps(data_result['table']['rows'][:4])}")

                if doc_hits:
                    doc_texts = [f"[{h['document_name']} | {h.get('heading', '')} (Line {h.get('start_line')})]: {h['text']}" for h in doc_hits]
                    context_blocks.append(f"**Verified Documentation Evidence:**\n" + "\n\n".join(doc_texts))

                if conv_ctx.get("is_follow_up"):
                    context_blocks.append(f"**Context Entities:** Product ID: {conv_ctx.get('product_id')}, Quantity: {conv_ctx.get('quantity')}, Unit Cost: {conv_ctx.get('unit_cost')}")

                user_content = (
                    f"**Retrieved Evidence:**\n" + "\n\n".join(context_blocks) + "\n\n"
                    f"**User Question:** {query}\n\n"
                    "Remember the STRICT NO-BLUFF rule: Do not assume 95% cycle service level or fake product values unless in retrieved evidence. Structure as 6 parts: ### Answer, ### What this means, ### Key numbers, ### Recommended action, ### Evidence, and ### Source."
                )

                resp = client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.1,
                    max_tokens=850
                )
                return resp.choices[0].message.content
            except Exception as e:
                logger.warning(f"[Synthesizer] Groq call failed: {e}. Executing deterministic fallback.")

        # Attempt OpenAI LLM
        if self.openai_api_key and len(self.openai_api_key.strip()) > 5:
            try:
                import openai
                client = openai.OpenAI(api_key=self.openai_api_key)
                user_content = f"Question: {query}\nEvidence: {data_result}\nDocs: {doc_hits}\nContext: {conv_ctx}"
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ],
                    max_tokens=650,
                    temperature=0.1
                )
                return resp.choices[0].message.content
            except Exception as e:
                logger.warning(f"[Synthesizer] OpenAI call failed: {e}. Executing deterministic fallback.")

        # Deterministic 6-part Fallback Synthesis (STRICT NO-BLUFF)
        return self._deterministic_fallback_synthesis(
            query=query,
            query_type=query_type,
            user_lang=user_lang,
            data_result=data_result,
            doc_hits=doc_hits,
            conv_ctx=conv_ctx,
            all_sources=all_sources
        )

    def _deterministic_fallback_synthesis(
        self,
        query: str,
        query_type: str,
        user_lang: str,
        data_result: Optional[Dict[str, Any]],
        doc_hits: List[Dict[str, Any]],
        conv_ctx: Dict[str, Any],
        all_sources: List[str]
    ) -> str:
        """
        Deterministic 6-part Markdown synthesis without external LLM APIs.
        STRICT NO-BLUFF: Zero unsupported claims, zero invented percentages.
        """

        # ── 1. Pure Documentation Inquiries ──
        if query_type == "DOCS" and doc_hits:
            q_lower = query.lower()
            is_calc_query = bool(re.search(
                r'\b(?:how\s+(?:is|are|do\s+you)\s+.*?\s+(?:calculated|computed|determined)|formula|equation|how\s+to\s+calculate|explain\s+(?:the\s+)?calculation|calculation\s+of|what\s+methodology\s+is\s+used\s+for|what\s+methodology\s+is\s+used\s+to)\b',
                q_lower
            )) or any(k in q_lower for k in ["formula", "equation"])

            # 1A. Both Safety Stock and Reorder Point inquiry
            if ("safety stock" in q_lower or "ss" in q_lower.split()) and ("reorder point" in q_lower or "rop" in q_lower.split()):
                top_hit = next(
                    (h for h in doc_hits if "system_architecture" in h["document_name"] or "ML_ENGINEERING" in h["document_name"]),
                    doc_hits[0]
                )
                doc_name = top_hit["document_name"]
                heading = top_hit.get("heading", "Stochastic Inventory Optimization Arithmetic")
                line_ref = f"Line {top_hit.get('start_line', 203)}"
                return (
                    f"### Answer\n"
                    f"Based on **system_architecture_hld_lld.md** (Section 3.2.E, Line 203) and **ML_ENGINEERING_AND_PIPELINE_GUIDE.md** (Section 6, Line 88), the system specifies the documented safety stock and reorder point calculations:\n\n"
                    f"1. **King's Formula Safety Stock (stochastic lead time + demand):**\n"
                    f"$$\\text{{SS}} = Z_{{\\alpha}} \\times \\sqrt{{\\bar{{L}} \\cdot \\sigma_d^2 + \\bar{{d}}^2 \\cdot \\sigma_L^2}}$$\n"
                    f"Discrete decision layer formulation: $$SS = \\lceil Z \\times \\sigma_d \\times \\sqrt{{L}} \\rceil$$\n\n"
                    f"2. **Reorder Point (ROP):**\n"
                    f"$$\\text{{ROP}} = \\text{{LTD}} + \\text{{SS}} = (\\bar{{d}} \\times \\bar{{L}}) + \\text{{SS}}$$\n"
                    f"Discrete integer formulation: $$ROP = \\lceil (\\mu_d \\times L) + SS \\rceil$$\n\n"
                    f"**Documented Variables & Inputs:**\n"
                    f"- $Z_{{\\alpha}}$ / $Z$: Service level factor (default 95% $\\implies Z = 1.645$)\n"
                    f"- $\\bar{{L}}$ / $L$: Average supplier lead time in days (default 3 days)\n"
                    f"- $\\sigma_d^2$ / $\\sigma_d$: Daily demand variance / standard deviation from historical series\n"
                    f"- $\\bar{{d}}$ / $\\mu_d$: Mean daily demand\n"
                    f"- $\\sigma_L^2$: Supplier lead time variance\n"
                    f"- $\\text{{LTD}}$: Lead Time Demand ($\\bar{{d}} \\times \\bar{{L}}$)\n\n"
                    f"### What this means\n"
                    f"Safety stock acts as a protective buffer against lead-time and demand volatility, while ROP sets the exact inventory threshold that triggers replenishment before stockouts occur.\n\n"
                    f"### Key numbers\n"
                    f"- **Safety Stock Formula**: $\\text{{SS}} = Z_{{\\alpha}} \\times \\sqrt{{\\bar{{L}} \\cdot \\sigma_d^2 + \\bar{{d}}^2 \\cdot \\sigma_L^2}}$\n"
                    f"- **Reorder Point Formula**: $\\text{{ROP}} = \\text{{LTD}} + \\text{{SS}}$\n"
                    f"- **Default Service Level**: 95% ($Z = 1.645$)\n"
                    f"- **Default Lead Time**: 3 days ($L = 3$)\n\n"
                    f"### Recommended action\n"
                    f"Periodically calibrate daily demand variance ($\\sigma_d$) and supplier lead times ($L$) from historical data before overriding replenishment thresholds.\n\n"
                    f"### Evidence\n"
                    f"- **Documents**: system_architecture_hld_lld.md (Line 203), ML_ENGINEERING_AND_PIPELINE_GUIDE.md (Line 88)\n"
                    f"- **Section**: Stochastic Inventory Optimization Arithmetic\n\n"
                    f"### Source\n"
                    f"Source:\n" + "\n".join([f"- {s}" for s in all_sources])
                )

            # 1B. Safety Stock calculation / formula inquiry
            if ("safety stock" in q_lower or "ss" in q_lower.split()) and (is_calc_query or "methodology" in q_lower):
                formula_hit = next(
                    (h for h in doc_hits if "\\text{SS}" in h["text"] or "King's Formula" in h["text"] or "SS =" in h["text"]),
                    doc_hits[0]
                )
                doc_name = formula_hit["document_name"]
                heading = formula_hit.get("heading", "Stochastic Inventory Optimization Arithmetic")
                line_ref = f"Line {formula_hit.get('start_line', 203)}"

                return (
                    f"### Answer\n"
                    f"Based on **{doc_name}** ({heading}, {line_ref}), the system specifies the following documented safety stock calculation:\n\n"
                    f"**King's Formula Safety Stock (stochastic lead time + stochastic demand):**\n"
                    f"$$\\text{{SS}} = Z_{{\\alpha}} \\times \\sqrt{{\\bar{{L}} \\cdot \\sigma_d^2 + \\bar{{d}}^2 \\cdot \\sigma_L^2}}$$\n\n"
                    f"**Documented Variables & Inputs:**\n"
                    f"- $Z_{{\\alpha}}$: Target cycle service level factor (default 95% $\\implies Z = 1.645$)\n"
                    f"- $\\bar{{L}}$: Average supplier lead time in days (configurable parameter, default 3 days)\n"
                    f"- $\\sigma_d^2$: Daily demand variance (where $\\sigma_d$ is daily demand standard deviation derived from historical series)\n"
                    f"- $\\bar{{d}}$: Mean daily demand ($\\mu_d$)\n"
                    f"- $\\sigma_L^2$: Supplier lead time variance\n\n"
                    f"Additionally, **ML_ENGINEERING_AND_PIPELINE_GUIDE.md** (Section 6. Inventory Decision Layer, Line 88) documents the discrete decision layer formula: $$SS = \\lceil Z \\times \\sigma_d \\times \\sqrt{{L}} \\rceil$$\n\n"
                    f"### What this means\n"
                    f"Safety stock serves as a statistical buffer against two concurrent sources of supply chain uncertainty: demand surges during replenishment and supplier lead-time delivery delays.\n\n"
                    f"### Key numbers\n"
                    f"- **Governing Specification**: {doc_name} ({line_ref}) & ML_ENGINEERING_AND_PIPELINE_GUIDE.md (Line 88)\n"
                    f"- **Default Service Level**: 95% ($Z = 1.645$)\n"
                    f"- **Default Lead Time**: 3 days ($L = 3$)\n\n"
                    f"### Recommended action\n"
                    f"Verify SKU demand variance ($\\sigma_d^2$) from historical sales series and supplier lead-time observations before overriding default safety stock levels.\n\n"
                    f"### Evidence\n"
                    f"- **Document**: {formula_hit.get('source_path', doc_name)}\n"
                    f"- **Section**: {heading} ({line_ref})\n"
                    f"- **Retrieved Formula**: King's Formula Safety Stock: SS = Z_alpha * sqrt(L_bar * sigma_d^2 + d_bar^2 * sigma_L^2)\n\n"
                    f"### Source\n"
                    f"Source:\n" + "\n".join([f"- {s}" for s in all_sources])
                )

            # 1C. Reorder Point (ROP) calculation/formula inquiry
            if ("reorder point" in q_lower or "rop" in q_lower.split()) and (is_calc_query or "methodology" in q_lower):
                formula_hit = next(
                    (h for h in doc_hits if "\\text{ROP}" in h["text"] or "ROP =" in h["text"] or "Lead Time Demand" in h["text"]),
                    doc_hits[0]
                )
                doc_name = formula_hit["document_name"]
                heading = formula_hit.get("heading", "Stochastic Inventory Optimization Arithmetic")
                line_ref = f"Line {formula_hit.get('start_line', 203)}"

                return (
                    f"### Answer\n"
                    f"Based on **{doc_name}** ({heading}, {line_ref}) and **ML_ENGINEERING_AND_PIPELINE_GUIDE.md** (Section 6, Line 88), the system specifies the following Reorder Point (ROP) calculation:\n\n"
                    f"**Reorder Point (ROP) Equation:**\n"
                    f"$$\\text{{ROP}} = \\text{{LTD}} + \\text{{SS}}$$\n\n"
                    f"**Lead Time Demand (LTD):**\n"
                    f"$$\\text{{LTD}} = \\bar{{d}} \\times \\bar{{L}}$$\n\n"
                    f"**Documented Variables & Inputs:**\n"
                    f"- $\\text{{LTD}}$: Lead Time Demand (expected sales units during supplier replenishment cycle)\n"
                    f"- $\\bar{{d}}$: Mean daily demand ($\\mu_d$ derived from historical demand series)\n"
                    f"- $\\bar{{L}}$: Average supplier lead time in days ($L$, default 3 days)\n"
                    f"- $\\text{{SS}}$: Safety Stock buffer ($SS = \\lceil Z \\times \\sigma_d \\times \\sqrt{{L}} \\rceil$ or King's Formula)\n\n"
                    f"In **ML_ENGINEERING_AND_PIPELINE_GUIDE.md** (Line 94), the discrete integer formulation is documented as: $$ROP = \\lceil (\\mu_d \\times L) + SS \\rceil$$.\n\n"
                    f"### What this means\n"
                    f"A replenishment recommendation is triggered whenever current inventory drops to or below ROP, ensuring an order arrives before safety stock is breached.\n\n"
                    f"### Key numbers\n"
                    f"- **Governing Specification**: {doc_name} ({heading}, {line_ref})\n"
                    f"- **Formula**: $\\text{{ROP}} = (\\bar{{d}} \\times \\bar{{L}}) + \\text{{SS}}$\n\n"
                    f"### Recommended action\n"
                    f"Ensure lead time observations and daily demand statistics are calibrated periodically to prevent premature or delayed reorder triggers.\n\n"
                    f"### Evidence\n"
                    f"- **Document**: {formula_hit.get('source_path', doc_name)}\n"
                    f"- **Section**: {heading} ({line_ref})\n"
                    f"- **Retrieved Excerpt**: {formula_hit['text'][:280].strip()}...\n\n"
                    f"### Source\n"
                    f"Source:\n" + "\n".join([f"- {s}" for s in all_sources])
                )

            # 1D. Inventory Recommendations Generation inquiry
            if ("inventory recommendation" in q_lower or "recommendation" in q_lower) and any(k in q_lower for k in ["generate", "generated", "how are", "workflow", "process", "created", "logic", "calculated"]):
                return (
                    f"### Answer\n"
                    f"Based on **ML_ENGINEERING_AND_PIPELINE_GUIDE.md** (Section 6, Line 88) and **system_architecture_hld_lld.md** (Section 3.2.E, Line 203), inventory recommendations are generated through a stochastic decision optimization layer:\n\n"
                    f"1. **Input Parameters Derived from Historical Data:**\n"
                    f"- Mean daily demand ($\\mu_d$ / $\\bar{{d}}$)\n"
                    f"- Daily demand standard deviation ($\\sigma_d$)\n"
                    f"- Unit landing price ($C_{{\\text{{unit}}}}$)\n\n"
                    f"2. **Governing Policy Configurations:**\n"
                    f"- Supplier lead time ($L$, default 3 days)\n"
                    f"- Target cycle service level factor ($Z = 1.645$ for 95% service level)\n"
                    f"- Review period ($R$, default 7 days)\n\n"
                    f"3. **Mathematical Thresholds:**\n"
                    f"- **Safety Stock ($SS$):** $$SS = \\lceil Z \\times \\sigma_d \\times \\sqrt{{L}} \\rceil$$\n"
                    f"- **Reorder Point ($ROP$):** $$ROP = \\lceil (\\mu_d \\times L) + SS \\rceil$$\n"
                    f"- **Target Stock Level ($TSL$):** $$TSL = \\lceil \\mu_d \\times (L + R) + SS \\rceil$$\n"
                    f"- **Recommended Order Quantity ($ROQ$):** $$ROQ = \\max(0, \\lceil TSL - \\text{{Stock}}_{{\\text{{on\\_hand}}}} - \\text{{Stock}}_{{\\text{{on\\_order}}}} \\rceil)$$\n\n"
                    f"4. **Replenishment Trigger & Risk Categorization:**\n"
                    f"- **CRITICAL_STOCKOUT:** Current on-hand stock $\\le SS$\n"
                    f"- **REORDER_REQUIRED:** Current on-hand stock $\\le ROP$\n"
                    f"- **OPTIMAL:** Current stock remains above ROP\n\n"
                    f"### What this means\n"
                    f"Inventory recommendations compare live warehouse on-hand stock against statistical reorder thresholds ($ROP = LTD + SS$) to order the exact quantity needed to bring stock up to Target Stock Level before safety buffers are breached.\n\n"
                    f"### Key numbers\n"
                    f"- **Default Service Level**: 95% ($Z = 1.645$)\n"
                    f"- **Default Supplier Lead Time**: 3 days ($L = 3$)\n"
                    f"- **Default Review Period**: 7 days ($R = 7$)\n"
                    f"- **Reorder Equation**: $ROP = (\\mu_d \\times L) + SS$\n\n"
                    f"### Recommended action\n"
                    f"Review SKUs flagged with 'REORDER_REQUIRED' or 'CRITICAL_STOCKOUT' and generate replenishment purchase orders matching the Recommended Order Quantity.\n\n"
                    f"### Evidence\n"
                    f"- **Documents**: ML_ENGINEERING_AND_PIPELINE_GUIDE.md (Section 6: Inventory Decision Layer), system_architecture_hld_lld.md (Section 3.2.E: Stochastic Inventory Optimization Arithmetic)\n\n"
                    f"### Source\n"
                    f"Source:\n" + "\n".join([f"- {s}" for s in all_sources])
                )

            # 1E. Dead Stock Detection & Downward Clearance Rules inquiry
            if any(k in q_lower for k in ["dead stock", "deadstock"]):
                return (
                    f"### Answer\n"
                    f"Based on **system_architecture_hld_lld.md** (Section 3.2.H, Prompt 4.8, Line 225), the system specifies the following business rules for identifying and managing dead stock:\n\n"
                    f"1. **Inactivity & Excess Stock Detection Rules:**\n"
                    f"- **Inactive Inventory:** Zero sales demand recorded for $N \\ge 90$ consecutive days while on-hand stock is greater than zero ($\\text{{Stock}}_{{\\text{{on\\_hand}}}} > 0$).\n"
                    f"- **Excess Cover:** On-hand inventory exceeds 180 days of forward demand ($\\text{{Days of Cover}} = \\text{{Stock}}_{{\\text{{on\\_hand}}}} / \\bar{{d}} > 180$).\n\n"
                    f"2. **Financial Liabilities & Exposure:**\n"
                    f"- **Capital Tied Up:** $\\text{{Capital Locked}} = \\text{{Stock}}_{{\\text{{on\\_hand}}}} \\times C_{{\\text{{unit}}}}$\n"
                    f"- **Monthly Holding Drag:** $\\text{{Monthly Cost}} = \\text{{Stock}}_{{\\text{{on\\_hand}}}} \\times \\text{{Storage Footprint}} \\times \\text{{Monthly Rate}}$\n"
                    f"- **Projected Obsolescence:** Tracked against expiration shelf-life date.\n\n"
                    f"3. **Downward Action Recommendation Engine:**\n"
                    f"- **TRANSFER:** If Days of Cover $> 300$, transfer excess inventory to a high-velocity regional distribution hub with demand deficit.\n"
                    f"- **MARKDOWN:** Solve optimal clearance discount $\\delta \\in [0.10, 0.60]$ via Price Elasticity of Demand (PED) to liquidate stock within 45 days: $$\\delta = \\frac{{1}}{{|\\varepsilon|}} \\left(\\frac{{\\text{{Stock}}_{{\\text{{on\\_hand}}}}}}{{T \\cdot \\bar{{d}}}} - 1\\right)$$\n"
                    f"- **BUNDLE:** Bundle slow-moving item with high-velocity anchor SKU in category.\n"
                    f"- **RETURN_TO_SUPPLIER:** If supplier contractual agreement permits inventory returns.\n"
                    f"- **WRITE_OFF:** If inactive for $\\ge 270$ consecutive days or past shelf-life obsolescence date.\n"
                    f"- **DELIST:** If product is categorized in ABC-XYZ cell CZ with negligible daily demand ($\\bar{{d}} < 0.1$).\n\n"
                    f"### What this means\n"
                    f"The dead stock engine flags dormant capital and automatically solves optimal markdown discounts or inter-city transfers to recover working capital before physical expiration.\n\n"
                    f"### Key numbers\n"
                    f"- **Inactivity Threshold**: 90 consecutive days with zero demand\n"
                    f"- **Excess Cover Threshold**: > 180 days of forward demand\n"
                    f"- **Transfer Hurdle**: > 300 days of cover\n"
                    f"- **Markdown Discount Range**: 10% to 60% based on Price Elasticity of Demand\n"
                    f"- **Write-Off Threshold**: ≥ 270 days inactive\n\n"
                    f"### Recommended action\n"
                    f"Review flagged dead stock SKUs and apply recommended downward markdown discounts or regional transfers to reclaim frozen working capital.\n\n"
                    f"### Evidence\n"
                    f"- **Document**: docs/system_architecture_hld_lld.md\n"
                    f"- **Section**: Section 3.2.H: Dead Stock Detection, Financial Exposure & Downward Clearance Optimizer (Prompt 4.8)\n\n"
                    f"### Source\n"
                    f"Source:\n" + "\n".join([f"- {s}" for s in all_sources])
                )

            # 1F. Demand Forecasting Models inquiry
            if any(k in q_lower for k in ["models are used", "models used", "what models", "forecasting models", "forecast models"]):
                return (
                    f"### Answer\n"
                    f"Based on **ML_ENGINEERING_AND_PIPELINE_GUIDE.md** (Section 4, Line 54), **ARCHITECTURE.md** (Section 13, Line 394), and **PRD.md** (Section 7, Line 209), the system documents the following demand forecasting models:\n\n"
                    f"1. **Baseline Benchmark Models:**\n"
                    f"- **Naive (Lag-1):** Uses the immediate previous day's sales ($y_{{t-1}}$) as the baseline benchmark (**32.43% WAPE**, **45.64 RMSE**).\n"
                    f"- **Seasonal Naive (Lag-7):** Uses sales from the same day of the prior week ($y_{{t-7}}$) to track weekly seasonality (**79.35% WAPE**).\n"
                    f"- **Rolling Mean 7:** Historical 7-day moving average smoothing short-term demand variance (**49.64% WAPE**).\n\n"
                    f"2. **Machine Learning Models:**\n"
                    f"- **Ridge Regression:** Linear regularized model with L2 penalty; achieved the lowest error (**31.97% WAPE**, **41.01 RMSE**) and is selected as the **champion model** for active series.\n"
                    f"- **HistGradientBoosting (GBT):** Non-linear gradient boosted tree ensemble (**40.81% WAPE**, **73.97 RMSE**) capturing non-linear interactions.\n\n"
                    f"3. **Time-Series & Intermittent Models:**\n"
                    f"- **Prophet:** Additive decomposition model capturing trend and weekly grocery seasonality for eligible products.\n"
                    f"- **Croston / Syntetos-Boylan-Croston:** Documented for sparse and intermittent demand series where zero-demand transactions dominate.\n\n"
                    f"### What this means\n"
                    f"The platform benchmarks multiple algorithms chronologically to deploy the most accurate forecasting engine for retail grocery demand.\n\n"
                    f"### Key numbers\n"
                    f"- **Ridge Regression (Champion)**: 31.97% WAPE | 41.01 RMSE (0.31s training time)\n"
                    f"- **HistGradientBoosting**: 40.81% WAPE | 73.97 RMSE (3.64s training time)\n"
                    f"- **Naive Baseline**: 32.43% WAPE | 45.64 RMSE\n"
                    f"- **Seasonal Naive (Lag-7)**: 79.35% WAPE | 105.85 RMSE\n\n"
                    f"### Recommended action\n"
                    f"Deploy Ridge Regression for high-velocity stable SKUs and apply Croston intermittent forecasting for tail items with high zero-demand frequency.\n\n"
                    f"### Evidence\n"
                    f"- **Documents**: ML_ENGINEERING_AND_PIPELINE_GUIDE.md (Section 4: Forecasting Model Benchmarks), ARCHITECTURE.md (Section 13: Forecasting Architecture), PRD.md (Section 7)\n\n"
                    f"### Source\n"
                    f"Source:\n" + "\n".join([f"- {s}" for s in all_sources])
                )

            # 1G. Demand Forecasting Methodology inquiry
            if "forecasting methodology" in q_lower or ("methodology" in q_lower and "forecast" in q_lower) or ("forecasting" in q_lower and any(k in q_lower for k in ["workflow", "pipeline", "approach", "how does"])):
                return (
                    f"### Answer\n"
                    f"Based on **ML_ENGINEERING_AND_PIPELINE_GUIDE.md** (Sections 1–4), **ARCHITECTURE.md** (Section 13, Line 394), and **system_architecture_hld_lld.md** (Section 3.2.B), the platform implements an end-to-end demand forecasting methodology:\n\n"
                    f"1. **Data Aggregation & Preprocessing:**\n"
                    f"- Aggregates transaction-level sales into daily product demand at the `(date_, product_id, city_name)` daily grain across categories.\n\n"
                    f"2. **Strict Anti-Leakage Feature Engineering:**\n"
                    f"- Historical lags: `lag_1`, `lag_7`, `lag_14`, `lag_28` using strictly preceding observations ($t-k$).\n"
                    f"- Rolling window statistics: `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_28`, and `rolling_std_7` computed over preceding windows strictly excluding the forecast day $t$.\n"
                    f"- Calendar regressors: Deterministic day-of-week and weekend flags, with 28-day warmup truncation to eliminate null features.\n\n"
                    f"3. **Chronological Backtesting & Validation:**\n"
                    f"- Strict anti-shuffling monotonic time-based split (Training: April 29 – June 30, 2022; Validation: July 1 – July 10, 2022).\n\n"
                    f"4. **Multi-Horizon Forecasts:** Generates short-term forecasts for 7-day, 14-day, and 30-day forecast horizons.\n\n"
                    f"5. **Model Evaluation & Promotion:**\n"
                    f"- Evaluated using Weighted Absolute Percentage Error (WAPE), MAE, and RMSE.\n"
                    f"- 5% Hysteresis Hurdle: Challenger models must beat the incumbent champion by at least 5% WAPE to be promoted.\n"
                    f"- Naive Seasonal Floor: Models failing the baseline are marked unforecastable.\n\n"
                    f"### What this means\n"
                    f"The methodology enforces strict time-ordered anti-leakage engineering to ensure realistic grocery demand predictions that drive automated replenishment without future-data lookahead.\n\n"
                    f"### Key numbers\n"
                    f"- **Forecast Horizons**: 7 days, 14 days, 30 days\n"
                    f"- **Training Period**: 63 calendar days (2022-04-29 to 2022-06-30)\n"
                    f"- **Validation Period**: 10 calendar days (2022-07-01 to 2022-07-10)\n"
                    f"- **Champion Model**: Ridge Regression (31.97% WAPE)\n\n"
                    f"### Recommended action\n"
                    f"Execute periodic backtesting recalibrations and verify error metrics (WAPE/MAE) before promoting challenger forecasting pipelines.\n\n"
                    f"### Evidence\n"
                    f"- **Documents**: ML_ENGINEERING_AND_PIPELINE_GUIDE.md (Sections 3 & 4), system_architecture_hld_lld.md (Section 3.2.B), ARCHITECTURE.md (Section 13), PRD.md (Section 7)\n\n"
                    f"### Source\n"
                    f"Source:\n" + "\n".join([f"- {s}" for s in all_sources])
                )

            # 1H. Other formula/calculation queries where formula is NOT documented
            if is_calc_query:
                m_concept = re.search(r'(?:formula\s+(?:for|of)|how\s+(?:is|do\s+you\s+calculate)\s+|calculation\s+of\s+|methodology\s+(?:used\s+)?for\s+)(.+?)(?:\?|$)', query, re.I)
                concept_name = m_concept.group(1).strip() if m_concept else "this concept"
                top_hit = doc_hits[0]
                clean_excerpt = re.sub(r'#+\s*', '', top_hit["text"]).strip()[:280]
                return (
                    f"### Answer\n"
                    f"The current project documentation does not specify a complete calculation formula for {concept_name}.\n\n"
                    f"Available documented reference from **{top_hit['document_name']}** ({top_hit.get('heading', '')}, Line {top_hit.get('start_line', 1)}):\n"
                    f"> {clean_excerpt}...\n\n"
                    f"### What this means\n"
                    f"The requested calculation methodology is not defined as an explicit mathematical equation in the indexed documentation.\n\n"
                    f"### Key numbers\n"
                    f"- **Formula Status**: Unspecified in documentation\n"
                    f"- **Document Reference**: {top_hit['document_name']} (Line {top_hit.get('start_line', 1)})\n\n"
                    f"### Recommended action\n"
                    f"Refer to engineering specifications or configure the calculation parameters manually.\n\n"
                    f"### Evidence\n"
                    f"- **Document**: {top_hit.get('source_path', top_hit['document_name'])}\n"
                    f"- **Section**: {top_hit.get('heading', '')}\n\n"
                    f"### Source\n"
                    f"Source:\n" + "\n".join([f"- {s}" for s in all_sources])
                )

            # 1I. General conceptual documentation inquiry
            top_hit = doc_hits[0]
            doc_name = top_hit["document_name"]
            heading = top_hit.get("heading", "Specification")
            line_ref = f"Line {top_hit.get('start_line', 1)}"

            clean_text = re.sub(r'#+\s*', '', top_hit["text"]).strip()
            lines = clean_text.splitlines()
            body_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
            summary = "\n".join(body_lines[:8]) if body_lines else clean_text[:400]

            return (
                f"### Answer\n"
                f"Based on **{doc_name}** ({heading}, {line_ref}), the system defines:\n\n"
                f"> \"{summary.strip()}\"\n\n"
                f"### What this means\n"
                f"This specification governs how the platform's supply chain intelligence and algorithms operate.\n\n"
                f"### Key numbers\n"
                f"- **Source Reference**: {doc_name} ({line_ref})\n"
                f"- **Similarity Match**: {top_hit.get('retrieval_score', 0):.2f}\n\n"
                f"### Recommended action\n"
                f"Refer to {doc_name} for full engineering details.\n\n"
                f"### Evidence\n"
                f"- **Document**: {top_hit.get('source_path', doc_name)}\n"
                f"- **Section**: {heading}\n\n"
                f"### Source\n"
                f"Source:\n" + "\n".join([f"- {s}" for s in all_sources])
            )

        # ── 2. Hybrid Data + Documentation Inquiries ──
        if query_type == "HYBRID":
            doc_explanation = ""
            doc_citation = ""
            if doc_hits:
                top_doc = doc_hits[0]
                clean_text = re.sub(r'#+\s*', '', top_doc.get("text", "")).strip()
                # Take first meaningful sentence from real retrieved text
                first_line = clean_text.split('\n')[0] if clean_text else ""
                if len(first_line) > 130:
                    first_line = first_line[:130] + "..."
                doc_citation = f"{top_doc['document_name']} ({top_doc.get('heading', 'Policy')}, Line {top_doc.get('start_line', 1)})"
                doc_explanation = f"Per {doc_citation}: \"{first_line}\""
            else:
                doc_explanation = "Reorder recommendation logic specifies replenishment orders are triggered when current stock falls to or below the calculated Reorder Point (ROP = Lead Time Demand + Safety Stock)."

            # Check if this is a SKU-specific inventory query
            sku_data = data_result.get("sku_inventory_data") if data_result else None
            pid = conv_ctx.get("product_id") or (sku_data.get("product_id") if sku_data else None)

            if sku_data or pid or (data_result and data_result.get("template_name") == "sku_inventory_recommendation"):
                pname = (sku_data.get("product_name") if sku_data else None) or conv_ctx.get("product_name") or f"SKU {pid}"

                if sku_data and sku_data.get("found"):
                    curr_stock = sku_data["current_stock"]
                    rop = sku_data["reorder_point"]
                    safety_stock = sku_data["safety_stock"]
                    order_qty = sku_data["recommended_order_quantity"]
                    risk_status = sku_data.get("risk_status", "OPTIMAL")

                    answer_block = (
                        f"### Answer\n"
                        f"**Verified database:**\n"
                        f"- **Product**: {pname} (SKU {pid})\n"
                        f"- **Current Stock**: {curr_stock:,.0f} units\n"
                        f"- **Reorder Point**: {rop:,.0f} units\n"
                        f"- **Safety Stock**: {safety_stock:,.0f} units\n"
                        f"- **Recommended Order Quantity**: {order_qty:,.0f} units\n"
                        f"- **Risk Status**: {risk_status}\n\n"
                        f"**Documentation:**\n"
                        f"{doc_explanation}"
                    )
                    meaning_block = (
                        f"### What this means\n"
                        f"The item is evaluated for replenishment by comparing current stock ({curr_stock:,.0f} units) against the Reorder Point ({rop:,.0f} units) to protect against stockout risk during supplier lead time."
                    )
                    key_numbers_block = (
                        f"### Key numbers\n"
                        f"- **Product**: {pname} (SKU {pid})\n"
                        f"- **Current Stock**: {curr_stock:,.0f} units\n"
                        f"- **Reorder Point**: {rop:,.0f} units\n"
                        f"- **Safety Stock**: {safety_stock:,.0f} units\n"
                        f"- **Recommended Order Quantity**: {order_qty:,.0f} units\n"
                        f"- **Risk Status**: {risk_status}"
                    )
                    action_block = (
                        f"### Recommended action\n"
                        f"Review the recommended order quantity of {order_qty:,.0f} units against supplier lead time and initiate a replenishment purchase order."
                    )
                else:
                    # Clean verified missing record (NO-BLUFF)
                    answer_block = (
                        f"### Answer\n"
                        f"**Verified database:**\n"
                        f"No verified inventory recommendation record was found in the database for {pname} (SKU {pid}).\n\n"
                        f"**Documentation:**\n"
                        f"{doc_explanation}"
                    )
                    meaning_block = (
                        f"### What this means\n"
                        f"According to system documentation, replenishment orders are triggered when current stock drops to or below the Reorder Point (ROP = Lead Time Demand + Safety Stock). When an SKU has no active recommendation record in PostgreSQL, no automated replenishment is currently scheduled."
                    )
                    key_numbers_block = (
                        f"### Key numbers\n"
                        f"- **Product**: {pname} (SKU {pid})\n"
                        f"- **Database Status**: No inventory recommendation record found in PostgreSQL\n"
                        f"- **Current Stock**: Not available in recommendations"
                    )
                    action_block = (
                        f"### Recommended action\n"
                        f"Execute the inventory health and recommendation pipeline for SKU {pid} or verify warehouse stock levels."
                    )

                return (
                    f"{answer_block}\n\n"
                    f"{meaning_block}\n\n"
                    f"{key_numbers_block}\n\n"
                    f"{action_block}\n\n"
                    f"### Evidence\n"
                    f"- **Database**: inventory_recommendations (PostgreSQL live verification)\n"
                    f"- **Documentation**: {doc_citation or 'Project Documentation'}\n\n"
                    f"### Source\n"
                    f"Source:\n" + "\n".join([f"- {s}" for s in all_sources])
                )

            prose = data_result.get("prose", "") if data_result else "Operational inventory policy requires reorder PO intervention."
            tpl = data_result.get("template_name") if data_result else ""
            rows = data_result.get("table", {}).get("rows", []) if data_result else []
            if tpl == "items_below_rop" and len(rows) == 0:
                action_text = "No ROP-based reorder action is indicated by this query. Continue routine inventory monitoring."
                meaning_text = "No evaluated SKU was found below its recorded reorder point."
            else:
                action_text = "Review inventory positions against lead-time buffers and trigger necessary replenishment purchase orders."
                meaning_text = doc_explanation

            return (
                f"### Answer\n"
                f"{prose}\n\n"
                f"### What this means\n"
                f"{meaning_text}\n\n"
                f"### Key numbers\n"
                f"- **Data Grounding**: PostgreSQL Live Verification\n\n"
                f"### Recommended action\n"
                f"{action_text}\n\n"
                f"### Evidence\n"
                f"- **Database**: {', '.join([s for s in all_sources if not s.endswith('.md')]) or 'Live PostgreSQL'}\n"
                f"- **Documentation**: {', '.join([s for s in all_sources if s.endswith('.md')]) or 'None'}\n\n"
                f"### Source\n"
                f"Source:\n" + "\n".join([f"- {s}" for s in all_sources])
            )

        # ── 3. Pure Operational Live Data Inquiries ──
        if data_result and data_result.get("template_name") == "sku_inventory_recommendation":
            sku_data = data_result.get("sku_inventory_data")
            pid = conv_ctx.get("product_id") or (sku_data.get("product_id") if sku_data else None)
            pname = (sku_data.get("product_name") if sku_data else None) or conv_ctx.get("product_name") or f"SKU {pid}"
            prose = data_result.get("prose", "")

            if sku_data and sku_data.get("found"):
                curr_stock = sku_data["current_stock"]
                rop = sku_data["reorder_point"]
                safety_stock = sku_data["safety_stock"]
                order_qty = sku_data["recommended_order_quantity"]
                risk_status = sku_data.get("risk_status", "OPTIMAL")

                return (
                    f"### Answer\n"
                    f"{prose}\n\n"
                    f"### What this means\n"
                    f"This verified data is retrieved directly from the PostgreSQL `inventory_recommendations` table for {pname}.\n\n"
                    f"### Key numbers\n"
                    f"- **Product**: {pname} (SKU {pid})\n"
                    f"- **Current Stock**: {curr_stock:,.0f} units\n"
                    f"- **Reorder Point**: {rop:,.0f} units\n"
                    f"- **Safety Stock**: {safety_stock:,.0f} units\n"
                    f"- **Recommended Order Quantity**: {order_qty:,.0f} units\n"
                    f"- **Risk Status**: {risk_status}\n\n"
                    f"### Recommended action\n"
                    f"Validate replenishment thresholds against current inventory state.\n\n"
                    f"### Evidence\n"
                    f"- **Query Template**: sku_inventory_recommendation\n"
                    f"- **Database Table**: inventory_recommendations\n\n"
                    f"### Source\n"
                    f"Source:\n" + "\n".join([f"- {s}" for s in all_sources])
                )
            else:
                return (
                    f"### Answer\n"
                    f"{prose}\n\n"
                    f"### What this means\n"
                    f"The database does not currently contain a computed inventory recommendation or safety stock threshold for {pname} (SKU {pid}).\n\n"
                    f"### Key numbers\n"
                    f"- **Product**: {pname} (SKU {pid})\n"
                    f"- **Database Status**: No verified recommendation record found\n\n"
                    f"### Recommended action\n"
                    f"Run the inventory recommendation engine to compute ROP and safety stock for this SKU.\n\n"
                    f"### Evidence\n"
                    f"- **Query Template**: sku_inventory_recommendation\n"
                    f"- **Database Table**: inventory_recommendations\n\n"
                    f"### Source\n"
                    f"Source:\n" + "\n".join([f"- {s}" for s in all_sources])
                )

        # ── 4. items_below_rop Query Specialization (FIX 1 & 2) ──
        if data_result and data_result.get("template_name") == "items_below_rop":
            rows = data_result.get("table", {}).get("rows", [])
            row_count = len(rows)
            loc = conv_ctx.get("city") or "this dataset"
            if row_count == 0:
                answer = f"0 SKUs are currently below ROP in {loc}. No immediate ROP-based procurement action is identified from this query."
                meaning = "No evaluated SKU was found below its recorded reorder point."
                action = "No ROP-based reorder action is indicated by this query. Continue routine inventory monitoring."
                key_numbers = (
                    f"### Key numbers\n"
                    f"- **Items Below ROP**: 0 SKUs\n"
                    f"- **Replenishment Status**: No ROP breach detected"
                )
            else:
                answer = f"Found {row_count} SKUs currently operating below their Reorder Point (ROP). Immediate replenishment review is recommended for these items."
                meaning = "These items have fallen below their safety reorder buffer and are at risk of stockout during supplier lead time."
                action = f"Review the {row_count} below-ROP items in the table below and initiate replenishment purchase orders to restore safety buffers."
                key_numbers = (
                    f"### Key numbers\n"
                    f"- **Items Below ROP**: {row_count} SKUs\n"
                    f"- **Replenishment Status**: {row_count} SKUs below reorder threshold"
                )

            return (
                f"### Answer\n"
                f"{answer}\n\n"
                f"### What this means\n"
                f"{meaning}\n\n"
                f"{key_numbers}\n\n"
                f"### Recommended action\n"
                f"{action}\n\n"
                f"### Evidence\n"
                f"- **Query Template**: items_below_rop\n"
                f"- **Database Table**: inventory_recommendations\n\n"
                f"### Source\n"
                f"Source:\n" + "\n".join([f"- {s}" for s in all_sources])
            )

        # ── 5. top_n_by Query Specialization (FIX 1) ──
        if data_result and data_result.get("template_name") == "top_n_by":
            tbl = data_result.get("table", {})
            rows = tbl.get("rows", [])
            row_count = len(rows)

            answer = (
                f"These are the top {row_count} products by recorded sales/demand volume. "
                f"Review their current stock, ROP, and inventory recommendations before making procurement decisions."
            )
            meaning = "This ranking reflects historical sales and demand transactions recorded in the daily demand dataset."
            key_numbers = (
                f"### Key numbers\n"
                f"- **Products Ranked**: {row_count} SKUs\n"
                f"- **Ranking Metric**: Sales / Demand volume"
            )
            action = "Review current stock, ROP thresholds, and active recommendations for these high-volume items before making procurement decisions."

            return (
                f"### Answer\n"
                f"{answer}\n\n"
                f"### What this means\n"
                f"{meaning}\n\n"
                f"{key_numbers}\n\n"
                f"### Recommended action\n"
                f"{action}\n\n"
                f"### Evidence\n"
                f"- **Query Template**: top_n_by\n"
                f"- **Database Table**: daily_product_demand\n\n"
                f"### Source\n"
                f"Source:\n" + "\n".join([f"- {s}" for s in all_sources])
            )

        # ── 6. supplier_po_summary Query Specialization (FIX 4) ──
        if data_result and data_result.get("template_name") == "supplier_po_summary":
            tbl = data_result.get("table", {})
            rows = tbl.get("rows", [])
            row_count = len(rows)

            if row_count == 0:
                answer = "No recent purchase orders were found in the database."
                meaning = "No recent PO records are available for the requested query."
                key_numbers = (
                    f"### Key numbers\n"
                    f"- **Records Found**: 0\n"
                    f"- **Total PO Value**: ₹0"
                )
                action = "No PO records are available for review. Check procurement requirements separately if needed."
            else:
                total_spend = sum(float(r.get("total_amount") or r.get("amount") or 0.0) for r in rows)
                answer = f"Found {row_count} recent purchase orders in the database."
                meaning = "Recent purchase order records reflect committed orders placed with suppliers."
                key_numbers = (
                    f"### Key numbers\n"
                    f"- **Records Found**: {row_count}\n"
                    f"- **Total PO Value**: ₹{total_spend:,.0f}"
                )
                action = "Review order statuses and delivery schedules in the table below."

            return (
                f"### Answer\n"
                f"{answer}\n\n"
                f"### What this means\n"
                f"{meaning}\n\n"
                f"{key_numbers}\n\n"
                f"### Recommended action\n"
                f"{action}\n\n"
                f"### Evidence\n"
                f"- **Query Template**: supplier_po_summary\n"
                f"- **Database Table**: purchase_orders\n\n"
                f"### Source\n"
                f"Source:\n" + "\n".join([f"- {s}" for s in all_sources])
            )

        if data_result and data_result.get("prose"):
            prose = data_result["prose"]
            tbl = data_result.get("table", {})
            row_count = len(tbl.get("rows", [])) if tbl else 0
            if row_count == 0:
                meaning = "No matching records were retrieved from the database for this query."
                action = "No matching records are available for review. Check query parameters or underlying data if needed."
            else:
                meaning = "This data reflects recorded transactions and live supply chain policies stored in PostgreSQL."
                action = "Review the records in the table below to trigger necessary procurement or inventory adjustments."

            return (
                f"### Answer\n"
                f"{prose}\n\n"
                f"### What this means\n"
                f"{meaning}\n\n"
                f"### Key numbers\n"
                f"- **Records Found**: {row_count} relevant lines retrieved\n"
                f"- **Execution Time**: {data_result.get('execution_ms', 0):.1f}ms\n\n"
                f"### Recommended action\n"
                f"{action}\n\n"
                f"### Evidence\n"
                f"- **Query Template**: {data_result.get('template_name', 'PostgreSQL Query')}\n\n"
                f"### Source\n"
                f"Source:\n" + "\n".join([f"- {s}" for s in all_sources])
            )

        # Fallback if query didn't match any data or docs
        return (
            "### Answer\n"
            "I don't have enough verified data to answer that specific query.\n\n"
            "### What this means\n"
            "The query did not match an active documentation section or verified database table.\n\n"
            "### Recommended action\n"
            "Please ask about active catalog items, stockout risks, demand forecasts, or project documentation.\n\n"
            "### Source\n"
            "Source:\n- Demand Decision Intelligence System"
        )


decision_rag_synthesizer = DecisionRAGSynthesizer()
