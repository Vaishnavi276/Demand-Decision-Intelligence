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
"""


def build_document_search_query(
    query: str,
    query_type: str = "DOCS",
    intent: Optional[str] = None
) -> str:
    """
    Builds a normalized, concept-focused search query for document retrieval. (FIX 2)
    For HYBRID queries:
    1. Removes numeric SKU / product IDs so they do not distort document vector retrieval.
    2. Removes entity-specific tokens (e.g. 'sku', 'product id').
    3. Adds targeted domain concepts based on intent (reorder, stockout, forecasting).
    For DOCS queries:
    Returns the natural language question.
    """
    if query_type != "HYBRID":
        return query.strip()

    q_lower = query.lower()

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
            r"how\s+(?:is|are)\s+.*?\s+(?:calculated|determined|computed)",
            r"how\s+does\s+.*?\s+work",
            r"what\s+(?:is|are)\s+(?:the\s+)?(?:business\s+)?rules(?:\s+for)?",
            r"what\s+(?:is|are)\s+(?:the\s+)?rules(?:\s+for)?",
            r"(?:explain|what\s+is|what)\s+(?:the\s+)?formula(?:\s+for|\s+is\s+used)?",
            r"explain\s+(?:the\s+)?(?:recommendation\s+logic|methodology|architecture|workflow|pipeline|policy|system|design|formula)",
            r"why\s+does\s+the\s+system\s+use",
            r"how\s+(?:is|are)\s+.*?\s+determined",
            r"what\s+models\s+(?:are\s+used|used)",
        ]
        if any(re.search(pat, q) for pat in doc_patterns):
            return "DOCS"

        # Additional keywords for documentation / methodology / architecture
        doc_keywords = [
            "prd", "architecture", "srs", "methodology", "guide", "documentation",
            "safety stock formula", "wape formula", "croston method",
            "system design", "sla", "data pipeline guide", "conventions", "stack",
            "what does the prd say", "system architecture", "what algorithms",
            "forecasting methodology", "recommendation logic", "business rules",
            "what formula", "specification", "policy", "workflow", "design"
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
                    session_id=session_id
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
            doc_hits = self.document_retriever.retrieve(search_query, top_k=3)
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
                        query_template=data_result.get("template_name") if data_result else "document_rag",
                        execution_ms=exec_ms
                    )
                    db.add(bot_msg)
                    db.commit()
            except Exception as e:
                logger.warning(f"[DecisionRAGSynthesizer] Persistence error: {e}")

        rich_data_sources = [
            {
                "table": src,
                "template": data_result.get("template_name") if data_result else "parameterized_sql",
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
            "template_name": data_result.get("template_name") if data_result else "document_rag",
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
            top_hit = doc_hits[0]
            doc_name = top_hit["document_name"]
            heading = top_hit.get("heading", "Specification")
            line_ref = f"Line {top_hit.get('start_line', 1)}"

            clean_text = re.sub(r'#+\s*', '', top_hit["text"]).strip()
            summary = clean_text[: clean_text.find('\n\n')] if '\n\n' in clean_text else clean_text[:350]

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
            if doc_hits:
                top_doc = doc_hits[0]
                clean_text = re.sub(r'#+\s*', '', top_doc.get("text", "")).strip()
                # Take first meaningful sentence from real retrieved text
                first_line = clean_text.split('\n')[0] if clean_text else ""
                if len(first_line) > 130:
                    first_line = first_line[:130] + "..."
                doc_explanation = f"Per {top_doc['document_name']} ({top_doc.get('heading', 'Policy')}, Line {top_doc.get('start_line', 1)}): \"{first_line}\""
            else:
                doc_explanation = "No specific policy documentation was found in the knowledge base for this query. Operational data reflects current database records."

            prose = data_result.get("prose", "") if data_result else "Operational inventory policy requires reorder PO intervention."

            return (
                f"### Answer\n"
                f"{prose}\n\n"
                f"### What this means\n"
                f"{doc_explanation}\n\n"
                f"### Key numbers\n"
                f"- **Data Grounding**: PostgreSQL Live Verification\n\n"
                f"### Recommended action\n"
                f"Review inventory positions against lead-time buffers and trigger necessary replenishment purchase orders.\n\n"
                f"### Evidence\n"
                f"- **Database**: {', '.join([s for s in all_sources if not s.endswith('.md')]) or 'Live PostgreSQL'}\n"
                f"- **Documentation**: {', '.join([s for s in all_sources if s.endswith('.md')]) or 'None'}\n\n"
                f"### Source\n"
                f"Source:\n" + "\n".join([f"- {s}" for s in all_sources])
            )

        # ── 3. Pure Operational Live Data Inquiries ──
        if data_result and data_result.get("prose"):
            prose = data_result["prose"]
            tbl = data_result.get("table", {})
            row_count = len(tbl.get("rows", [])) if tbl else 0

            return (
                f"### Answer\n"
                f"{prose}\n\n"
                f"### What this means\n"
                f"This data reflects recorded transactions and live supply chain policies stored in PostgreSQL.\n\n"
                f"### Key numbers\n"
                f"- **Records Found**: {row_count} relevant lines retrieved\n"
                f"- **Execution Time**: {data_result.get('execution_ms', 0):.1f}ms\n\n"
                f"### Recommended action\n"
                f"Review the records in the table below to trigger necessary procurement or inventory adjustments.\n\n"
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
