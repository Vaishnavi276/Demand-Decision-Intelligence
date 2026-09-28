# Decision Intelligence RAG Chatbot (`rag-chatbot`)

A dedicated, isolated Retrieval-Augmented Generation (RAG) module for the **Demand & Decision Intelligence System**.

This folder encapsulates all RAG logic—document chunking, embeddings, vector storage, multi-turn conversational context, semantic retrieval, hybrid query routing, and zero-bluff synthesis—allowing teammates to review, benchmark, and evolve the RAG pipeline independently.

---

## Table of Contents
1. [Directory Structure](#directory-structure)
2. [Architecture Overview](#architecture-overview)
3. [Document Ingestion & Chunking](#document-ingestion--chunking)
4. [Embeddings & Vector Store](#embeddings--vector-store)
5. [Semantic Retrieval & Lexical Fallback](#semantic-retrieval--lexical-fallback)
6. [Query Routing (DATA vs DOCS vs HYBRID)](#query-routing-data-vs-docs-vs-hybrid)
7. [SQL Grounding & Thin Adapter Integration](#sql-grounding--thin-adapter-integration)
8. [Multi-Turn Conversational Context](#multi-turn-conversational-context)
9. [LLM Synthesis & Deterministic Fallback](#llm-synthesis--deterministic-fallback)
10. [Strict No-Bluff Rule & Evidence Citations](#strict-no-bluff-rule--evidence-citations)
11. [How to Run Ingestion](#how-to-run-ingestion)
12. [How to Run Tests](#how-to-run-tests)
13. [Required Environment Variables](#required-environment-variables)

---

## 1. Directory Structure

```
rag-chatbot/
├── README.md                           # This comprehensive guide
├── backend/
│   ├── __init__.py                     # Package initialization
│   ├── context_manager.py              # Multi-turn context & entity carryover
│   ├── decision_rag_synthesizer.py     # Main hybrid orchestration engine
│   ├── knowledge_base_service.py       # Document chunking & ChromaDB interface
│   └── retrieval/
│       ├── __init__.py                 # Exports DocumentRetriever & SQLRetriever
│       ├── document_retriever.py       # Semantic vector & lexical retrieval
│       └── sql_retriever.py            # Thin adapter to existing nl_query_service
├── ingestion/
│   ├── __init__.py                     # Ingestion package
│   └── ingest_documents.py             # CLI to chunk and index documentation
├── tests/
│   ├── conftest.py                     # Test configuration & fixtures
│   ├── test_document_retrieval.py      # Vector search & similarity score tests
│   ├── test_hybrid_routing.py          # Query classifier & routing tests
│   ├── test_multiturn_context.py       # Follow-up questions & cost calculation tests
│   └── test_no_bluff.py                # Zero fake defaults & rejection tests
├── docs/
│   └── ARCHITECTURE.md                 # Detailed architecture specification
└── vector_store/
    └── .gitkeep                        # Gitkeep (Chroma binaries ignored by git)
```

---

## 2. Architecture Overview

The system isolates document intelligence while reusing PostgreSQL business intelligence without duplication:

1. **Context Resolution**: The user's query and session history are analyzed by `ContextManager`. Entities like Product/SKU, City, and Reorder Quantity are resolved.
2. **Intent Classification**: The query is routed to `DATA`, `DOCS`, or `HYBRID`.
3. **Retrieval**:
   - `DATA`: Handled by `SQLRetriever` calling `backend.services.nl_query_service`.
   - `DOCS`: Handled by `DocumentRetriever` querying ChromaDB vector store.
   - `HYBRID`: Blends both SQL tabular data and document policy excerpts.
4. **Synthesis**: Synthesized using Groq (`llama-3.3-70b-versatile`) or OpenAI (`gpt-4o-mini`). If API keys are unset, a deterministic 6-part Markdown synthesis is executed.
5. **Audit Trail**: Every response includes an evidence drawer containing exact line numbers, source document paths, similarity scores, and SQL template names.

---

## 3. Document Ingestion & Chunking

Ingestion is handled by `ingestion/ingest_documents.py` and `backend/knowledge_base_service.py`.

### Target Documents
The ingestion pipeline explicitly validates and indexes the following core project documents:
- `docs/ARCHITECTURE.md`
- `docs/PRD.md`
- `docs/SRS.md`
- `docs/ML_ENGINEERING_AND_PIPELINE_GUIDE.md`
- `docs/system_architecture_hld_lld.md`
- `docs/data-engines.md`
- `docs/DATASET_PREPARATION_REPORT.md`
- `PROJECT.md`, `STACK.md`, `DECISIONS.md`, `CONVENTIONS.md`, `README.md`

### Markdown-Aware Chunking Strategy
- **Headings Tracking**: Respects Markdown heading boundaries (`#`, `##`, `###`). Chunks retain their parent section heading.
- **Line Tracking**: Every chunk records its exact 1-indexed `start_line` and `end_line` within the source file.
- **Context Sliding**: Long sections (>850 characters) are split cleanly with a 2-line sliding window to maintain syntactic continuity.
- **Chunk Metadata**: Each chunk stores `chunk_id`, `document_name`, `source_path`, `heading`, `start_line`, `end_line`, and `text`.

---

## 4. Embeddings & Vector Store

- **Embedding Model**: `BAAI/bge-small-en-v1.5` running locally via `fastembed` (384 dimensions, ONNX runtime). No external OpenAI embedding API is required.
- **Vector Database**: `ChromaDB` (Persistent client) stored in `rag-chatbot/vector_store/`.
- **Collection Configuration**: Initialized with `hnsw:space: "cosine"`.
- **Git Hygiene**: Generated Chroma database files are ignored via `.gitignore`; only `rag-chatbot/vector_store/.gitkeep` is checked in.

---

## 5. Semantic Retrieval & Lexical Fallback

### Metric-Consistent Similarity Scores
ChromaDB distance metrics are inspected at runtime:
- For cosine distance $d \in [0, 2]$, similarity is computed as:
  $$\text{score} = \text{round}(\max(0.0, \min(1.0, 1.0 - d)), 4)$$
- For squared L2 distance: $\text{score} = \frac{1}{1 + d}$.

### Resilient Lexical Fallback
If ChromaDB or FastEmbed is unavailable or offline, `KnowledgeBaseService` automatically triggers an in-memory lexical fallback search:
- Token frequency and Markdown heading matching.
- **Labeling Rule**: Scores from lexical search are labeled `retrieval_score` with `score_type: "lexical_frequency"` and `retrieval_method: "lexical"`—they are **never** misleadingly labeled as semantic vector similarity.

---

## 6. Query Routing (DATA vs DOCS vs HYBRID)

The query router in `backend/decision_rag_synthesizer.py` (`route_query` / `classify_query_type`) categorizes inquiries:

| Route | Criteria | Example Queries |
| :--- | :--- | :--- |
| **`DOCS`** | Inquiries about architecture, PRD, methodology, formulas, or system design. | *"What is the demand forecasting methodology?"*, *"What does the PRD say about SLAs?"* |
| **`DATA`** | Operational questions regarding stock levels, demand volume, dead stock, or alerts. | *"Which products are below ROP?"*, *"Show top 10 SKUs by sales volume"* |
| **`HYBRID`** | Explanatory questions requiring live operational figures backed by inventory policy documentation. | *"Why is SKU 19512 recommended for reorder?"*, *"Why are these items below ROP?"* |

---

## 7. SQL Grounding & Thin Adapter Integration

### Architecture Rule: No SQL Duplication
`rag-chatbot/` does **not** duplicate PostgreSQL queries or business models. Instead:
- `backend/retrieval/sql_retriever.py` serves as a thin adapter calling `backend.services.nl_query_service.handle_user_natural_language_query`.
- All SQL security guarantees (guarded templates, tenant dataset binding, SQL injection prevention) remain centrally managed by the existing backend.
- The existing backend (`backend/api/assistant.py`) connects to `rag-chatbot` via `backend/services/rag_adapter.py`.

---

## 8. Multi-Turn Conversational Context

Managed by `backend/context_manager.py`:
- **Entity Extraction**: Captures product IDs (e.g. `476763`), order quantities (e.g. `500 units`), cities (e.g. `Delhi`), and cost intent.
- **Turn Carryover**: In a sequence such as:
  1. *User*: *"Which product has the highest demand?"* (Assistant returns Product `476763`)
  2. *User*: *"What would it cost to procure 500 units?"*
  The second turn carries over `product_id = 476763`, extracts `quantity = 500`, and queries `SupplierProduct` in PostgreSQL for the verified `unit_cost`.
- **Zero Fake Defaults**: The system **never** assumes `19512`, `Delhi`, `500`, or `45.0`. If any required input is missing, the calculation is safely rejected.

---

## 9. LLM Synthesis & Deterministic Fallback

### LLM Synthesis
- Uses **Groq** (`llama-3.3-70b-versatile`) with temperature 0.1 for high-speed, grounded reasoning.
- Falls back to **OpenAI** (`gpt-4o-mini`) if Groq is not configured.
- Structured with a system prompt enforcing the Strict No-Bluff Rule.

### Deterministic 6-Part Fallback
When no LLM API keys are provided in `.env`, the engine produces a grounded 6-part executive Markdown response:
```markdown
### Answer
[Executive summary of verified data or documentation]

### What this means
[Operational interpretation based strictly on retrieved records]

### Key numbers
- **[Metric]**: [Verified value from PostgreSQL or Document citations]

### Recommended action
[Actionable guidance only when supported by underlying data]

### Evidence
- **Data Evidence**: [Exact template rows or document excerpts]

### Source
Source:
- [List of database tables and document files]
```

---

## 10. Strict No-Bluff Rule & Evidence Citations

1. **Missing Data Rejection**: Asking *"What is the total procurement cost?"* without context returns:
   > *"I don't have enough verified data to calculate the procurement cost."*
2. **No Hallucinated Claims**: HYBRID answers quote actual retrieved excerpts rather than asserting generic claims (e.g. no invented "95% cycle service level").
3. **Auditable Evidence**: Every response provides structured evidence metadata:
   - `data_sources`: PostgreSQL tables, templates, row counts, and query execution times.
   - `document_sources`: Document names, relative file paths, chunk IDs, line ranges (`L15-L42`), and similarity match percentages.

---

## 11. How to Run Ingestion

To build or refresh the vector store from project documentation:

```bash
# Using project virtualenv
.\.venv\Scripts\python.exe rag-chatbot/ingestion/ingest_documents.py

# Force rebuild
.\.venv\Scripts\python.exe rag-chatbot/ingestion/ingest_documents.py --force
```

---

## 12. How to Run Tests

The test suite covers document retrieval, similarity scoring, hybrid routing, multi-turn carryover, and no-bluff validation:

```bash
# Run isolated RAG test suite
.\.venv\Scripts\python.exe -m pytest rag-chatbot/tests/ -v

# Run both RAG tests and existing assistant regression tests
.\.venv\Scripts\python.exe -m pytest rag-chatbot/tests/ backend/tests/ -v
```

---

## 13. Required Environment Variables

Configured in `.env` (or `.env.example`):

```bash
# LLM API Keys (Optional - System uses deterministic 6-part fallback if unset)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
OPENAI_API_KEY=your_openai_api_key_here
```
