"""
Document Ingestion CLI & Pipeline
Location: rag-chatbot/ingestion/ingest_documents.py
Project: Demand-Decision-Intelligence

Executes:
1. Target document validation against file system.
2. Explicit reporting of indexed vs missing files.
3. Markdown-aware chunking preserving heading and line numbers.
4. Embedding generation via fastembed and persistence into ChromaDB vector store.
"""

import os
import sys
import logging
from typing import Dict, Any, List

# Ensure parent and project root paths are in sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_RAG_ROOT = os.path.dirname(_CURRENT_DIR)
_PROJECT_ROOT = os.path.dirname(_RAG_ROOT)

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _RAG_ROOT not in sys.path:
    sys.path.insert(0, _RAG_ROOT)

# Import isolated knowledge base service
try:
    from rag_chatbot.backend.knowledge_base_service import (
        knowledge_base_service,
        TARGET_DOCUMENTS,
    )
except ImportError:
    try:
        from backend.knowledge_base_service import (
            knowledge_base_service,
            TARGET_DOCUMENTS,
        )
    except ImportError:
        from ..backend.knowledge_base_service import (
            knowledge_base_service,
            TARGET_DOCUMENTS,
        )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_documents")


def run_ingestion(force_rebuild: bool = True) -> Dict[str, Any]:
    """
    Executes the ingestion pipeline.
    Validates documents, chunks text, embeds chunks, and saves to ChromaDB.
    """
    print("=" * 70)
    print("Starting Document Ingestion for Decision Intelligence RAG")
    print("=" * 70)

    print(f"Target Documents ({len(TARGET_DOCUMENTS)} files):")
    for doc in TARGET_DOCUMENTS:
        full_p = os.path.join(_PROJECT_ROOT, doc)
        status = "FOUND" if os.path.exists(full_p) else "MISSING"
        print(f"  [{status:7s}] {doc}")

    print("\nProcessing and embedding documents into ChromaDB...")
    result = knowledge_base_service.ingest_documents(force_rebuild=force_rebuild)

    print("\n" + "=" * 70)
    print("Ingestion Summary Report:")
    print("=" * 70)
    print(f"Status           : {result.get('status')}")
    print(f"Indexed Files    : {len(result.get('indexed_files', []))}")
    print(f"Missing Files    : {len(result.get('missing_files', []))}")
    print(f"Empty Files      : {len(result.get('empty_files', []))}")
    print(f"Total Chunks     : {result.get('total_chunks', 0)}")
    print(f"Vector Store Path: {result.get('vector_store_path')}")

    if result.get("missing_files"):
        print("\nMissing Documents (Not Indexed):")
        for m in result["missing_files"]:
            print(f"  - {m}")

    print("=" * 70)
    return result


if __name__ == "__main__":
    force = "--force" in sys.argv or "-f" in sys.argv or True
    run_ingestion(force_rebuild=force)
