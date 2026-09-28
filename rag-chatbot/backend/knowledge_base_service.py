"""
Knowledge Base Service — Semantic Document Ingestion & Vector Retrieval
Location: rag-chatbot/backend/knowledge_base_service.py
Project: Demand-Decision-Intelligence

Responsibilities:
1. Ingests verified project documentation with line-tracking and Markdown section awareness.
2. Fast local embedding generation via fastembed (BAAI/bge-small-en-v1.5).
3. Persistent ChromaDB storage at rag-chatbot/vector_store/ (excluded from git).
4. Verifies distance metric space ('cosine', 'l2', 'ip') for mathematically sound similarity scoring.
5. Zero-dependency lexical fallback returning explicit 'retrieval_score' (never misleadingly labeled semantic similarity).
6. Comprehensive stats reporting indexed vs missing files.
"""

import os
import re
import math
import time
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Paths
_SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
_RAG_DIR = os.path.dirname(_SERVICE_DIR)
_ROOT_DIR = os.path.dirname(_RAG_DIR)
DEFAULT_VECTOR_STORE_DIR = os.path.join(_RAG_DIR, "vector_store")

# Explicit list of target project documentation files to index
TARGET_DOCUMENTS = [
    "docs/ARCHITECTURE.md",
    "docs/PRD.md",
    "docs/SRS.md",
    "docs/ML_ENGINEERING_AND_PIPELINE_GUIDE.md",
    "docs/system_architecture_hld_lld.md",
    "docs/data-engines.md",
    "docs/DATASET_PREPARATION_REPORT.md",
    "PROJECT.md",
    "STACK.md",
    "DECISIONS.md",
    "CONVENTIONS.md",
    "README.md",
]


class KnowledgeBaseService:
    def __init__(
        self,
        vector_store_path: str = DEFAULT_VECTOR_STORE_DIR,
        model_name: str = "BAAI/bge-small-en-v1.5",
        project_root: str = _ROOT_DIR
    ):
        self.vector_store_path = vector_store_path
        self.model_name = model_name
        self.project_root = project_root
        self.collection_name = "ddi_documentation"
        self._embed_model = None
        self._chroma_client = None
        self._collection = None
        self._in_memory_chunks: List[Dict[str, Any]] = []

    def _get_embed_model(self):
        """Lazy-load fastembed TextEmbedding model."""
        if self._embed_model is None:
            try:
                from fastembed import TextEmbedding
                self._embed_model = TextEmbedding(model_name=self.model_name)
            except Exception as e:
                logger.warning(f"[KnowledgeBase] Could not initialize fastembed: {e}. Lexical fallback will be used.")
                self._embed_model = None
        return self._embed_model

    def _get_collection(self):
        """Lazy-load ChromaDB collection with persistent storage."""
        if self._collection is None:
            try:
                import chromadb
                os.makedirs(self.vector_store_path, exist_ok=True)
                self._chroma_client = chromadb.PersistentClient(path=self.vector_store_path)
                self._collection = self._chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                logger.warning(f"[KnowledgeBase] Could not initialize ChromaDB: {e}.")
                self._collection = None
        return self._collection

    def _compute_similarity(self, distance: float, space: str = "cosine") -> float:
        """
        Mathematically maps distance to a normalized similarity score [0.0, 1.0]
        based on the actual configured Chroma metric space.
        """
        if space == "cosine":
            # In Chroma (HNSW), cosine distance = 1 - cos(theta), where dist in [0, 2]
            sim = 1.0 - distance
            return round(max(0.0, min(1.0, sim)), 4)
        elif space == "l2":
            # L2 squared distance: sim = 1 / (1 + distance)
            return round(1.0 / (1.0 + max(0.0, distance)), 4)
        elif space == "ip":
            # Inner product distance: dist = 1 - <u,v>
            return round(max(0.0, min(1.0, 1.0 - distance)), 4)
        else:
            return round(max(0.0, min(1.0, 1.0 - distance)), 4)

    def chunk_document(self, relative_path: str, content: str) -> List[Dict[str, Any]]:
        """
        Splits Markdown content into contextual, line-tracked chunks.
        Tracks Markdown headings (#, ##, ###), line start/end numbers, and source paths.
        """
        lines = content.splitlines()
        chunks: List[Dict[str, Any]] = []
        doc_name = os.path.basename(relative_path)

        current_heading = doc_name
        current_lines: List[str] = []
        chunk_start_line = 1
        chunk_index = 0

        for line_num, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Heading change trigger
            heading_match = re.match(r'^(#{1,4})\s+(.+)$', stripped)
            if heading_match:
                # Flush previous buffer if substantial
                if current_lines and sum(len(l) for l in current_lines) >= 100:
                    chunk_text = "\n".join(current_lines).strip()
                    if len(chunk_text) > 25:
                        chunk_index += 1
                        chunks.append({
                            "chunk_id": f"{relative_path}#chunk-{chunk_index}",
                            "document_name": doc_name,
                            "source_path": relative_path,
                            "text": chunk_text,
                            "start_line": chunk_start_line,
                            "end_line": line_num - 1,
                            "heading": current_heading,
                        })
                    current_lines = []
                    chunk_start_line = line_num

                current_heading = heading_match.group(2).strip()

            current_lines.append(line)

            # Max size split (~850 chars or ~140 words)
            approx_len = sum(len(l) for l in current_lines)
            if approx_len >= 850:
                chunk_text = "\n".join(current_lines).strip()
                if len(chunk_text) > 25:
                    chunk_index += 1
                    chunks.append({
                        "chunk_id": f"{relative_path}#chunk-{chunk_index}",
                        "document_name": doc_name,
                        "source_path": relative_path,
                        "text": chunk_text,
                        "start_line": chunk_start_line,
                        "end_line": line_num,
                        "heading": current_heading,
                    })
                # Maintain 2-line sliding context
                current_lines = current_lines[-2:] if len(current_lines) > 2 else []
                chunk_start_line = max(1, line_num - 1)

        # Flush trailing lines
        if current_lines:
            chunk_text = "\n".join(current_lines).strip()
            if len(chunk_text) > 25:
                chunk_index += 1
                chunks.append({
                    "chunk_id": f"{relative_path}#chunk-{chunk_index}",
                    "document_name": doc_name,
                    "source_path": relative_path,
                    "text": chunk_text,
                    "start_line": chunk_start_line,
                    "end_line": len(lines),
                    "heading": current_heading,
                })

        return chunks

    def ingest_documents(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Inspects verified project documentation, chunks text, generates embeddings,
        and saves to persistent ChromaDB collection.
        Explicitly tracks and reports indexed vs missing documents.
        """
        collection = self._get_collection()

        if collection and not force_rebuild:
            count = collection.count()
            if count > 0:
                stats = self.get_document_stats()
                return {
                    "status": "already_indexed",
                    "message": f"Knowledge base already indexed with {count} chunks across {stats['total_documents']} documents.",
                    "stats": stats,
                }

        if force_rebuild and collection:
            self.clear_index()
            collection = self._get_collection()

        all_chunks: List[Dict[str, Any]] = []
        indexed_files: List[str] = []
        missing_files: List[str] = []
        empty_files: List[str] = []

        for rel_path in TARGET_DOCUMENTS:
            abs_path = os.path.join(self.project_root, rel_path)
            if not os.path.exists(abs_path):
                missing_files.append(rel_path)
                continue

            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()

                if not content.strip():
                    empty_files.append(rel_path)
                    continue

                doc_chunks = self.chunk_document(rel_path, content)
                if doc_chunks:
                    all_chunks.extend(doc_chunks)
                    indexed_files.append(rel_path)
            except Exception as e:
                logger.error(f"[KnowledgeBase] Error reading {rel_path}: {e}")

        self._in_memory_chunks = all_chunks

        if not all_chunks:
            return {
                "status": "warning",
                "message": "No valid document chunks were extracted.",
                "indexed_files": indexed_files,
                "missing_files": missing_files,
                "empty_files": empty_files,
                "total_chunks": 0,
            }

        # Embed and insert into ChromaDB
        embed_model = self._get_embed_model()
        if collection and embed_model:
            texts = [c["text"] for c in all_chunks]
            ids = [c["chunk_id"] for c in all_chunks]
            metadatas = [{
                "document_name": c["document_name"],
                "source_path": c["source_path"],
                "start_line": int(c["start_line"]),
                "end_line": int(c["end_line"]),
                "heading": c.get("heading", ""),
            } for c in all_chunks]

            embeddings = [[float(v) for v in vec] for vec in embed_model.embed(texts)]
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )

        return {
            "status": "success",
            "indexed_files": indexed_files,
            "missing_files": missing_files,
            "empty_files": empty_files,
            "total_chunks": len(all_chunks),
            "total_documents": len(indexed_files),
            "vector_store_path": self.vector_store_path,
        }

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic vector search over documentation chunks.
        Falls back seamlessly to lexical token matching if Chroma/FastEmbed is unavailable.
        """
        if not query or not query.strip():
            return []

        collection = self._get_collection()
        embed_model = self._get_embed_model()

        # Vector search path
        if collection and embed_model and collection.count() > 0:
            try:
                raw_vec = list(embed_model.embed([query]))[0]
                query_vec = [float(x) for x in raw_vec]

                # Determine distance metric configured on collection
                col_meta = collection.metadata or {}
                space = col_meta.get("hnsw:space", "cosine")

                results = collection.query(
                    query_embeddings=[query_vec],
                    n_results=min(top_k, collection.count())
                )

                hits: List[Dict[str, Any]] = []
                if results and results.get("ids") and len(results["ids"]) > 0:
                    ids = results["ids"][0]
                    docs = results["documents"][0] if results.get("documents") else []
                    metas = results["metadatas"][0] if results.get("metadatas") else []
                    distances = results["distances"][0] if results.get("distances") else []

                    for i in range(len(ids)):
                        dist = float(distances[i]) if i < len(distances) else 0.5
                        score = self._compute_similarity(dist, space=space)
                        meta = metas[i] if i < len(metas) else {}

                        hits.append({
                            "chunk_id": ids[i],
                            "text": docs[i] if i < len(docs) else "",
                            "document_name": meta.get("document_name", "Documentation"),
                            "source_path": meta.get("source_path", ""),
                            "start_line": int(meta.get("start_line", 1)),
                            "end_line": int(meta.get("end_line", 1)),
                            "heading": meta.get("heading", ""),
                            "retrieval_score": score,
                            "similarity_score": score,
                            "score_type": f"{space}_similarity",
                            "retrieval_method": "semantic_vector",
                        })
                    return hits
            except Exception as e:
                logger.warning(f"[KnowledgeBase] Vector query failed: {e}. Executing lexical fallback.")

        # Lexical Fallback search
        return self._lexical_search(query, top_k=top_k)

    def _lexical_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Lightweight keyword/token frequency fallback.
        Explicitly marks scores as 'retrieval_score' and 'score_type' as 'lexical_frequency'.
        """
        if not self._in_memory_chunks:
            for rel_path in TARGET_DOCUMENTS:
                abs_path = os.path.join(self.project_root, rel_path)
                if os.path.exists(abs_path):
                    try:
                        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                        if content.strip():
                            self._in_memory_chunks.extend(self.chunk_document(rel_path, content))
                    except Exception:
                        pass

        query_tokens = set(re.findall(r'\b\w{3,}\b', query.lower()))
        if not query_tokens:
            return []

        scored_hits = []
        for c in self._in_memory_chunks:
            text_lower = c["text"].lower()
            heading_lower = c.get("heading", "").lower()

            match_count = sum(text_lower.count(tok) for tok in query_tokens)
            heading_bonus = sum(3 for tok in query_tokens if tok in heading_lower)
            score_raw = match_count + heading_bonus

            if score_raw > 0:
                normalized_score = round(min(0.95, 0.35 + (score_raw * 0.05)), 4)
                scored_hits.append({
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "document_name": c["document_name"],
                    "source_path": c["source_path"],
                    "start_line": int(c["start_line"]),
                    "end_line": int(c["end_line"]),
                    "heading": c.get("heading", ""),
                    "retrieval_score": normalized_score,
                    "similarity_score": normalized_score,
                    "score_type": "lexical_frequency",
                    "retrieval_method": "lexical",
                })

        scored_hits.sort(key=lambda x: x["retrieval_score"], reverse=True)
        return scored_hits[:top_k]

    def get_document_stats(self) -> Dict[str, Any]:
        """
        Returns complete statistics on the current knowledge base index.
        Explicitly reports indexed documents vs missing documents.
        """
        collection = self._get_collection()
        total_chunks = collection.count() if collection else len(self._in_memory_chunks)

        existing_docs = []
        missing_docs = []
        for d in TARGET_DOCUMENTS:
            abs_p = os.path.join(self.project_root, d)
            if os.path.exists(abs_p):
                existing_docs.append(os.path.basename(d))
            else:
                missing_docs.append(d)

        return {
            "total_chunks": total_chunks,
            "total_documents": len(existing_docs) if total_chunks > 0 else 0,
            "document_count": len(TARGET_DOCUMENTS),
            "indexed_documents": existing_docs if total_chunks > 0 else [],
            "indexed_files": [d for d in TARGET_DOCUMENTS if os.path.exists(os.path.join(self.project_root, d))] if total_chunks > 0 else [],
            "missing_files": missing_docs,
            "embedding_model": self.model_name,
            "vector_store_path": self.vector_store_path,
            "collection_name": self.collection_name,
            "is_persistent": True if collection else False,
        }

    def clear_index(self) -> None:
        """Clears the collection to allow complete re-indexing."""
        if self._chroma_client:
            try:
                self._chroma_client.delete_collection(name=self.collection_name)
            except Exception:
                pass
            self._collection = None
        self._in_memory_chunks = []


# Singleton instance pointing to rag-chatbot vector_store
knowledge_base_service = KnowledgeBaseService()
