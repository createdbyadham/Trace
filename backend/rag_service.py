import chromadb
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from threading import Lock
from datetime import datetime
import logging
import os
import string
import uuid

logger = logging.getLogger(__name__)


# ── RAG Service (Singleton) ────────────────────────────────────────
#
# Embedding strategy:
#   ChromaDB 1.0.x ships with a built-in ONNX `all-MiniLM-L6-v2`
#   embedding function ("default").  We let ChromaDB own the entire
#   dense-embedding lifecycle (index + query) so there is zero chance
#   of a vector-space mismatch between stored docs and queries.
#   → documents are passed via `documents=`
#   → queries   are passed via `query_texts=`
#
# Reranking:
#   We load a CrossEncoder (ms-marco-MiniLM-L-6-v2) with an ONNX
#   backend for fast CPU inference.  Falls back to PyTorch if the
#   optimum / onnxruntime stack is missing.

class RAGService:
    _instance = None
    _lock = Lock()
    _is_initialized = False

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RAGService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        with self._lock:
            if not self._is_initialized:
                try:
                    # 1. Cross-Encoder Reranker ─ prefer ONNX, fallback PyTorch
                    self.reranker = self._load_reranker()

                    # 2. ChromaDB ─ uses its built-in ONNX all-MiniLM-L6-v2
                    #    Do NOT pass a custom embedding_function; that would
                    #    conflict with collections already persisted with the
                    #    "default" config in ChromaDB 1.0.x.
                    chroma_path = os.path.join(os.getcwd(), "chroma_store")
                    self.chroma_client = chromadb.PersistentClient(path=chroma_path)
                    self.collection = self.chroma_client.get_or_create_collection(
                        name="receipts",
                    )

                    # 3. Sparse Search (BM25) ─ kept in-memory
                    self.bm25 = None
                    self.doc_map: dict[int, dict] = {}
                    self._refresh_bm25()

                    self._is_initialized = True
                    logger.info("Hybrid RAG Service (Dense + Sparse + Rerank) initialized")
                except Exception as e:
                    logger.error(f"Failed to init RAG: {e}")
                    raise

    # ── Model Loader ────────────────────────────────────────────────

    @staticmethod
    def _load_reranker() -> CrossEncoder:
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        try:
            model = CrossEncoder(model_name, backend="onnx")
            logger.info(f"Reranker '{model_name}' loaded with ONNX backend")
            return model
        except Exception as e:
            logger.warning(f"ONNX unavailable for reranker ({e}); falling back to PyTorch")
            return CrossEncoder(model_name)

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _make_id() -> str:
        """Collision-free receipt ID: timestamp + short UUID."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"receipt_{ts}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple BM25 tokenizer: lowercase, strip punctuation, split."""
        return text.lower().translate(
            str.maketrans("", "", string.punctuation)
        ).split()

    def _refresh_bm25(self):
        """Rebuild the in-memory BM25 index from all ChromaDB documents."""
        try:
            all_docs = self.collection.get()
            documents = all_docs["documents"]
            ids = all_docs["ids"]
            metadatas = all_docs["metadatas"]

            if not documents:
                self.bm25 = None
                self.doc_map = {}
                return

            tokenized_corpus = [self._tokenize(doc) for doc in documents]
            self.bm25 = BM25Okapi(tokenized_corpus)

            self.doc_map = {
                i: {"id": ids[i], "doc": documents[i], "meta": metadatas[i]}
                for i in range(len(ids))
            }
            logger.info(f"BM25 index refreshed with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error refreshing BM25: {e}")

    # ── Public API ──────────────────────────────────────────────────

    def add_receipt(self, text: str, metadata: dict,
                    items: list[dict] | None = None) -> str:
        """
        Single entry-point for storing any receipt.

        Creates **multiple** ChromaDB documents per receipt:
          1. One "full_receipt" chunk  – the complete structured text
             (good for merchant / total / date queries)
          2. One "item" chunk per line-item, with merchant + date context
             (good for item-specific and category queries like
              "did I buy a shredder?" or "how much on writing tools?")

        All chunks share a `receipt_group` key so retrieval can
        de-duplicate and always hand the LLM the full receipt.
        """
        receipt_group = self._make_id()

        # Normalise metadata ─ ChromaDB only accepts str | int | float | bool
        base_meta = {
            "source":        str(metadata.get("source", "receipt_ocr")),
            "title":         str(metadata.get("title", "Unknown")),
            "date":          str(metadata.get("date", "Unknown")),
            "total":         float(metadata.get("total", 0.0)),
            "tax":           float(metadata.get("tax", 0.0)),
            "item_count":    int(metadata.get("item_count", 0)),
            "timestamp":     str(metadata.get("timestamp", datetime.now().isoformat())),
            "receipt_group": receipt_group,
            "chunk_type":    "full_receipt",
        }

        # ── Batch lists ──────────────────────────────────────────────
        docs = [text]
        metas = [base_meta]
        ids = [f"{receipt_group}_full"]

        # Per-item chunks
        if items:
            merchant = metadata.get("title", "Unknown")
            date = metadata.get("date", "Unknown")

            for idx, item in enumerate(items):
                desc  = str(item.get("desc", item.get("name", "Item")))
                price = float(item.get("price", 0))
                qty   = item.get("qty", 1)

                item_text = (
                    f"Purchased at {merchant} on {date}: "
                    f"{desc} – ${price:.2f} (qty: {qty})"
                )

                item_meta = {
                    **base_meta,
                    "chunk_type": "item",
                }

                docs.append(item_text)
                metas.append(item_meta)
                ids.append(f"{receipt_group}_item_{idx}")

        self.collection.add(
            documents=docs,
            metadatas=metas,
            ids=ids,
        )

        self._refresh_bm25()
        logger.info(
            f"Stored receipt {receipt_group} ('{base_meta['title']}') – "
            f"1 summary + {len(items or [])} item chunks"
        )
        return receipt_group

    # ── Fallback ─────────────────────────────────────────────────────

    def _fallback_all_receipts(self, max_receipts: int = 20) -> str:
        """
        Return every *full_receipt* document when the normal retrieval
        pipeline scores everything below threshold.

        For a personal expense tracker with a modest number of receipts
        this is cheap (a few KB) and guarantees the LLM can reason about
        categories, item types, etc. that embedding models struggle with.
        """
        try:
            results = self.collection.get(
                where={"chunk_type": "full_receipt"},
            )

            # Legacy data that pre-dates chunk_type tagging
            if not results or not results["documents"]:
                results = self.collection.get()
                if not results or not results["documents"]:
                    return ""

            parts: list[str] = []
            for i, (doc, meta) in enumerate(
                zip(results["documents"], results["metadatas"])
            ):
                if i >= max_receipts:
                    break
                group = meta.get("receipt_group", f"receipt_{i}")
                parts.append(
                    f"[Receipt ID: {group}]\n"
                    f"Merchant: {meta.get('title', 'Unknown')} | "
                    f"Date: {meta.get('date', 'N/A')} | "
                    f"Total: ${float(meta.get('total', 0)):.2f} | "
                    f"Tax: ${float(meta.get('tax', 0)):.2f} | "
                    f"Items: {meta.get('item_count', 0)}\n"
                    f"Content:\n{doc}"
                )

            logger.info(
                f"Fallback: returning {len(parts)} full receipt(s) to LLM"
            )
            return "\n\n---\n\n".join(parts)
        except Exception as e:
            logger.error(f"Fallback retrieval failed: {e}")
            return ""

    # ── Main retrieval ───────────────────────────────────────────────

    async def get_relevant_context(self, query: str, top_k: int = 5) -> str:
        """
        Hybrid retrieval pipeline:
          1. Dense search  (ChromaDB cosine similarity via built-in ONNX embedder)
          2. Sparse search (BM25 keyword matching)
          3. Merge & deduplicate candidates
          4. Rerank with Cross-Encoder
          5. De-duplicate by receipt_group and always surface the
             *full receipt* so the LLM has complete item-level detail
          6. Format top-k for the LLM context window

        **Fallback**: if the scored pipeline produces no results (e.g. the
        query is a category like "writing tools" that doesn't match any
        chunk literally), return *all* full-receipt docs so the LLM can
        do the semantic reasoning itself.
        """
        doc_count = self.collection.count()
        if doc_count == 0:
            return ""

        # Fetch more candidates since each receipt now has many chunks
        fetch_k = min(top_k * 4, doc_count)

        # ── 1. Dense retrieval ──────────────────────────────────────
        dense_results = self.collection.query(
            query_texts=[query],
            n_results=fetch_k,
        )

        # ── 2. Sparse retrieval (BM25) ─────────────────────────────
        sparse_results: list[dict] = []
        if self.bm25 and self.doc_map:
            tokenized_query = self._tokenize(query)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )[:fetch_k]
            sparse_results = [
                self.doc_map[i] for i in top_indices if i in self.doc_map
            ]

        # ── 3. Merge candidates (deduplicate by doc ID) ────────────
        candidates: dict[str, dict] = {}

        for i, doc_id in enumerate(dense_results["ids"][0]):
            candidates[doc_id] = {
                "doc": dense_results["documents"][0][i],
                "meta": dense_results["metadatas"][0][i],
            }

        for item in sparse_results:
            if item["id"] not in candidates:
                candidates[item["id"]] = {
                    "doc": item["doc"],
                    "meta": item["meta"],
                }

        if not candidates:
            return self._fallback_all_receipts()

        # ── 4. Rerank with Cross-Encoder ────────────────────────────
        candidate_ids = list(candidates.keys())
        pairs = [[query, candidates[cid]["doc"]] for cid in candidate_ids]

        scores = self.reranker.predict(pairs)

        scored = sorted(
            zip(candidate_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # ── 5. De-dup by receipt_group & resolve full receipt ───────
        #    When an *item* chunk matches, we still hand the LLM the
        #    full receipt so it can reason over all line-items.
        seen_groups: set[str] = set()
        final_context: list[str] = []

        for doc_id, score in scored:
            if len(final_context) >= top_k:
                break
            if score < -5:
                continue

            meta = candidates[doc_id]["meta"]
            # receipt_group links all chunks of one receipt; fall back
            # to doc_id for legacy docs that pre-date chunking.
            receipt_group = meta.get("receipt_group", doc_id)

            if receipt_group in seen_groups:
                continue
            seen_groups.add(receipt_group)

            # If the hit was an item chunk, swap in the full receipt text
            doc_text = candidates[doc_id]["doc"]
            if meta.get("chunk_type") == "item":
                full_id = f"{receipt_group}_full"
                try:
                    full_doc = self.collection.get(ids=[full_id])
                    if full_doc and full_doc["documents"]:
                        doc_text = full_doc["documents"][0]
                        meta = full_doc["metadatas"][0]
                except Exception:
                    pass  # fall through to the item-chunk text

            final_context.append(
                f"[Receipt ID: {receipt_group}]\n"
                f"Merchant: {meta.get('title', 'Unknown')} | "
                f"Date: {meta.get('date', 'N/A')} | "
                f"Total: ${float(meta.get('total', 0)):.2f} | "
                f"Tax: ${float(meta.get('tax', 0)):.2f} | "
                f"Items: {meta.get('item_count', 0)} | "
                f"Relevance: {float(score):.2f}\n"
                f"Content:\n{doc_text}"
            )

        # ── 6. Fallback when reranker filtered everything out ───────
        if not final_context:
            return self._fallback_all_receipts()

        return "\n\n---\n\n".join(final_context)
