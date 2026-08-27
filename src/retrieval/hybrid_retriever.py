"""
Hybrid retrieval: combines dense (embedding/FAISS) similarity with sparse
(BM25) lexical scoring, then fuses the two ranked lists.

Why hybrid: dense embeddings capture semantic meaning but frequently miss
exact matches on numbers, tickers, dates, and rare proper nouns ("$4.2B",
"CIK 0000320193") that matter a lot in financial text. BM25 nails exact
terms but misses paraphrase/semantic matches. Fusing both consistently
outperforms either alone on domain-specific corpora.
"""
import pickle
from dataclasses import dataclass

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import (EMBEDDING_MODEL, HYBRID_ALPHA, INDEX_DIR,
                         TOP_K_DENSE, TOP_K_SPARSE)
from src.ingest.chunking import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    dense_score: float
    sparse_score: float
    fused_score: float


class HybridRetriever:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.chunks: list[Chunk] = []
        self.faiss_index: faiss.Index | None = None
        self.bm25: BM25Okapi | None = None

    # ---------- index build ----------
    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]

        # Dense index
        embeddings = self.embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine sim
        self.faiss_index.add(np.array(embeddings, dtype="float32"))

        # Sparse index
        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)

        print(f"[hybrid_retriever] indexed {len(chunks)} chunks")

    def save(self) -> None:
        faiss.write_index(self.faiss_index, str(INDEX_DIR / "dense.index"))
        with open(INDEX_DIR / "sparse.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        with open(INDEX_DIR / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self) -> None:
        index_path = INDEX_DIR / "dense.index"
        if not index_path.exists():
            raise RuntimeError(
                f"No index found at {index_path}. Build it first:\n"
                "  python -m src.ingest.fetch_filings --tickers AAPL MSFT GOOGL --years 2021 2022 2023\n"
                "  python -m src.ingest.build_index"
            )
        self.faiss_index = faiss.read_index(str(index_path))
        with open(INDEX_DIR / "sparse.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        with open(INDEX_DIR / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

    # ---------- query ----------
    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        if scores.max() == scores.min():
            return np.zeros_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())

    def retrieve(self, query: str, ticker_filter: str | None = None) -> list[RetrievedChunk]:
        # Dense search
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        dense_scores, dense_idx = self.faiss_index.search(np.array(q_emb, dtype="float32"), TOP_K_DENSE)
        dense_scores, dense_idx = dense_scores[0], dense_idx[0]

        # Sparse search
        tokenized_query = query.lower().split()
        sparse_scores_all = np.array(self.bm25.get_scores(tokenized_query))
        sparse_top_idx = np.argsort(sparse_scores_all)[::-1][:TOP_K_SPARSE]

        # Union of candidate indices
        candidate_idx = set(dense_idx.tolist()) | set(sparse_top_idx.tolist())
        candidate_idx.discard(-1)

        dense_lookup = {int(i): float(s) for i, s in zip(dense_idx, dense_scores)}
        results = []
        for idx in candidate_idx:
            chunk = self.chunks[idx]
            if ticker_filter and chunk.ticker != ticker_filter:
                continue
            d_score = dense_lookup.get(idx, 0.0)
            s_score = float(sparse_scores_all[idx])
            results.append((idx, chunk, d_score, s_score))

        if not results:
            return []

        d_arr = self._normalize(np.array([r[2] for r in results]))
        s_arr = self._normalize(np.array([r[3] for r in results]))
        fused = HYBRID_ALPHA * d_arr + (1 - HYBRID_ALPHA) * s_arr

        retrieved = [
            RetrievedChunk(chunk=r[1], dense_score=r[2], sparse_score=r[3], fused_score=float(f))
            for r, f in zip(results, fused)
        ]
        retrieved.sort(key=lambda r: r.fused_score, reverse=True)
        return retrieved
