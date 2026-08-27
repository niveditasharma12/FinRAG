"""
Cross-encoder reranking. Bi-encoder (dense) retrieval scores query and
document independently then compares vectors — fast but less precise.
A cross-encoder scores the (query, document) pair jointly, which is far
more accurate but too slow to run over the whole corpus. Standard pattern:
retrieve ~15-30 broad candidates cheaply (hybrid retriever), then rerank
the shortlist with the expensive-but-accurate cross-encoder.
"""
from sentence_transformers import CrossEncoder

from src.config import RERANKER_MODEL, TOP_K_AFTER_RERANK
from src.retrieval.hybrid_retriever import RetrievedChunk


class Reranker:
    def __init__(self):
        self.model = CrossEncoder(RERANKER_MODEL)

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c.fused_score = float(s)  # overwrite fused score with cross-encoder relevance score
        candidates.sort(key=lambda c: c.fused_score, reverse=True)
        return candidates[:TOP_K_AFTER_RERANK]
