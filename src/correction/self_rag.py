"""
Self-correcting retrieval, inspired by CRAG (Corrective RAG) and Self-RAG.

The core problem this solves: naive RAG always generates an answer from
whatever it retrieved, even when the retrieved chunks are irrelevant or
insufficient. That's the single biggest source of hallucination in
production RAG systems. This module adds a grading step after retrieval:

  1. Grade each retrieved chunk's relevance to the query (LLM-as-judge, cheap model)
  2. If average relevance is below threshold -> trigger a fallback:
       a. Rewrite the query and retry retrieval (up to MAX_QUERY_REWRITES times)
       b. If still weak after retries -> return an explicit "insufficient evidence"
          response instead of letting the LLM hallucinate an answer
"""
import json
from dataclasses import dataclass

from src.config import MAX_QUERY_REWRITES, RELEVANCE_SCORE_THRESHOLD
from src.llm import get_llm
from src.retrieval.hybrid_retriever import RetrievedChunk


@dataclass
class GradedResult:
    chunks: list[RetrievedChunk]
    avg_relevance: float
    is_sufficient: bool
    rewrites_used: int


GRADER_PROMPT = """You are grading whether a retrieved passage is relevant enough to
help answer a question. Score from 0.0 (irrelevant) to 1.0 (directly answers it).

Question: {question}

Passage:
{passage}

Return ONLY a JSON number, e.g. 0.85"""

REWRITE_PROMPT = """The following query returned mostly irrelevant search results from a
corpus of SEC 10-K filings. Rewrite it to be more likely to match relevant passages —
consider using more specific financial terminology, or restating vague phrasing.

Original query: {query}

Return ONLY the rewritten query, no explanation."""


class SelfCorrectingRetriever:
    def __init__(self, retriever, reranker):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = get_llm()

    def _grade_chunk(self, question: str, chunk: RetrievedChunk) -> float:
        """Grade a single chunk (kept as fallback for small batches)."""
        prompt = GRADER_PROMPT.format(question=question, passage=chunk.chunk.text[:1200])
        response = self.llm.invoke(prompt)
        try:
            return float(response.content.strip())
        except ValueError:
            return 0.0  # fail closed: treat unparseable grade as irrelevant, not relevant

    def _grade_chunks_batch(self, question: str, chunks: list[RetrievedChunk]) -> list[float]:
        """Grade ALL chunks in a single LLM call for efficiency.
        
        Instead of N sequential calls (one per chunk), this sends all chunks
        in one prompt and gets back a list of scores. Reduces retrieval time
        from N * 65s to ~65s on Groq free tier.
        """
        if not chunks:
            return []
        
        if len(chunks) == 1:
            # Single chunk: use the simple method
            return [self._grade_chunk(question, chunks[0])]
        
        # Build numbered passages for batch grading
        passages = "\n\n".join(
            f"[Passage {i+1}]\n{c.chunk.text[:800]}"  # shorter per-passage to fit context
            for i, c in enumerate(chunks)
        )
        
        batch_prompt = f"""You are grading whether retrieved passages are relevant to answer a question.
Score each passage from 0.0 (irrelevant) to 1.0 (directly answers it).

Question: {question}

Passages:
{passages}

Return ONLY a JSON list of {len(chunks)} scores, one per passage in order.
Example format: [0.85, 0.2, 0.9, 0.1]"""
        
        response = self.llm.invoke(batch_prompt)
        raw = response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        
        try:
            scores = json.loads(raw)
            # Validate: must be a list of correct length
            if isinstance(scores, list) and len(scores) == len(chunks):
                # Ensure all are floats
                return [float(s) for s in scores]
            else:
                # Wrong format, fall back to individual grading
                print(f"[SelfCorrectingRetriever] Batch response wrong length: {len(scores)} vs {len(chunks)}")
                return [self._grade_chunk(question, c) for c in chunks]
        except (json.JSONDecodeError, ValueError):
            # Parse error, fall back to individual grading
            print(f"[SelfCorrectingRetriever] Batch parse error, falling back to individual grading")
            return [self._grade_chunk(question, c) for c in chunks]

    def _rewrite_query(self, query: str) -> str:
        response = self.llm.invoke(REWRITE_PROMPT.format(query=query))
        return response.content.strip()

    def retrieve_with_correction(self, query: str, ticker_filter: str | None = None) -> GradedResult:
        rewrites_used = 0
        current_query = query

        while True:
            candidates = self.retriever.retrieve(current_query, ticker_filter=ticker_filter)
            reranked = self.reranker.rerank(current_query, candidates)

            if not reranked:
                avg_relevance = 0.0
            else:
                # Batch grade all chunks in a single LLM call for efficiency
                scores = self._grade_chunks_batch(query, reranked)  # grade against ORIGINAL question
                avg_relevance = sum(scores) / len(scores)
                # drop individually low-scoring chunks even if the average passes
                reranked = [c for c, s in zip(reranked, scores) if s >= RELEVANCE_SCORE_THRESHOLD]

            if avg_relevance >= RELEVANCE_SCORE_THRESHOLD or rewrites_used >= MAX_QUERY_REWRITES:
                return GradedResult(
                    chunks=reranked,
                    avg_relevance=avg_relevance,
                    is_sufficient=avg_relevance >= RELEVANCE_SCORE_THRESHOLD and len(reranked) > 0,
                    rewrites_used=rewrites_used,
                )

            current_query = self._rewrite_query(current_query)
            rewrites_used += 1
