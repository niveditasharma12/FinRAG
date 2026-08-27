"""
Orchestrates the full FinRAG flow:
  route -> (per sub-question) retrieve+correct -> generate -> verify -> combine
"""
from dataclasses import dataclass, field

from src.correction.self_rag import SelfCorrectingRetriever
from src.llm import get_llm
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
from src.retrieval.query_router import QueryRouter, QueryType
from src.retrieval.reranker import Reranker
from src.verification.citation_check import CitationVerifier

GENERATION_PROMPT = """Answer the question using ONLY the provided context from SEC filings.
If the context doesn't contain enough information, say so explicitly rather than guessing.
Cite specific numbers/facts as they appear in the context.

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class PipelineResult:
    answer: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    query_type: QueryType = QueryType.SIMPLE
    sub_question_answers: dict[str, str] = field(default_factory=dict)
    insufficient_evidence: bool = False


class FinRAGPipeline:
    def __init__(self, use_self_correction: bool = True, use_citation_verification: bool = True):
        self.router = QueryRouter()
        self.retriever = HybridRetriever()
        self.retriever.load()
        self.reranker = Reranker()
        self.self_correcting = SelfCorrectingRetriever(self.retriever, self.reranker)
        self.verifier = CitationVerifier()
        self.llm = get_llm()

        self.use_self_correction = use_self_correction
        self.use_citation_verification = use_citation_verification

    def _retrieve(self, question: str, ticker_filter: str | None = None) -> list[RetrievedChunk]:
        if self.use_self_correction:
            graded = self.self_correcting.retrieve_with_correction(question, ticker_filter=ticker_filter)
            return graded.chunks if graded.is_sufficient else []
        candidates = self.retriever.retrieve(question, ticker_filter=ticker_filter)
        return self.reranker.rerank(question, candidates)

    def _generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "I don't have sufficient evidence in the indexed filings to answer this confidently."
        context = "\n\n---\n\n".join(f"[{c.chunk.ticker} {c.chunk.filing_date}] {c.chunk.text}" for c in chunks)
        response = self.llm.invoke(GENERATION_PROMPT.format(context=context, question=question))
        return response.content

    def answer(self, question: str) -> PipelineResult:
        routed = self.router.route(question)

        if routed.query_type == QueryType.SIMPLE:
            ticker = routed.tickers_mentioned[0] if routed.tickers_mentioned else None
            chunks = self._retrieve(question, ticker_filter=ticker)
            answer_text = self._generate(question, chunks)
            if self.use_citation_verification and chunks:
                checks = self.verifier.verify(answer_text, chunks)
                answer_text = self.verifier.filter_unsupported(answer_text, checks)
            return PipelineResult(answer=answer_text, chunks=chunks, query_type=routed.query_type,
                                   insufficient_evidence=not chunks)

        # multi_hop / comparison: answer each sub-question independently, then synthesize
        sub_answers: dict[str, str] = {}
        all_chunks: list[RetrievedChunk] = []
        for sub_q in routed.sub_questions:
            ticker = next((t for t in routed.tickers_mentioned if t.upper() in sub_q.upper()), None)
            chunks = self._retrieve(sub_q, ticker_filter=ticker)
            sub_answers[sub_q] = self._generate(sub_q, chunks)
            all_chunks.extend(chunks)

        synthesis_prompt = (
            f"Original question: {question}\n\n"
            f"Sub-answers gathered:\n" +
            "\n".join(f"- {q}: {a}" for q, a in sub_answers.items()) +
            "\n\nSynthesize a single coherent answer to the original question using these sub-answers."
        )
        final_answer = self.llm.invoke(synthesis_prompt).content

        if self.use_citation_verification and all_chunks:
            checks = self.verifier.verify(final_answer, all_chunks)
            final_answer = self.verifier.filter_unsupported(final_answer, checks)

        return PipelineResult(
            answer=final_answer,
            chunks=all_chunks,
            query_type=routed.query_type,
            sub_question_answers=sub_answers,
        )
