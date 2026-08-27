"""
Post-generation grounding check.

Even with good retrieval, LLMs sometimes state things not actually present
in the retrieved context (subtle number swaps, over-generalization, etc.).
This module splits the generated answer into individual claims, then checks
each claim against the retrieved chunks it cites using an LLM-as-judge
entailment check. Unsupported claims are flagged (and can be stripped) before
the answer reaches the user.
"""
import json
from dataclasses import dataclass

from src.config import CITATION_SUPPORT_THRESHOLD
from src.llm import get_llm
from src.retrieval.hybrid_retriever import RetrievedChunk


@dataclass
class ClaimCheck:
    claim: str
    supported: bool
    support_score: float
    cited_chunk_id: str | None


SPLIT_PROMPT = """Split the following answer into a list of individual factual claims.
Each claim should be a standalone sentence. Return ONLY a JSON list of strings.

Answer:
{answer}"""

ENTAILMENT_PROMPT = """Does the SOURCE passage support the CLAIM? Score 0.0 (not supported /
contradicted) to 1.0 (fully supported). Be strict: a claim about a specific number or fact
must have that exact number/fact present in the source to score above 0.5.

SOURCE:
{source}

CLAIM:
{claim}

Return ONLY a JSON number."""


class CitationVerifier:
    def __init__(self):
        self.llm = get_llm()

    def _split_claims(self, answer: str) -> list[str]:
        response = self.llm.invoke(SPLIT_PROMPT.format(answer=answer))
        raw = response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return [answer]  # fall back to treating the whole answer as one claim

    def _best_support(self, claim: str, chunks: list[RetrievedChunk]) -> tuple[float, str | None]:
        best_score, best_id = 0.0, None
        for c in chunks:
            response = self.llm.invoke(ENTAILMENT_PROMPT.format(source=c.chunk.text[:1200], claim=claim))
            try:
                score = float(response.content.strip())
            except ValueError:
                score = 0.0
            if score > best_score:
                best_score, best_id = score, c.chunk.chunk_id
        return best_score, best_id

    def verify(self, answer: str, chunks: list[RetrievedChunk]) -> list[ClaimCheck]:
        claims = self._split_claims(answer)
        results = []
        for claim in claims:
            score, chunk_id = self._best_support(claim, chunks)
            results.append(ClaimCheck(
                claim=claim,
                supported=score >= CITATION_SUPPORT_THRESHOLD,
                support_score=score,
                cited_chunk_id=chunk_id,
            ))
        return results

    def filter_unsupported(self, answer: str, checks: list[ClaimCheck]) -> str:
        """Return the answer with unsupported claims flagged inline (kept, not silently dropped,
        so the user/interviewer can see exactly what the verifier caught)."""
        unsupported = [c.claim for c in checks if not c.supported]
        if not unsupported:
            return answer
        flags = "\n\n⚠️ Unverified claims (not clearly supported by retrieved sources):\n" + \
                "\n".join(f"- {c}" for c in unsupported)
        return answer + flags
