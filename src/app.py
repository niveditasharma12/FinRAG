"""
FastAPI entrypoint. Run with:
    uvicorn src.app:app --reload
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="FinRAG", description="Self-correcting multi-hop RAG over SEC filings")

_pipeline = None
_init_error: str | None = None


def _get_pipeline():
    """Lazily build the pipeline (loads FAISS/BM25 index + models) on first use.

    This keeps app import cheap and lets /health respond even when the index
    hasn't been built or the API key isn't set yet.
    """
    global _pipeline, _init_error
    if _pipeline is not None:
        return _pipeline
    try:
        from src.pipeline import FinRAGPipeline
        from src.llm import is_groq_provider

        # Groq free tier (8 000 TPM) can only handle ~1 call per minute.
        # Disable self-correction and citation verification which add many
        # extra LLM calls and would exhaust the quota.
        use_features = not is_groq_provider()
        _pipeline = FinRAGPipeline(
            use_self_correction=use_features,
            use_citation_verification=use_features,
        )
        _init_error = None
    except Exception as e:
        _init_error = str(e)
        _pipeline = None
        raise HTTPException(
            status_code=503,
            detail=f"FinRAG pipeline not ready: {_init_error}",
        )
    return _pipeline


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    query_type: str
    num_sources: int
    sub_question_answers: dict[str, str] = {}
    insufficient_evidence: bool = False


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    pipeline = _get_pipeline()
    result = pipeline.answer(req.question)
    return QueryResponse(
        answer=result.answer,
        query_type=result.query_type.value,
        num_sources=len(result.chunks),
        sub_question_answers=result.sub_question_answers,
        insufficient_evidence=result.insufficient_evidence,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pipeline_ready": _pipeline is not None,
        "error": _init_error,
    }
