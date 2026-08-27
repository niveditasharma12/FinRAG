# FinRAG — Self-Correcting Multi-Hop RAG over SEC Filings

An agentic RAG system that answers questions over SEC 10-K/10-Q filings, built to
demonstrate production-grade RAG techniques beyond naive "embed + retrieve + generate":

- **Hybrid retrieval**: dense (embeddings) + sparse (BM25), fused and reranked with a cross-encoder
- **Query routing**: classifies queries as simple lookup / multi-hop / comparison, decomposes complex ones
- **GraphRAG**: entity/relationship graph for questions vector search alone can't answer
- **Self-correction (CRAG-style)**: grades retrieved chunks for relevance, falls back (rewrite query / say "I don't know") when retrieval is weak
- **Citation verification**: every claim is checked against its source chunk before being returned
- **Evaluation harness**: RAGAS-based faithfulness / relevancy / precision metrics, so you can report real before/after numbers

## Architecture

```
                        ┌─────────────────┐
  User query ─────────► │  Query Router     │
                        │  (classify+decompose)
                        └────────┬─────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                                     ▼
     ┌─────────────────┐                  ┌──────────────────┐
     │ Hybrid Retriever │                  │   Graph Query     │
     │ (BM25 + Dense)   │                  │  (multi-hop only) │
     └────────┬─────────┘                  └─────────┬────────┘
              ▼                                       │
     ┌─────────────────┐                              │
     │ Cross-Encoder    │◄─────────────────────────────┘
     │ Reranker         │
     └────────┬─────────┘
              ▼
     ┌─────────────────┐
     │ Relevance Grader │──── weak? ──► Fallback (rewrite / web / abstain)
     │ (Self-RAG/CRAG)  │
     └────────┬─────────┘
              ▼
     ┌─────────────────┐
     │ LLM Generation   │
     └────────┬─────────┘
              ▼
     ┌─────────────────┐
     │ Citation Verifier│──── unsupported claim? ──► strip / flag
     └────────┬─────────┘
              ▼
          Final Answer + Citations
```

## India variant: BSE/NSE instead of SEC EDGAR

There is **no free, official, documented Indian equivalent of SEC EDGAR**. This is a
real constraint worth understanding rather than papering over — see below for what's
actually available and how this scaffold handles it.

| Source | What you get | Catch |
|---|---|---|
| `bse` package (unofficial) | Recent announcements + PDF links, via BSE's internal JSON endpoints | Not documented/versioned by BSE; can break without notice |
| NSE/BSE annual-report archive pages | Direct PDF links per company | No search API — largely manual |
| MCA21 | The legal system-of-record for every filing | Paid, per-document, not scrapable at scale for free |

**This scaffold uses two ingestion paths, meant to be combined:**
1. `src/ingest/fetch_indian_filings.py api --companies TCS INFY RELIANCE` — pulls recent
   results/announcements via the unofficial `bse` package.
2. `src/ingest/fetch_indian_filings.py manual --pdf path/to/annual_report.pdf --ticker TCS`
   — registers a hand-downloaded annual report PDF (the more reliable route for full annual
   reports, which the announcements feed doesn't surface cleanly).

**Why this is a stronger RAG problem, not a weaker one:** Indian filings are PDF-native —
multi-column layouts, financial tables that don't extract as clean text, occasional scanned
pages. `src/ingest/pdf_chunking.py` handles this by extracting tables separately as markdown
(so numbers survive chunking instead of getting garbled into prose) and flagging likely-scanned
pages so you know where OCR (pytesseract) would be needed rather than silently losing content.
This is exactly the kind of "the data is messier than the demo" problem that separates a real
RAG project from a tutorial one — worth narrating explicitly in an interview.

```bash
python -m src.ingest.fetch_indian_filings api --companies TCS INFY RELIANCE
python -m src.ingest.build_index_india
```

## Setup

```bash
python -m venv venv && source venv/bin/activate   # Windows (Git Bash): source venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env   # then put your real key in .env
export ANTHROPIC_API_KEY=your_key_here  # or just rely on .env
```

> **Windows note:** some pip wheel builds are blocked by Windows Smart App Control /
> WDAC policies. The pins in `requirements.txt` (`faiss-cpu==1.13.0`, `tiktoken==0.12.0`,
> `scipy==1.16.3`) are known-good builds; if a different version fails with
> "An Application Control policy has blocked this file", pin another build of that package.

## Project layout

```
src/
  ingest/          # EDGAR fetching, chunking
  retrieval/        # hybrid_retriever.py, reranker.py, query_router.py
  graph/             # entity extraction, networkx graph, graph query
  correction/        # relevance grading + fallback logic
  verification/      # citation grounding checker
  eval/               # RAGAS harness
  app.py              # FastAPI orchestrator
```

## Quickstart

```bash
python -m src.ingest.fetch_filings --tickers AAPL MSFT GOOGL --years 2021 2022 2023
python -m src.ingest.build_index
uvicorn src.app:app --reload
# POST /query {"question": "How did Apple's R&D spend as % of revenue change vs Microsoft from 2021 to 2023?"}
```

## The interview story

> "I built a RAG system over SEC filings and instrumented it with RAGAS. Naive
> retrieval scored 0.61 faithfulness with a 23% hallucination rate on multi-hop
> questions. Adding query decomposition + self-correction + citation verification
> brought faithfulness to 0.89 and cut hallucinations to 4%. Here's the eval report."

That's a concrete, numbers-backed story — plug in your own measured results once you run `src/eval/harness.py`.
