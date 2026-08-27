"""
Query routing and decomposition.

Naive RAG treats every question the same: embed it, retrieve top-k, generate.
That breaks on real questions like "How did Apple's R&D as % of revenue
change vs Microsoft from 2021 to 2023?" — a single retrieval pass can't
surface six different numbers across two companies and three years.

This router classifies the query, and for multi-hop/comparison queries,
decomposes it into independent sub-questions that are each retrieved and
answered separately, then combined at generation time.
"""
import json
from dataclasses import dataclass
from enum import Enum

from src.llm import get_llm


class QueryType(str, Enum):
    SIMPLE = "simple"          # single fact lookup, one document
    MULTI_HOP = "multi_hop"    # requires chaining facts (e.g. via entities/relationships)
    COMPARISON = "comparison"  # requires aggregating facts across multiple docs/entities


@dataclass
class RoutedQuery:
    query_type: QueryType
    sub_questions: list[str]
    tickers_mentioned: list[str]


ROUTER_SYSTEM_PROMPT = """You are a query router for a financial-filings RAG system.
Classify the user's question and, if it requires multiple pieces of information,
break it into independent, self-contained sub-questions that can each be answered
by retrieving from a single filing.

Return ONLY valid JSON, no prose, in this exact shape:
{
  "query_type": "simple" | "multi_hop" | "comparison",
  "sub_questions": ["...", "..."],
  "tickers_mentioned": ["AAPL", "MSFT"]
}

Rules:
- "simple": one fact, one document. sub_questions = [original question].
- "multi_hop": answering requires chaining through related entities (e.g. "which of
  Apple's suppliers also appear in Apple's risk factors section?").
- "comparison": requires the same metric from 2+ companies or 2+ time periods.
  Break into one sub-question per (company, period) pair.
- Extract any stock tickers or clearly named companies you can identify.
"""


class QueryRouter:
    def __init__(self):
        self.llm = get_llm()

    def route(self, query: str) -> RoutedQuery:
        response = self.llm.invoke([
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", query),
        ])
        raw = response.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Fail safe: treat as a simple single-shot query rather than crashing the pipeline
            return RoutedQuery(query_type=QueryType.SIMPLE, sub_questions=[query], tickers_mentioned=[])

        return RoutedQuery(
            query_type=QueryType(parsed.get("query_type", "simple")),
            sub_questions=parsed.get("sub_questions") or [query],
            tickers_mentioned=parsed.get("tickers_mentioned") or [],
        )
