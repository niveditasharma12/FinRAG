"""Build and persist the hybrid (dense + sparse) index from chunked filings.

Usage:
    python -m src.ingest.build_index
"""
from src.ingest.chunking import load_and_chunk_all
from src.retrieval.hybrid_retriever import HybridRetriever

if __name__ == "__main__":
    chunks = load_and_chunk_all()
    if not chunks:
        raise SystemExit("No chunks found — run src.ingest.fetch_filings first.")

    retriever = HybridRetriever()
    retriever.build(chunks)
    retriever.save()
    print("[build_index] index saved to data/index/")
