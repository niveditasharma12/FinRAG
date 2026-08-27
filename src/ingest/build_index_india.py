"""Build the hybrid index from Indian PDF filings (BSE announcements + manual PDFs).

Usage:
    python -m src.ingest.build_index_india
"""
from src.ingest.pdf_chunking import load_and_chunk_all
from src.retrieval.hybrid_retriever import HybridRetriever

if __name__ == "__main__":
    chunks = load_and_chunk_all()
    if not chunks:
        raise SystemExit(
            "No chunks found. Either run:\n"
            "  python -m src.ingest.fetch_indian_filings api --companies TCS INFY RELIANCE\n"
            "or drop PDFs manually into data/raw_filings/manual/<TICKER>/ and run:\n"
            "  python -m src.ingest.fetch_indian_filings manual --pdf path/to.pdf --ticker TCS"
        )

    retriever = HybridRetriever()
    retriever.build(chunks)
    retriever.save()
    print("[build_index_india] index saved to data/index/")
