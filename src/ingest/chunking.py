"""
Chunk raw filing text into overlapping windows, tagged with metadata
(ticker, filing_date) so retrieval results can be filtered/grouped and
citations can point back to a specific document.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, RAW_FILINGS_DIR


@dataclass
class Chunk:
    chunk_id: str
    text: str
    ticker: str
    filing_date: str
    source_path: str


def clean_text(raw: str) -> str:
    """Strip excessive whitespace/boilerplate noise common in EDGAR HTML dumps."""
    text = re.sub(r"\n{3,}", "\n\n", raw)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Drop page-number-only lines and repeated table-of-contents artifacts
    lines = [l for l in text.split("\n") if not re.fullmatch(r"\s*\d{1,4}\s*", l)]
    return "\n".join(lines).strip()


def load_and_chunk_all() -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []

    for ticker_dir in RAW_FILINGS_DIR.iterdir():
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        for txt_path in ticker_dir.glob("*.txt"):
            filing_date = txt_path.stem
            raw = txt_path.read_text(encoding="utf-8", errors="ignore")
            cleaned = clean_text(raw)
            pieces = splitter.split_text(cleaned)
            for i, piece in enumerate(pieces):
                chunks.append(Chunk(
                    chunk_id=f"{ticker}_{filing_date}_{i}",
                    text=piece,
                    ticker=ticker,
                    filing_date=filing_date,
                    source_path=str(txt_path),
                ))
    print(f"[chunking] produced {len(chunks)} chunks from {RAW_FILINGS_DIR}")
    return chunks
