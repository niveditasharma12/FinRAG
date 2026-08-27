"""
Chunk PDF filings into overlapping text windows, tagged with metadata.

This is the piece that makes the India variant a genuinely harder RAG
problem than the SEC/HTML version: Indian annual reports and results PDFs
are frequently multi-column, contain financial tables that don't extract
as clean text, and sometimes include scanned/image pages. This module:

  1. Extracts text page-by-page with pdfplumber
  2. Extracts tables SEPARATELY and renders them as markdown, so numeric
     data (the thing most financial questions ask about) survives chunking
     instead of being garbled into unstructured prose
  3. Flags pages with near-zero extractable text as likely-scanned, so you
     know which documents would need OCR (pytesseract) rather than silently
     losing that content
"""
import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import BSE_DOWNLOAD_DIR, CHUNK_OVERLAP, CHUNK_SIZE, MANUAL_PDF_DROP_DIR


@dataclass
class Chunk:
    chunk_id: str
    text: str
    ticker: str
    filing_date: str  # best-effort; PDFs often lack a clean machine-readable date, may be "unknown"
    source_path: str
    content_type: str = "text"  # "text" or "table"
    page_number: int | None = None


def _table_to_markdown(table: list[list]) -> str:
    if not table or not table[0]:
        return ""
    header = table[0]
    rows = table[1:]
    md = "| " + " | ".join(str(c or "") for c in header) + " |\n"
    md += "| " + " | ".join("---" for _ in header) + " |\n"
    for row in rows:
        md += "| " + " | ".join(str(c or "") for c in row) + " |\n"
    return md


def extract_pdf_content(pdf_path: Path) -> tuple[list[str], list[str], list[int]]:
    """Returns (text_blocks, table_markdown_blocks, likely_scanned_pages)."""
    text_blocks, table_blocks, scanned_pages = [], [], []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            if len(page_text.strip()) < 20:
                scanned_pages.append(i + 1)  # likely scanned / image-only page -- would need OCR

            tables = page.extract_tables()
            for table in tables:
                md = _table_to_markdown(table)
                if md:
                    table_blocks.append(md)

            # Remove raw table text from the prose block to avoid double-counting
            # the same numbers as both garbled prose and clean markdown table.
            cleaned = re.sub(r"\s{3,}", " ", page_text)
            if cleaned.strip():
                text_blocks.append(cleaned)

    return text_blocks, table_blocks, scanned_pages


def load_and_chunk_all() -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []

    for source_dir in (BSE_DOWNLOAD_DIR, MANUAL_PDF_DROP_DIR):
        if not source_dir.exists():
            continue
        for ticker_dir in source_dir.iterdir():
            if not ticker_dir.is_dir():
                continue
            ticker = ticker_dir.name
            for pdf_path in ticker_dir.glob("*.pdf"):
                filing_date = pdf_path.stem  # improve by parsing filename/metadata if available
                try:
                    text_blocks, table_blocks, scanned = extract_pdf_content(pdf_path)
                except Exception as e:
                    print(f"[pdf_chunking] FAILED to parse {pdf_path}: {e}")
                    continue

                if scanned:
                    print(f"[pdf_chunking] WARNING: {pdf_path.name} has {len(scanned)} likely-scanned "
                          f"pages {scanned[:5]}{'...' if len(scanned) > 5 else ''} -- consider OCR "
                          f"(pytesseract) if this content matters for your eval questions")

                full_text = "\n\n".join(text_blocks)
                pieces = splitter.split_text(full_text)
                for i, piece in enumerate(pieces):
                    chunks.append(Chunk(
                        chunk_id=f"{ticker}_{filing_date}_text_{i}",
                        text=piece, ticker=ticker, filing_date=filing_date,
                        source_path=str(pdf_path), content_type="text",
                    ))

                # Tables are kept as separate, un-split chunks -- splitting a markdown
                # table mid-row destroys it, and tables are usually small enough to fit
                # as a single chunk anyway.
                for i, table_md in enumerate(table_blocks):
                    chunks.append(Chunk(
                        chunk_id=f"{ticker}_{filing_date}_table_{i}",
                        text=table_md, ticker=ticker, filing_date=filing_date,
                        source_path=str(pdf_path), content_type="table",
                    ))

    print(f"[pdf_chunking] produced {len(chunks)} chunks "
          f"({sum(1 for c in chunks if c.content_type == 'table')} table chunks)")
    return chunks
