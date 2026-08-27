"""
Fetch 10-K filings from SEC EDGAR for a list of tickers/years.

SEC EDGAR is free, requires no auth, but demands a descriptive User-Agent header
(company/contact string) or it will 403 you. See config.EDGAR_USER_AGENT.

Uses the structured submissions JSON API (data.sec.gov), which exposes each
filing's `primaryDocument` filename directly -- much more reliable than
scraping the HTML filing index, whose first "10-K" row now points at an
XBRL viewer stub rather than the actual document.
"""
import argparse
import json
import time

import requests

from src.config import EDGAR_BASE_URL, EDGAR_USER_AGENT, RAW_FILINGS_DIR

HEADERS = {"User-Agent": EDGAR_USER_AGENT}


def get_cik_for_ticker(ticker: str) -> str:
    """Map a stock ticker to its 10-digit zero-padded CIK number."""
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    mapping = resp.json()
    for entry in mapping.values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker {ticker} not found in EDGAR ticker map")


def list_10k_filings(cik: str, years: list[int]) -> list[dict]:
    """Return metadata for 10-K filings matching the requested filing years."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    recent = resp.json()["filings"]["recent"]

    filings = []
    for form, accn, filing_date, primary_doc in zip(
        recent["form"], recent["accessionNumber"],
        recent["filingDate"], recent["primaryDocument"],
    ):
        if form != "10-K" or int(filing_date[:4]) not in years:
            continue
        accession_nodash = accn.replace("-", "")
        filings.append({
            "accession": accession_nodash,
            "filing_date": filing_date,
            "primary_document": primary_doc,
            # Full document URL, e.g. .../Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm
            "href": (
                f"{EDGAR_BASE_URL}/Archives/edgar/data/{int(cik)}/"
                f"{accession_nodash}/{primary_doc}"
            ),
        })
    return filings


def download_filing_text(doc_url: str) -> str:
    """Download the filing text, preferring the .txt full-text version.

    SEC EDGAR now serves inline XBRL viewer HTML as the primaryDocument,
    which contains no readable text when scraped.  Instead, we look for the
    companion .txt filing in the same accession directory (e.g.
    0000320193-21-000105.txt) which contains the full plaintext filing.
    """
    import re

    from bs4 import BeautifulSoup

    # Derive the accession directory URL from the primary document URL.
    # e.g. https://www.sec.gov/Archives/edgar/data/320193/000032019321000105/aapl-20210925.htm
    #   -> https://www.sec.gov/Archives/edgar/data/320193/000032019321000105/
    dir_url = doc_url.rsplit("/", 1)[0] + "/"

    # 1. Try to find and download the .txt full-text filing from the index
    try:
        resp = requests.get(dir_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        txt_href = None
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if href.endswith(".txt") and not href.endswith(".hdr.sgml"):
                # Prefer the accession-number .txt file (e.g. 0000320193-21-000105.txt)
                txt_href = href
                break
        if txt_href:
            if txt_href.startswith("/"):
                txt_url = "https://www.sec.gov" + txt_href
            elif txt_href.startswith("http"):
                txt_url = txt_href
            else:
                txt_url = dir_url + txt_href
            resp2 = requests.get(txt_url, headers=HEADERS, timeout=60)
            resp2.raise_for_status()
            raw_text = resp2.text
            # Strip HTML/XBRL tags — the .txt filing contains inline XBRL
            soup2 = BeautifulSoup(raw_text, "html.parser")
            for tag in soup2(["script", "style", "noscript"]):
                tag.decompose()
            text = soup2.get_text(separator="\n")
            text = re.sub(r"\n{3,}", "\n\n", text)
            if len(text.strip()) > 5000:
                return text
    except Exception:
        pass  # fall through to primary document

    # 2. Fallback: download the primary document and extract text
    resp = requests.get(doc_url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    # If the result is too short, it's probably an XBRL viewer stub — try .txt fallback
    if len(text.strip()) < 5000:
        try:
            resp2 = requests.get(dir_url, headers=HEADERS, timeout=30)
            resp2.raise_for_status()
            soup2 = BeautifulSoup(resp2.content, "html.parser")
            for a_tag in soup2.find_all("a", href=True):
                href = a_tag["href"]
                if href.endswith(".txt") and not href.endswith(".hdr.sgml"):
                    txt_url2 = (
                        "https://www.sec.gov" + href if href.startswith("/")
                        else href if href.startswith("http") else dir_url + href
                    )
                    resp3 = requests.get(txt_url2, headers=HEADERS, timeout=60)
                    resp3.raise_for_status()
                    soup3 = BeautifulSoup(resp3.text, "html.parser")
                    for t in soup3(["script", "style", "noscript"]):
                        t.decompose()
                    text = re.sub(r"\n{3,}", "\n\n", soup3.get_text(separator="\n"))
                    if len(text.strip()) > 5000:
                        return text
                    break
        except Exception:
            pass
    return text


def fetch_all(tickers: list[str], years: list[int]) -> None:
    for ticker in tickers:
        print(f"[fetch] {ticker}: resolving CIK...")
        cik = get_cik_for_ticker(ticker)
        filings = list_10k_filings(cik, years)
        print(f"[fetch] {ticker}: found {len(filings)} filings for years {years}")

        for filing in filings:
            out_dir = RAW_FILINGS_DIR / ticker
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{filing['filing_date']}.txt"
            if out_path.exists():
                continue
            try:
                text = download_filing_text(filing["href"])
                if len(text.strip()) < 5000:
                    raise RuntimeError(
                        f"downloaded document suspiciously small ({len(text)} chars)"
                    )
                out_path.write_text(text, encoding="utf-8")
                meta_path = out_dir / f"{filing['filing_date']}.meta.json"
                meta_path.write_text(json.dumps(filing, indent=2))
                print(f"[fetch] saved {out_path} ({len(text):,} chars)")
            except Exception as e:
                print(f"[fetch] FAILED {ticker} {filing['filing_date']}: {e}")
            time.sleep(0.3)  # be polite to EDGAR's rate limits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--years", nargs="+", type=int, required=True)
    args = parser.parse_args()
    fetch_all(args.tickers, args.years)
