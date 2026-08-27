"""
Fetch corporate filings (financial results, annual reports) for Indian
equities from BSE, since there is no official free Indian equivalent to
SEC EDGAR.

Two paths, use both for resilience:

1. `fetch_via_bse_api()` — uses the `bse` PyPI package (pip install bse),
   which wraps BSE's internal/unofficial JSON endpoints. Good for pulling
   recent financial-results announcements and their PDF attachment URLs
   programmatically. This can break without notice since it's not an
   official, documented API — wrap calls defensively.

2. `register_manual_pdf()` — for annual reports (which are large, formal
   PDFs usually not surfaced cleanly through the announcements feed), the
   most reliable approach is to download them by hand from:
     - NSE: https://www.nseindia.com/companies-listing/corporate-filings-annual-reports
     - BSE: https://www.bseindia.com/corporates/ann.html (search by scrip)
     - Company investor-relations pages
   and drop them into data/raw_filings/manual/<TICKER>/. This function
   just validates and registers them so the rest of the pipeline treats
   them identically to API-fetched filings.
"""
import argparse
import shutil
import time
from pathlib import Path

from src.config import BSE_DOWNLOAD_DIR, MANUAL_PDF_DROP_DIR


def fetch_via_bse_api(scrip_names: list[str], max_announcements: int = 10) -> None:
    """Pull recent financial-results announcements (with PDF links) for each
    company via the unofficial `bse` package, and download the attached PDFs."""
    try:
        from bse import BSE
    except ImportError:
        raise SystemExit("Run: pip install bse")

    with BSE(download_folder=str(BSE_DOWNLOAD_DIR)) as bse:
        for name in scrip_names:
            print(f"[bse_fetch] resolving scrip code for {name}...")
            try:
                scrip_code = bse.getScripCode(name)
            except Exception as e:
                print(f"[bse_fetch] FAILED to resolve {name}: {e}")
                continue

            out_dir = BSE_DOWNLOAD_DIR / name.upper()
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                announcements = bse.announcements(scripcode=scrip_code)
            except Exception as e:
                print(f"[bse_fetch] FAILED to fetch announcements for {name}: {e}")
                continue

            # `announcements` returns a dict/list of recent filings with attachment info;
            # structure varies by package version -- inspect `bse.samples` in the package
            # source if this needs adjusting for your installed version.
            items = announcements.get("Table", announcements) if isinstance(announcements, dict) else announcements
            for item in list(items)[:max_announcements]:
                pdf_name = item.get("ATTACHMENTNAME") or item.get("attachmentname")
                if not pdf_name:
                    continue
                try:
                    pdf_url = f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{pdf_name}"
                    dest = out_dir / pdf_name
                    if dest.exists():
                        continue
                    import requests
                    resp = requests.get(pdf_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
                    resp.raise_for_status()
                    dest.write_bytes(resp.content)
                    print(f"[bse_fetch] saved {dest}")
                except Exception as e:
                    print(f"[bse_fetch] FAILED to download {pdf_name}: {e}")
                time.sleep(0.5)  # be conservative -- this is an unofficial endpoint


def register_manual_pdf(pdf_path: str, ticker: str) -> None:
    """Copy a manually-downloaded annual report / results PDF into the
    pipeline's expected directory structure."""
    src = Path(pdf_path)
    if not src.exists():
        raise FileNotFoundError(pdf_path)
    dest_dir = MANUAL_PDF_DROP_DIR / ticker.upper()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    shutil.copy(src, dest)
    print(f"[manual] registered {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    api_parser = sub.add_parser("api", help="fetch recent announcements via unofficial BSE API")
    api_parser.add_argument("--companies", nargs="+", required=True, help="e.g. TCS INFY RELIANCE")

    manual_parser = sub.add_parser("manual", help="register a manually-downloaded PDF")
    manual_parser.add_argument("--pdf", required=True)
    manual_parser.add_argument("--ticker", required=True)

    args = parser.parse_args()
    if args.mode == "api":
        fetch_via_bse_api(args.companies)
    elif args.mode == "manual":
        register_manual_pdf(args.pdf, args.ticker)
