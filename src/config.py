"""Central configuration for the FinRAG pipeline."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (if present) so ANTHROPIC_API_KEY etc. work.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_FILINGS_DIR = DATA_DIR / "raw_filings"
INDEX_DIR = DATA_DIR / "index"
GRAPH_PATH = DATA_DIR / "entity_graph.gpickle"

for d in (RAW_FILINGS_DIR, INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Models ---
# Override with LLM_MODEL env var; default is a valid Anthropic model ID.
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-5")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # dense retrieval
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Chunking ---
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# --- Retrieval ---
TOP_K_DENSE = 15
TOP_K_SPARSE = 15
TOP_K_AFTER_RERANK = 6
HYBRID_ALPHA = 0.5  # weight on dense score vs sparse score when fusing (0=sparse only, 1=dense only)

# --- Self-correction (CRAG-style) ---
RELEVANCE_SCORE_THRESHOLD = 0.6   # below this, trigger fallback
MAX_QUERY_REWRITES = 2

# --- Citation verification ---
CITATION_SUPPORT_THRESHOLD = 0.55  # min entailment-ish score to keep a claim

# --- EDGAR (US variant, kept for reference) ---
EDGAR_USER_AGENT = "FinRAG research-project you@example.com"  # SEC requires a contact UA string
EDGAR_BASE_URL = "https://www.sec.gov"

# --- BSE/NSE (India variant) ---
# There is no official free Indian equivalent to SEC EDGAR. The `bse` package wraps
# BSE's internal (unofficial) JSON endpoints used by bseindia.com itself. It self-throttles
# requests, but there's no documented rate-limit contract -- expect breakage over time and
# code defensively (retries, caching downloaded PDFs so you never refetch).
BSE_DOWNLOAD_DIR = RAW_FILINGS_DIR  # PDFs land here, mirroring the EDGAR raw_filings layout
# Manually-curated PDFs (downloaded by hand from NSE/BSE annual-report archive pages or investor
# relations sites) are the most reliable fallback when the unofficial API breaks or rate-limits you.
MANUAL_PDF_DROP_DIR = RAW_FILINGS_DIR / "manual"
MANUAL_PDF_DROP_DIR.mkdir(parents=True, exist_ok=True)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
