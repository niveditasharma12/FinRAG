# FinRAG — Self-Correcting Multi-Hop RAG over SEC Filings

A production-grade RAG system that answers questions over SEC 10-K/10-Q filings, featuring self-correction, multi-hop reasoning, and citation verification.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red)

---

## Demo

<!-- Replace these with actual screenshots. Capture with: -->
<!-- 1. Run the app: streamlit run app_streamlit.py -->
<!-- 2. Take screenshots of each query example below -->
<!-- 3. Save as images/demo-simple.png, images/demo-comparison.png, etc. -->

### Simple Query

<!-- ![Simple Query Demo](images/demo-simple.png) -->

```
Query: "What is Apple's R&D spending?"
```

**Result:**
```
Query Type:    SIMPLE
Sources:       6 chunks
Response Time: 15.3s

Answer:
Apple's research and development expense (R&D spending) reported in the filings is:

* 2022: $26,251 million
* 2021: $21,914 million
* 2020: $18,752 million

These figures appear in the 2022 Form 10-K under "Research and development expense"
for each year.
```

---

### Multi-Hop Comparison Query

<!-- ![Multi-Hop Query Demo](images/demo-comparison.png) -->

```
Query: "How does Apple's R&D spending compare to Microsoft's R&D spending?"
```

**Result:**
```
Query Type:    COMPARISON
Sub-Questions: 2
Sources:       12 chunks
Response Time: 31.3s

Sub-Question 1: "What is Apple's R&D spending?"
→ Apple invested $29.9B in R&D in 2023...

Sub-Question 2: "What is Microsoft's R&D spending?"
→ Microsoft spent $27.2B on R&D in 2023...

Answer:
Apple's R&D spending was $29.9B (7.6% of revenue) while Microsoft
spent $27.2B (12.8% of revenue). Microsoft invests a higher percentage
of revenue in R&D despite lower absolute spending.
```

---

### Streamlit UI Demo

<!-- ![Streamlit UI](images/demo-ui.png) -->

The Streamlit frontend provides:

| Feature | Description |
|---------|-------------|
| Query Input | Text field with search button |
| Progress Indicators | Real-time status during query |
| Results Display | Answer with metrics and citations |
| Query History | Track previous queries |
| Architecture Diagram | Interactive system visualization |

**Start the UI:**
```bash
# Terminal 1: Backend
uvicorn src.app:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
streamlit run app_streamlit.py
```

Open http://localhost:8501

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Hybrid Retrieval** | FAISS (dense vectors) + BM25 (keywords), fused and reranked |
| **Query Routing** | Classifies simple / multi-hop / comparison queries automatically |
| **Self-Correction** | Grades retrieval quality, rewrites queries when results are weak |
| **Batch Grading** | Grades all chunks in 1 LLM call (83% faster than individual) |
| **Citation Verification** | Checks every claim against source documents |
| **Rate Limiting** | Smart retry logic for Groq free tier (8K TPM) |
| **Streamlit UI** | Interactive frontend with architecture visualization |

## Architecture

```
User Question
      |
      v
+------------------+
|  Query Router    |  Classify: simple / multi_hop / comparison
|  (LLM)          |  Decompose into sub-questions
+--------+---------+
         |
         v
+------------------+
| Hybrid Retriever |  FAISS (semantic) + BM25 (keywords)
|                  |  Score Fusion + Cross-Encoder Reranking
+--------+---------+
         |
         v
+------------------+
| Self-Correction  |  Grade chunks (batch LLM call)
|                  |  Filter low-relevance results
|                  |  Rewrite query if needed (max 2 retries)
+--------+---------+
         |
         v
+------------------+
| Answer Generator |  Generate with SEC-specific prompt
|  (LLM)          |  Inline citations [TICKER DATE]
+--------+---------+
         |
         v
+------------------+
| Citation Check   |  Split answer into claims
|                  |  Verify each against source chunks
|                  |  Flag unsupported claims
+--------+---------+
         |
         v
    Final Answer
```

---

## Quick Start

### 1. Clone & Setup

```bash
# Clone the repo
git clone https://github.com/niveditasharma12/FinRAG.git
cd FinRAG

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy the example env file
cp .env.example .env
```

Edit `.env` and add your API key:

```env
# Option 1: Groq (free tier, fast)
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here

# Option 2: Anthropic (paid, higher quality)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Get a free Groq API key at: https://console.groq.com/keys

### 3. Fetch SEC Filings & Build Index

```bash
# Fetch filings for companies (first time only)
python -m src.ingest.fetch_filings --tickers AAPL MSFT GOOGL --years 2021 2022 2023

# Build the search index
python -m src.ingest.build_index
```

### 4. Run the Application

**Option A: Streamlit Frontend (recommended)**

```bash
# Terminal 1: Start the backend
uvicorn src.app:app --host 0.0.0.0 --port 8000

# Terminal 2: Start the frontend
streamlit run app_streamlit.py
```

Open http://localhost:8501 in your browser.

**Option B: FastAPI Only**

```bash
uvicorn src.app:app --reload
```

Test with curl:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Apple'\''s R&D spending?"}'
```

---

## Taking Screenshots for READM
### Simple query input and result
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a87e6495-6e08-41f4-8c85-34e06b901889" />

### Multi-hop query with decomposition

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/5f7c297d-7bd5-4e46-8084-75a881e4a5c2" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/dfb624ba-c84b-4ccb-a6e9-c4a3f0027a06" />


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/cd204ed0-e75c-444d-ada0-46c1b278de39" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/45f353f9-7c86-4e23-b571-4cfb83cd5763" />

### Architecture diagram (fully visible)
<img width="1402" height="812" alt="image" src="https://github.com/user-attachments/assets/1f7f6123-6206-4d72-b08e-8303e685b753" />
<img width="1435" height="335" alt="image" src="https://github.com/user-attachments/assets/d30abb11-4986-41e6-89c2-93412cdbdecd" />

### Query history panel
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f90550d3-20ac-4a75-9d35-1ac3ace93133" />

---

## Example Queries

| Type | Example |
|------|---------|
| **Simple** | "What is Apple's revenue?" |
| **Comparison** | "How does Microsoft's R&D compare to Google's?" |
| **Multi-hop** | "Which companies mention AI in their risk factors?" |
| **Trend** | "How has Tesla's profit margin changed over 3 years?" |

---

## Project Structure

```
FinRAG/
├── app_streamlit.py          # Streamlit frontend
├── requirements.txt          # Python dependencies
├── .env.example              # API key template
│
├── src/
│   ├── app.py                # FastAPI backend
│   ├── config.py             # Central configuration
│   ├── llm.py                # LLM factory with rate limiting
│   ├── pipeline.py           # Main RAG orchestration
│   │
│   ├── retrieval/
│   │   ├── hybrid_retriever.py   # FAISS + BM25 search
│   │   ├── reranker.py           # Cross-encoder reranking
│   │   └── query_router.py       # Query classification
│   │
│   ├── correction/
│   │   └── self_rag.py           # Self-correction with batch grading
│   │
│   ├── verification/
│   │   └── citation_check.py     # Citation verification
│   │
│   ├── ingest/
│   │   ├── fetch_filings.py      # SEC EDGAR fetcher
│   │   ├── build_index.py        # Index builder
│   │   └── chunking.py           # Document chunking
│   │
│   └── eval/
│       └── harness.py            # RAGAS evaluation
│
└── data/
    ├── raw_filings/               # Downloaded SEC filings
    └── index/                     # Built search indexes
        ├── dense.index            # FAISS index
        ├── sparse.pkl             # BM25 index
        └── chunks.pkl             # Chunk metadata
```

---

## Configuration

Key settings in `src/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 2000 | Characters per chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `TOP_K_DENSE` | 15 | FAISS results |
| `TOP_K_SPARSE` | 15 | BM25 results |
| `TOP_K_AFTER_RERANK` | 6 | Final chunks for generation |
| `HYBRID_ALPHA` | 0.5 | Dense vs sparse weight |
| `RELEVANCE_SCORE_THRESHOLD` | 0.6 | Self-correction threshold |
| `MAX_QUERY_REWRITES` | 2 | Max query rewrite attempts |

---

## Performance

| Query Type | Response Time | LLM Calls |
|------------|---------------|-----------|
| Simple | ~15s | 2-3 |
| Multi-hop (2 sub-Qs) | ~30s | 4-6 |
| Multi-hop (3 sub-Qs) | ~50s | 6-8 |

**Optimizations applied:**
- Batch chunk grading: 83% faster retrieval quality checks
- Rate limit reduction: 65s → 10s between LLM calls
- Groq free tier support with automatic retry on 429 errors

---

## API Reference

### `POST /query`

```json
// Request
{"question": "What is Apple's R&D spending?"}

// Response
{
  "answer": "Apple's R&D spending was $29.9B in 2023...",
  "query_type": "simple",
  "num_sources": 6,
  "sub_question_answers": {},
  "insufficient_evidence": false
}
```

### `GET /health`

```json
{"status": "ok", "pipeline_ready": true, "error": null}
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| LLM | Groq / Anthropic | Text generation |
| Embeddings | BAAI/bge-small-en-v1.5 | Semantic search |
| Reranker | ms-marco-MiniLM-L-6-v2 | Result ranking |
| Vector Store | FAISS | Fast similarity search |
| Keyword Search | BM25 | Traditional text matching |
| Orchestration | LangChain | LLM pipeline management |
| Backend | FastAPI | REST API |
| Frontend | Streamlit | User interface |

---

## India Variant (BSE/NSE)

For Indian filings, use:

```bash
python -m src.ingest.fetch_indian_filings api --companies TCS INFY RELIANCE
python -m src.ingest.build_index_india
```

---

## Evaluation

Run the RAGAS evaluation harness:

```bash
python -m src.eval.harness
```

This measures faithfulness, relevancy, and precision metrics.

---

## License

MIT License

---

Built with ❤️ for financial research
