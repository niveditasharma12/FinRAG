"""
FinRAG - Financial RAG Query Interface
Streamlit frontend for querying SEC filings using RAG.

Run with:
    streamlit run app_streamlit.py
"""
import streamlit as st
import requests
import time
import json

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="FinRAG - Financial RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stAlert > div {
        padding: 1rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .scrollable-diagram {
        overflow-x: auto;
        overflow-y: auto;
        max-height: 600px;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        background-color: #fafafa;
    }
    .scrollable-diagram::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    .scrollable-diagram::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 5px;
    }
    .scrollable-diagram::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 5px;
    }
    .scrollable-diagram::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
</style>
""", unsafe_allow_html=True)

# ── Initialize Session State ────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# ── API Configuration ──────────────────────────────────────
API_BASE_URL = "http://localhost:8000"

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=80)
    st.title("FinRAG")
    st.markdown("---")
    
    # Connection status
    st.subheader("Connection Status")
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=5).json()
        if health.get("pipeline_ready"):
            st.success("Pipeline Ready")
        else:
            st.warning("Pipeline Loading...")
    except:
        st.error("Server Offline")
    
    st.markdown("---")
    
    # Statistics
    st.subheader("Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.total_queries)
    with col2:
        st.metric("History", len(st.session_state.history))
    
    st.markdown("---")
    
    # Sample queries
    st.subheader("Sample Queries")
    sample_queries = [
        "What is Apple's revenue?",
        "How does Microsoft's R&D compare to Google's?",
        "What are the main risk factors for Amazon?",
        "Compare Tesla and Ford's profit margins",
    ]
    for query in sample_queries:
        if st.button(query, key=f"sample_{query[:20]}", use_container_width=True):
            st.session_state.query_input = query
            st.rerun()
    
    st.markdown("---")
    st.caption("Powered by LangChain + FAISS")

# ── Main Content ───────────────────────────────────────────
st.markdown('<h1 class="main-header">📊 FinRAG</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Self-correcting multi-hop RAG over SEC filings</p>', unsafe_allow_html=True)

# Query Input
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input(
        "Ask a question about SEC filings",
        placeholder="e.g., What is Apple's R&D spending?",
        key="query_input",
        label_visibility="collapsed",
    )
with col2:
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)

# Process Query
if search_button and query:
    st.session_state.total_queries += 1
    
    # Create progress indicators
    progress_bar = st.progress(0, text="Initializing...")
    status_text = st.empty()
    
    start_time = time.time()
    
    try:
        # Step 1: Routing
        progress_bar.progress(10, text="Routing query...")
        status_text.info("🔄 Classifying query type...")
        
        # Step 2: Retrieval
        progress_bar.progress(30, text="Retrieving relevant documents...")
        status_text.info("📚 Searching through SEC filings...")
        
        # Step 3: Generation
        progress_bar.progress(60, text="Generating answer...")
        status_text.info("🤖 Generating response...")
        
        # Make API call
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": query},
            timeout=600,
        )
        
        if response.status_code == 200:
            result = response.json()
            elapsed = time.time() - start_time
            
            # Complete progress
            progress_bar.progress(100, text="Done!")
            status_text.success(f"✅ Completed in {elapsed:.1f} seconds")
            
            # Store in history
            st.session_state.history.insert(0, {
                "query": query,
                "result": result,
                "time": elapsed,
            })
            
            # Display results
            st.markdown("---")
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Query Type", result["query_type"].upper())
            with col2:
                st.metric("Sources Used", result["num_sources"])
            with col3:
                st.metric("Response Time", f"{elapsed:.1f}s")
            with col4:
                st.metric("Insufficient Evidence", "Yes" if result["insufficient_evidence"] else "No")
            
            st.markdown("---")
            
            # Main Answer
            st.subheader("📝 Answer")
            st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
            
            # Sub-questions (if multi-hop)
            if result.get("sub_question_answers"):
                st.markdown("---")
                st.subheader("🔗 Sub-Questions Decomposition")
                for i, (sub_q, sub_a) in enumerate(result["sub_question_answers"].items(), 1):
                    with st.expander(f"Sub-question {i}: {sub_q}", expanded=False):
                        st.write(sub_a)
            
        else:
            progress_bar.progress(100, text="Error occurred")
            status_text.error(f"Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        progress_bar.progress(100, text="Connection failed")
        status_text.error("❌ Could not connect to the FinRAG server. Make sure it's running on port 8000.")
    except requests.exceptions.Timeout:
        progress_bar.progress(100, text="Request timed out")
        st.error("⏱️ Request timed out. The query might be too complex.")
    except Exception as e:
        progress_bar.progress(100, text="Error occurred")
        st.error(f"❌ An error occurred: {str(e)}")

# ── Query History ──────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Query History")
    
    for i, entry in enumerate(st.session_state.history[:5]):  # Show last 5
        with st.expander(f"Q{i+1}: {entry['query'][:60]}... ({entry['time']:.1f}s)", expanded=False):
            st.write(f"**Query Type:** {entry['result']['query_type']}")
            st.write(f"**Sources:** {entry['result']['num_sources']}")
            st.write("**Answer:**")
            st.write(entry['result']['answer'])
            
            if entry['result'].get('sub_question_answers'):
                st.write("**Sub-questions:**")
                for sub_q, sub_a in entry['result']['sub_question_answers'].items():
                    st.write(f"- {sub_q}")
    
    if st.button("Clear History", type="secondary"):
        st.session_state.history = []
        st.rerun()

# ── About Section ──────────────────────────────────────────
st.markdown("---")
st.header("ℹ️ About FinRAG")

about_tab1, about_tab2, about_tab3 = st.tabs(["How It Works", "Use Cases", "Technology"])

with about_tab1:
    st.markdown("""
    ### How FinRAG Works
    
    FinRAG is a **Self-Correcting Multi-Hop RAG** (Retrieval-Augmented Generation) system 
    designed specifically for financial documents, particularly **SEC 10-K filings**.
    
    #### The Pipeline:
    
    1. **Query Routing** 🔄
       - Classifies your question as simple, multi-hop, or comparison
       - Decomposes complex questions into independent sub-questions
       - Extracts relevant stock tickers (AAPL, MSFT, etc.)
    
    2. **Hybrid Retrieval** 🔍
       - Uses **FAISS** (dense vectors) + **BM25** (keyword matching)
       - Combines both approaches for better recall
       - Reranks results using a cross-encoder model
    
    3. **Self-Correction** ✅
       - Grades each retrieved chunk for relevance
       - Rewrites queries if initial retrieval is poor
       - Prevents hallucination by ensuring evidence exists
    
    4. **Answer Generation** 🤖
       - Generates answers using only verified context
       - Cites specific facts and figures from filings
       - Synthesizes sub-answers for multi-hop questions
    
    5. **Citation Verification** 📋
       - Splits answer into individual claims
       - Verifies each claim against source documents
       - Flags unsupported statements
    """)

with about_tab2:
    st.markdown("""
    ### How FinRAG Helps You
    
    #### 📊 Financial Analysis
    - **Compare companies**: "How does Apple's R&D compare to Microsoft's?"
    - **Track trends**: "What is Tesla's revenue growth over the last 3 years?"
    - **Risk assessment**: "What are Amazon's main risk factors?"
    
    #### 🔍 Research Efficiency
    - **Instant answers**: No need to read hundreds of pages of 10-K filings
    - **Multi-document synthesis**: Combines information from multiple filings
    - **Verified claims**: Ensures answers are grounded in actual filings
    
    #### 🎯 Who Benefits?
    - **Financial analysts**: Quick due diligence and comparison
    - **Investors**: Research companies before investing
    - **Students**: Learn about financial reporting
    - **Journalists**: Fact-check financial claims
    
    #### 💡 Example Use Cases
    
    | Query Type | Example |
    |------------|---------|
    | Simple | "What is Apple's net income?" |
    | Comparison | "Compare Google and Meta's advertising revenue" |
    | Multi-hop | "Which companies mention AI in their risk factors?" |
    | Trend | "How has Microsoft's cloud revenue grown?" |
    """)

with about_tab3:
    st.markdown("""
    ### Technology Stack
    
    | Component | Technology | Purpose |
    |-----------|------------|--------|
    | **LLM** | Groq / Anthropic | Text generation and reasoning |
    | **Embeddings** | BAAI/bge-small-en-v1.5 | Semantic search |
    | **Reranker** | ms-marco-MiniLM-L-6-v2 | Result ranking |
    | **Vector Store** | FAISS | Fast similarity search |
    | **Keyword Search** | BM25 | Traditional text matching |
    | **Orchestration** | LangChain | LLM pipeline management |
    | **Frontend** | Streamlit | User interface |
    | **Backend** | FastAPI | REST API |
    
    #### Key Features:
    - **Self-Correction**: Automatically rewrites queries when retrieval is poor
    - **Multi-Hop Reasoning**: Breaks complex questions into sub-problems
    - **Citation Verification**: Ensures factual accuracy of answers
    - **Hybrid Search**: Combines semantic + keyword search for better results
    """)

# ── Architecture Tab ──────────────────────────────────────
st.header("🏗️ Architecture")

arch_tab1, arch_tab2, arch_tab3 = st.tabs(["System Architecture", "Data Flow", "Component Details"])

with arch_tab1:
    st.markdown("### System Architecture")
    
    # Pure HTML/CSS architecture diagram - fits screen, no Graphviz needed
    arch_html = """
    <style>
    .arch-flow { display:flex; flex-direction:column; align-items:center; gap:0; padding:10px 0; font-family:Arial,sans-serif; }
    .arch-box { border:2px solid #999; border-radius:10px; padding:8px 14px; text-align:center; font-weight:bold; font-size:13px; min-width:200px; max-width:300px; }
    .arch-arrow { font-size:20px; color:#666; line-height:1; }
    .arch-arrow-label { font-size:11px; color:#888; margin-top:-4px; margin-bottom:-4px; }
    .layer-user      { background:#E3F2FD; border-color:#1565C0; color:#1565C0; }
    .layer-frontend   { background:#E8F5E9; border-color:#2E7D32; color:#2E7D32; }
    .layer-api        { background:#FFF3E0; border-color:#E65100; color:#E65100; }
    .layer-pipeline   { background:#FFFDE7; border-color:#F57F17; color:#F57F17; }
    .layer-retrieval  { background:#F3E5F5; border-color:#6A1B9A; color:#6A1B9A; }
    .layer-correction { background:#FFEBEE; border-color:#C62828; color:#C62828; }
    .layer-storage    { background:#E0F2F1; border-color:#00695C; color:#00695C; }
    .layer-llm        { background:#FBE9E7; border-color:#D84315; color:#D84315; border-style:dashed; }
    .arch-row { display:flex; gap:16px; align-items:center; justify-content:center; flex-wrap:wrap; }
    .arch-llm { font-size:11px; padding:4px 10px; border-radius:15px; border:2px dashed #D84315; background:#FBE9E7; color:#D84315; text-align:center; }
    .arch-conn { display:flex; align-items:center; gap:6px; }
    .arch-conn-label { font-size:10px; color:#999; }
    .arch-sublabel { font-size:10px; font-weight:normal; margin-top:2px; }
    .arch-badge { display:inline-block; padding:1px 6px; border-radius:8px; font-size:10px; margin-left:4px; font-weight:normal; }
    </style>
    
    <div class="arch-flow">
        <div class="arch-box layer-user">User (You)</div>
        <div class="arch-arrow">&#9660;</div>
        
        <div class="arch-box layer-frontend">Streamlit Frontend<div class="arch-sublabel">Query Input | Results | History</div></div>
        <div class="arch-arrow">&#9660;</div>
        <div class="arch-arrow-label">HTTP POST</div>
        
        <div class="arch-box layer-api">FastAPI Backend<div class="arch-sublabel">/health | /query</div></div>
        <div class="arch-arrow">&#9660;</div>
        <div class="arch-arrow-label">invoke pipeline</div>
        
        <div class="arch-box layer-pipeline">FinRAG Pipeline<div class="arch-sublabel">Orchestrates entire flow</div></div>
        <div class="arch-arrow">&#9660;</div>
        
        <div class="arch-row">
            <div style="display:flex;flex-direction:column;align-items:center;">
                <div class="arch-box layer-pipeline">1. Query Router<div class="arch-sublabel">Classify | Decompose</div></div>
                <div class="arch-arrow">&#9660;</div>
                <div class="arch-box layer-pipeline">2. Hybrid Retrieval<div class="arch-sublabel">FAISS + BM25</div></div>
                <div class="arch-arrow">&#9660;</div>
                <div class="arch-box layer-pipeline">3. Self-Correction<div class="arch-sublabel">Grade | Rewrite</div></div>
                <div class="arch-arrow">&#9660;</div>
                <div class="arch-box layer-pipeline">4. Answer Generation<div class="arch-sublabel">LLM generates response</div></div>
                <div class="arch-arrow">&#9660;</div>
                <div class="arch-box layer-pipeline">5. Citation Verification<div class="arch-sublabel">Check claims vs sources</div></div>
            </div>
            
            <div style="display:flex;flex-direction:column;align-items:center;gap:6px;">
                <div class="arch-llm">LLM (Groq / Anthropic)<br/><span style="font-size:9px;">Router + Generator + Verifier</span></div>
                <div style="font-size:18px;color:#D84315;">&#8596;</div>
                <div class="arch-llm">Embedding (BGE)<br/><span style="font-size:9px;">Query + Chunk vectors</span></div>
            </div>
        </div>
        
        <div class="arch-arrow">&#9660;</div>
        <div class="arch-arrow-label">search & index</div>
        
        <div class="arch-row">
            <div class="arch-box layer-retrieval" style="min-width:120px;">FAISS<div class="arch-sublabel">Dense vectors</div></div>
            <div class="arch-box layer-retrieval" style="min-width:120px;">BM25<div class="arch-sublabel">Keyword match</div></div>
            <div class="arch-box layer-retrieval" style="min-width:120px;">Fusion<div class="arch-sublabel">Score combine</div></div>
            <div class="arch-box layer-retrieval" style="min-width:120px;">Reranker<div class="arch-sublabel">Top-6 selection</div></div>
        </div>
        
        <div class="arch-arrow">&#9660;</div>
        <div class="arch-arrow-label">load from disk</div>
        
        <div class="arch-row">
            <div class="arch-box layer-storage" style="min-width:120px;">SEC Filings<div class="arch-sublabel">PDF / HTML</div></div>
            <div class="arch-box layer-storage" style="min-width:120px;">Chunks<div class="arch-sublabel">Pickle</div></div>
            <div class="arch-box layer-storage" style="min-width:120px;">Index<div class="arch-sublabel">FAISS + BM25</div></div>
        </div>
    </div>
    """
    import streamlit.components.v1 as components
    components.html(arch_html, height=700, scrolling=True)
    
    # Clickable component details
    st.markdown("---")
    st.markdown("### 🔍 Click to Explore Components")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("🖥️ Streamlit Frontend", expanded=False):
            st.markdown("""
            **Interactive web interface** for querying SEC filings.
            
            - Query input with search button
            - Real-time progress indicators
            - Results display with metrics
            - Query history tracking
            """)
        with st.expander("🌐 FastAPI Backend", expanded=False):
            st.markdown("""
            **REST API** handling HTTP requests.
            
            - `GET /health` - Server status
            - `POST /query` - Submit questions
            - Lazy pipeline initialization
            - Error handling with HTTP codes
            """)
        with st.expander("🔄 Query Router", expanded=False):
            st.markdown("""
            **Classifies and decomposes** user queries.
            
            - `simple` - Single fact lookup
            - `multi_hop` - Chain facts across docs
            - `comparison` - Compare companies
            
            Extracts tickers (AAPL, MSFT...)
            **LLM Calls:** 1 | **Latency:** ~2s
            """)
    
    with col2:
        with st.expander("🔍 Hybrid Retrieval", expanded=False):
            st.markdown("""
            **Dual search** for better recall.
            
            - FAISS: Semantic similarity (BGE)
            - BM25: Keyword matching (TF-IDF)
            - Fusion: 0.5 x dense + 0.5 x sparse
            - Reranker: Cross-encoder MiniLM
            
            Returns top-6 chunks.
            **Latency:** ~1s
            """)
        with st.expander("✅ Self-Correction", expanded=False):
            st.markdown("""
            **Ensures quality** before generation.
            
            - Grade chunks (batch LLM call)
            - Filter by threshold (0.6)
            - Rewrite query if poor quality
            - Retry retrieval (max 2 times)
            
            **83% faster** with batch grading.
            """)
        with st.expander("🤖 LLM & Embedding", expanded=False):
            st.markdown("""
            **External AI services**.
            
            **LLM:** Groq (free) or Anthropic
            - Used for: routing, grading, generation, verification
            - Rate limited: 10s between calls
            
            **Embedding:** BAAI/bge-small-en-v1.5
            - 384-dim vectors for semantic search
            """)

with arch_tab2:
    st.markdown("""
    ### Data Flow Diagram
    
    ```
    USER QUERY
    "What is Apple's R&D spending?"
           │
           ▼
    ┌──────────────────────────────────────────────────────────┐
    │  STEP 1: QUERY ROUTING                                    │
    │  ┌────────────────────────────────────────────────────┐  │
    │  │  Input: "What is Apple's R&D spending?"            │  │
    │  │  Output: {                                          │  │
    │  │    query_type: "simple",                            │  │
    │  │    sub_questions: ["What is Apple's R&D spending?"],│  │
    │  │    tickers_mentioned: ["AAPL"]                      │  │
    │  │  }                                                  │  │
    │  └────────────────────────────────────────────────────┘  │
    └──────────────────────────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────────────────────────┐
    │  STEP 2: RETRIEVAL                                       │
    │  ┌────────────────────────────────────────────────────┐  │
    │  │  2a. FAISS Search (dense vectors)                   │  │
    │  │      - Embed query with BGE model                   │  │
    │  │      - Find top-15 similar chunks                   │  │
    │  │                                                     │  │
    │  │  2b. BM25 Search (keyword matching)                 │  │
    │  │      - Tokenize query                               │  │
    │  │      - Find top-15 matching chunks                  │  │
    │  │                                                     │  │
    │  │  2c. Score Fusion                                   │  │
    │  │      - Combine: 0.5 × dense + 0.5 × sparse         │  │
    │  │      - Rerank with cross-encoder                    │  │
    │  │      - Return top-6 chunks                          │  │
    │  └────────────────────────────────────────────────────┘  │
    └──────────────────────────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────────────────────────┐
    │  STEP 3: SELF-CORRECTION (Optional)                       │
    │  ┌────────────────────────────────────────────────────┐  │
    │  │  3a. Grade each chunk (batch LLM call)              │  │
    │  │      - Score: 0.0 (irrelevant) to 1.0 (relevant)    │  │
    │  │      - Average: 0.85 ✓ (above 0.6 threshold)       │  │
    │  │                                                     │  │
    │  │  3b. Filter low-scoring chunks                      │  │
    │  │      - Keep only chunks with score >= 0.6           │  │
    │  │      - Final: 4 high-quality chunks                 │  │
    │  └────────────────────────────────────────────────────┘  │
    └──────────────────────────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────────────────────────┐
    │  STEP 4: ANSWER GENERATION                                │
    │  ┌────────────────────────────────────────────────────┐  │
    │  │  Context: [Chunk 1] [Chunk 2] [Chunk 3] [Chunk 4]  │  │
    │  │                                                     │  │
    │  │  Prompt:                                            │  │
    │  │  "Answer the question using ONLY the provided       │  │
    │  │   context from SEC filings..."                      │  │
    │  │                                                     │  │
    │  │  Output: "Apple's R&D spending was $29.9B in 2023" │  │
    │  └────────────────────────────────────────────────────┘  │
    └──────────────────────────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────────────────────────┐
    │  STEP 5: CITATION VERIFICATION (Optional)                  │
    │  ┌────────────────────────────────────────────────────┐  │
    │  │  5a. Split answer into claims                       │  │
    │  │      - Claim 1: "Apple's R&D was $29.9B"           │  │
    │  │      - Claim 2: "in fiscal year 2023"              │  │
    │  │                                                     │  │
    │  │  5b. Verify each claim                              │  │
    │  │      - Check against source chunks                 │  │
    │  │      - Score: 0.0 (unsupported) to 1.0 (supported) │  │
    │  │                                                     │  │
    │  │  5c. Flag unsupported claims                        │  │
    │  │      - Add warning if score < 0.55                 │  │
    │  └────────────────────────────────────────────────────┘  │
    └──────────────────────────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────────────────────────┐
    │  FINAL OUTPUT                                            │
    │  ┌────────────────────────────────────────────────────┐  │
    │  │  {                                                  │  │
    │  │    answer: "Apple's R&D spending was $29.9B...",    │  │
    │  │    query_type: "simple",                            │  │
    │  │    num_sources: 6,                                  │  │
    │  │    sub_question_answers: {},                        │  │
    │  │    insufficient_evidence: false                     │  │
    │  │  }                                                  │  │
    │  └────────────────────────────────────────────────────┘  │
    └──────────────────────────────────────────────────────────┘
    ```
    """)

with arch_tab3:
    st.markdown("""
    ### Component Details
    
    #### 🔍 Query Router
    - **Input**: User question
    - **Output**: Query type, sub-questions, tickers
    - **LLM Call**: 1
    - **Latency**: ~2s
    
    #### 📚 Hybrid Retriever
    - **Dense Search**: FAISS with BGE embeddings
    - **Sparse Search**: BM25 with TF-IDF weighting
    - **Fusion**: Weighted sum (α=0.5)
    - **Reranking**: Cross-encoder MiniLM
    - **Latency**: ~1s
    
    #### ✅ Self-Correction
    - **Grading**: Batch LLM call (all chunks at once)
    - **Threshold**: 0.6 relevance score
    - **Rewrites**: Up to 2 query rewrites
    - **Latency**: ~10s (with rate limiting)
    
    #### 🤖 Answer Generator
    - **Context**: Top-6 reranked chunks
    - **Prompt**: SEC-specific instructions
    - **Citations**: Inline source references
    - **Latency**: ~10s (with rate limiting)
    
    #### 📋 Citation Verifier
    - **Claim Splitting**: LLM-based
    - **Entailment Check**: Per-claim verification
    - **Threshold**: 0.55 support score
    - **Latency**: ~10s per claim
    
    ---
    
    #### ⚡ Performance Characteristics
    
    | Metric | Value |
    |--------|-------|
    | Simple Query | ~15s |
    | Multi-hop (2 sub-Qs) | ~30s |
    | Multi-hop (3 sub-Qs) | ~50s |
    | Batch Grading Speedup | 83% |
    | Rate Limit (Groq) | 10s between calls |
    | Max Retries | 10 (exponential backoff) |
    """)

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.caption("FinRAG v1.0 | Built with Streamlit, LangChain, and FAISS")
