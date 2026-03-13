# TECH_STACK — Technology Stack & Decisions



## Technology Decision Summary

| Layer | Component | Choice | Alternatives Considered |
|-------|----------|--------|------------------------|
| Frontend | UI Framework | **Streamlit** | Gradio, FastAPI + Vue |
| Orchestration | Agent Framework | **LangGraph** | CrewAI, AutoGen |
| LLM | Chat Model | **DeepSeek-V3** | GPT-4o-mini, Claude 3.5 Haiku |
| RAG | Vector Store | **ChromaDB** | FAISS, Weaviate, Pinecone |
| RAG | Embedding Model | **bge-small-zh-v1.5** | text-embedding-3-small, m3e-base |
| RAG | PDF Parser | **PyMuPDF (fitz)** | pdfplumber, pypdf |
| Search | Web Search API | **Tavily** | DuckDuckGo, SerpAPI, Bing |
| Deployment | Primary Platform | **HuggingFace Spaces** | Streamlit Cloud, Railway, Render |
| Deployment | Fallback Platform | **Streamlit Cloud** | — |



## Decision Records

### DR-1: LLM — DeepSeek-V3

**Decision**: Use DeepSeek-V3 as the primary LLM.

**Rationale**:
- Exceptional Chinese language performance (critical for Gaokao context)
- Fully compatible with OpenAI API format — `langchain-openai`'s `ChatOpenAI` works out of the box, only need to change `base_url`
- Extremely low cost: ~¥1 per million input tokens
- Stable access from mainland China (no proxy needed)

**Usage**:
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    temperature=0.7,
    streaming=True,
)
```



### DR-2: Vector Store — ChromaDB

**Decision**: Use ChromaDB over FAISS.

**Rationale**:
- **Native metadata filtering**: Query with `subject="math"` / `year=2024` directly — FAISS requires manual post-processing
- **Built-in persistence**: `persist_directory` parameter handles storage automatically — no manual pickle management
- **HuggingFace Spaces compatibility**: FAISS has known pickle dependency conflicts on HF Spaces
- **Streamlit integration**: Community-maintained `streamlit-chromadb-connection` package available

**Usage**:
```python
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma(
    collection_name="gaokao_docs",
    embedding_function=embedding,
    persist_directory="chroma_store/",
)
```



### DR-3: Embedding — bge-small-zh-v1.5

**Decision**: Use `bge-small-zh-v1.5` for local embedding.

**Rationale**:
- **Completely free**: Open-source model, local inference, zero API cost
- **Lightweight**: ~90MB — well within HuggingFace Spaces' 16GB RAM
- **Top Chinese performance**: Ranks among the best on Chinese MTEB benchmarks
- **512-dim vectors**: Moderate dimensionality, efficient storage and retrieval

**Fallback**: If local resources are tight, switch to DeepSeek Embedding API (also low cost).



### DR-4: Web Search — Tavily

**Decision**: Use Tavily Search API as the required web search component.

**Rationale**:
- LangChain **first-class integration**: `TavilySearchResults` works out of the box
- **Free tier**: 1000 requests/month, sufficient for MVP demo
- Returns structured results (title, url, content), easy to inject into prompts
- Optimized for RAG/AI use cases, filters ads and low-quality pages

**Usage**:
```python
from langchain_community.tools.tavily_search import TavilySearchResults

search_tool = TavilySearchResults(
    max_results=3,
    api_key=os.environ["TAVILY_API_KEY"],
)
```



### DR-5: Deployment — HuggingFace Spaces (Primary)

**Decision**: Use HuggingFace Spaces as primary deployment over Streamlit Cloud.

**Rationale**:

| Metric | HuggingFace Spaces | Streamlit Cloud |
|--------|-------------------|----------------|
| Free RAM | **16 GB** | 1 GB |
| Community exposure | HF community, AI-focused audience | General |
| Streamlit support | Native Streamlit SDK | Native |
| Secrets management | HF Space Secrets panel | Streamlit Secrets |

16GB RAM is the decisive factor: bge-small-zh (~90MB) + ChromaDB + LangGraph fit comfortably.



### DR-6: No External Reactor Framework

**Decision**: Do NOT add Reactor / event-bus middleware between Streamlit and LangGraph.

**Rationale**:
- LangGraph `stream()` + conditional edges = Reactor core pattern (event dispatcher + routing table + callbacks)
- An extra framework (e.g., `rxpy`, `pyee`) only adds complexity in a 24h sprint
- A thin adapter function (~30 lines) is sufficient to bridge LangGraph events → Streamlit UI



## Dependency List

```txt
# Core Framework
streamlit>=1.40.0
langchain>=0.3.0
langchain-openai>=0.2.0            # DeepSeek-V3 via OpenAI-compatible API
langchain-community>=0.3.0         # TavilySearchResults, ChromaDB wrapper
langchain-huggingface>=0.1.0       # HuggingFaceEmbeddings for bge
langgraph>=0.2.0

# RAG
chromadb>=0.5.0
pymupdf>=1.25.0                    # PDF parsing (fitz)
sentence-transformers>=3.0.0       # Run bge-small-zh-v1.5 locally

# Web Search
tavily-python>=0.5.0

# Utilities
python-dotenv>=1.0.0
```



## Environment Variables

```bash
# .env.example

# Required
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
TAVILY_API_KEY=your_tavily_api_key_here

# Optional (sensible defaults)
CHROMA_PERSIST_DIR=chroma_store/
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
DEEPSEEK_MODEL=deepseek-chat
```
