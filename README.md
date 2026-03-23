# Gaokao Tutor

<p align="center"><a href="README_zh.md">中文文档</a></p>

A multi-agent conversational AI that helps high school students prepare for the Chinese College Entrance Examination (Gaokao). Built with **LangGraph** (orchestration), **FastAPI** (SSE streaming backend), and **Next.js** (reactive frontend). It routes user queries through a supervisor to provide subject tutoring, personalized study plans, and emotional support.

## Core Features

- **Academic Q&A** — Parallel RAG + web search retrieval (fan-out/fan-in), step-by-step solutions with source citations, hallucination evaluation with automatic retry loop
- **Study Planner** — Generates personalized revision schedules enriched with the latest Gaokao policy data
- **Emotional Support** — Empathetic responses in the persona of an experienced homeroom teacher
- **Intent Routing** — LLM-based supervisor classifies user intent and dispatches to the appropriate workflow
- **LLM Fallback** — Automatic failover to a backup LLM provider when the primary API times out or returns 5xx
- **Distributed Tracing** — OpenTelemetry instrumentation with Jaeger export and SQLite fallback
- **State Persistence** — PostgreSQL-backed checkpointer for multi-turn conversation memory
- **Configuration-Driven** — YAML settings + XML prompt registry; swap behavior without code changes
- **Real-Time Observability** — Frontend reasoning path visualization and system logs from SSE node lifecycle events
- **Markdown Rendering** — Assistant responses rendered with full GFM markdown support (tables, code blocks, lists)

## Tech Stack

| Layer | Choice | Purpose |
| ----- | ------ | ------- |
| Frontend | Next.js 16 + Tailwind CSS 4 | Reactive chat UI with SSE streaming |
| Backend API | FastAPI + Uvicorn | SSE streaming endpoint (`POST /stream`) |
| Orchestration | LangGraph | Multi-agent state graph with fan-out/fan-in |
| LLM | DeepSeek-V3 (+ configurable fallback) | Chat completion |
| Vector Store | ChromaDB | Local knowledge retrieval (L2 → normalized relevance) |
| Embedding | BAAI/bge-m3 via SiliconFlow | Text vectorization |
| PDF Parser | PyMuPDF | Document ingestion |
| Web Search | DuckDuckGo | Online fallback retrieval |
| State Persistence | PostgreSQL (psycopg) | Multi-turn conversation memory via LangGraph checkpointer |
| Observability | OpenTelemetry + Jaeger + SQLite | Distributed tracing |
| Configuration | YAML + XML | Runtime settings and prompt templates |

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ and npm
- [Conda](https://docs.conda.io/) (recommended)
- PostgreSQL (optional, for state persistence)
- Jaeger 2.x (optional, for trace visualization)

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/gaokao_tutor.git
cd gaokao_tutor

# Create and activate conda environment
conda create -n gaokao_tutor python=3.11 -y
conda activate gaokao_tutor

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys:
#   DEEPSEEK_API_KEY=<your-key>
#   SILICONFLOW_API_KEY=<your-key>
#   DB_URI=postgresql://postgres:<password>@localhost:5432/gaokao_tutor  (optional)
```

### Build Knowledge Base

```bash
python scripts/build_index.py
```

### Frontend Setup

```bash
cd frontend
npm install
```

### Run

```bash
# Terminal 1 — Backend (SSE streaming API)
uvicorn app:app --reload --port 8000

# Terminal 2 — Frontend (Next.js dev server)
cd frontend
npm run dev
```

The frontend runs at `http://localhost:3000`, the backend API at `http://localhost:8000`.

### Optional: Jaeger Tracing

```bash
# Start Jaeger (traces visible at http://localhost:16686)
./local_tools/jaeger-2.15.0-windows-amd64/jaeger.exe
```

## Architecture

```text
User ──► Next.js (SSE consumer) ──► FastAPI /stream ──► LangGraph StateGraph
                                                            │
                                          ┌─────────────────┼─────────────────┐
                                          ▼                 ▼                 ▼
                                     Academic          Planner          Emotional
                                   ┌────┴────┐           │                  │
                                   ▼         ▼           ▼                  ▼
                              rag_retrieve  web_search  search_policy   emotional_response
                                   └────┬────┘           │
                                        ▼                ▼
                                  generate_answer    generate_plan
                                        │
                                        ▼
                                evaluate_hallucination
                                   (retry loop)
```

## Project Structure

```text
gaokao_tutor/
├── app.py                        # FastAPI SSE streaming endpoint
├── config/
│   ├── settings.yaml             # Runtime parameters (temperatures, timeouts, retries)
│   └── prompts/                  # XML prompt templates (supervisor, academic, planner, emotional)
├── src/
│   ├── graph/                    # LangGraph orchestration
│   │   ├── supervisor.py         # Intent routing + keypoint extraction
│   │   ├── academic.py           # Parallel RAG + web search, answer generation, hallucination eval
│   │   ├── planner.py            # Policy search + study plan generation
│   │   ├── emotional.py          # Emotional support
│   │   ├── builder.py            # Graph construction + compilation
│   │   └── state.py              # TutorState definition
│   ├── rag/                      # RAG engine (loader, indexer, retriever)
│   ├── tools/                    # Search + RAG tool wrappers
│   ├── config/                   # YAML settings + XML prompt loader
│   ├── database/                 # PostgreSQL checkpointer lifecycle
│   ├── tracing/                  # OpenTelemetry setup, decorators, SQLite exporter
│   └── schemas.py                # Pydantic request models
├── frontend/                     # Next.js 16 + Tailwind CSS 4
│   ├── app/page.tsx              # Main page with SSE consumer
│   └── components/               # Chat area, reasoning path panel, sidebar
├── data/                         # Knowledge base source documents
├── tests/                        # 17 test modules (~250 test cases)
├── scripts/                      # Index builder, CLI test client
└── docs/                         # Architecture and requirements docs
```

## Testing

```bash
# Run all unit tests (excludes integration tests that require live APIs)
python -m pytest tests/ --ignore=tests/test_integration.py -v

# Run integration tests (requires .env with valid API keys + chroma_store/)
python -m pytest tests/test_integration.py -v
```

## License

MIT
