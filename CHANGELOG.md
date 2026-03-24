# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.2.0] — 2026-03-23

### Added

**Hybrid Retrieval RAG**
- Three-stage retrieval pipeline: ChromaDB vector search → BM25 keyword search → BGE Reranker (`BAAI/bge-reranker-v2-m3` via SiliconFlow API)
- Chinese word segmentation for BM25 via `jieba`; corpus built lazily from ChromaDB at first query
- New `src/rag/reranker.py`: SiliconFlow Reranking API wrapper with graceful degradation (returns original-order results on API failure)
- Added `rank-bm25` and `jieba` to `requirements.txt`
- New `rag:` section in `config/settings.yaml`: `vector_top_k`, `bm25_top_k`, `reranker_top_n`, `reranker_model`

**Centralized LLM Factory + Supervisor Model Switch**
- New `get_node_llm(node_name, **overrides)` in `src/graph/llm.py`: reads `model`, `base_url`, `api_key_env`, `temperature` per node from `settings.yaml`; falls back to `DEEPSEEK_*` env vars if not overridden
- Removed four duplicated `_get_llm()` factory functions from `supervisor.py`, `academic.py`, `planner.py`, `emotional.py`
- Supervisor now uses **Qwen2.5-7B-Instruct on SiliconFlow** (`temperature=0.0`) for low-latency intent routing; generation nodes retain DeepSeek-V3
- Added `model`, `base_url`, `api_key_env` fields to `supervisor:` section in `settings.yaml`

**Cross-Provider LLM Fallback**
- Fallback now points to **SiliconFlow + Qwen2.5-7B-Instruct** (true cross-infrastructure failover)
- Updated `.env.example`: uncommented and populated `FALLBACK_MODEL`, `FALLBACK_API_KEY`, `FALLBACK_BASE_URL`; added `RERANKER_MODEL` placeholder; removed stale `TAVILY_API_KEY`

**Enriched SSE Stream**
- `node_event` end payloads now include `duration_ms` (backend-computed, monotonic) and `error` (string or null)
- New SSE event type `usage`: `{"type": "usage", "node": "...", "input_tokens": N, "output_tokens": N, "total_tokens": N}` emitted after each LLM node call (when model returns `usage_metadata`)
- Node timing tracked in-memory within `generate_sse()` using `time.monotonic()`

**Frontend — Token Usage Display**
- `RightPanel` now shows a per-session token usage counter (input / output / total); resets on new chat
- System Logs: new `[PERF]` entries show node completion time (`duration_ms`); `[ERROR]` entries surface backend node errors; `[USAGE]` entries log per-node token consumption
- Extended `LogEntry.type` union: `"perf" | "usage"` in addition to existing `"info" | "error" | "warning"`

**Frontend — Graph DAG Visualization**
- Tab toggle in the Reasoning Path section: "Node Trail" (existing sequential view) vs "Graph View" (new)
- Graph View renders the full 9-node LangGraph topology as a static DAG (hand-laid SVG edges + CSS-positioned nodes)
- Node states: `idle` (gray/dashed) → `running` (orange + pulse animation) → `done` (green + duration badge)
- `NodeEvent` interface extended with `durationMs?: number` from the enriched SSE payload

### Changed

- `config/prompts/academic_answer.xml`: removed instruction to disclose internal retrieval sources; replaced with directive to answer naturally without mentioning reference materials or retrieval process
- `config/prompts/academic_system.xml`: replaced `引用来源` (cite sources) requirement with `自然作答` (answer naturally) guideline

### Fixed

- `src/rag/retriever.py`: increased default `vector_top_k` to 10 (was 5) to improve recall before reranking

### Known Issues

- Supervisor (Qwen2.5-7B) may misroute queries about specific past exams due to training data knowledge cutoff. Scheduled for v0.3.0.
- Exam paper RAG chunking is character-count-based; the composition section (作文) may be diluted in mixed chunks. Section-aware chunking planned for v0.3.0.

---

## [0.1.0] — 2026-03-18

### Added

- Multi-agent LangGraph `StateGraph` with three branches: Academic, Planner, Emotional
- LLM-based Supervisor node for intent classification and keypoint extraction (single API call)
- Academic branch: parallel fan-out to `rag_retrieve` + `web_search`, fan-in at `generate_answer`, hallucination evaluation with retry loop (up to `max_retries` from config)
- Planner branch: `search_policy` (DuckDuckGo) → `generate_plan`
- Emotional branch: single LLM call with homeroom teacher persona
- FastAPI `POST /stream` SSE endpoint using `graph.astream_events(version="v2")`
- Node lifecycle events (`node_event`) and token streaming (`token`) via SSE
- Next.js 16 + Tailwind CSS frontend: chat area, reasoning path panel (linear node trail), system logs, left sidebar
- ChromaDB vector store with BAAI/bge-m3 embedding (SiliconFlow API), L2→relevance normalization
- DuckDuckGo web search with configurable timeout and graceful fallback
- OpenTelemetry distributed tracing: `@traced_node` decorator on all nodes, `traced_llm_call` / `traced_retrieval` / `traced_search` context managers, OTLP→Jaeger + SQLite fallback exporter
- PostgreSQL-backed LangGraph checkpointer for multi-turn conversation memory; graceful stateless degradation when `DB_URI` unset
- Configuration system: `config/settings.yaml` (YAML, dot-notation access) + `config/prompts/*.xml` (8 XML templates), thread-safe cache with invalidation
- LLM fallback mechanism (`invoke_with_fallback`) catching 6 OpenAI error types
- FastAPI auto-instrumentation via `opentelemetry-instrumentation-fastapi`
- 17 test modules, ~250 test cases (fully mocked, no live API dependency)
- GitHub Actions CI: unit tests (Python 3.11/3.12/3.13 matrix) + security audit job
