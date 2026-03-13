# TODO — 24h Sprint Development Plan


**Status Legend**: `[ ]` pending | `[x]` done | `[-]` in progress | `[~]` deferred



## Critical Path

```
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
Scaffold → RAG      → Tools   → Prompts → Graph   → Frontend
  1h        3h        1.5h      2h        5h        3h     = 15.5h
```

Phase 6 (Deploy) and Phase 7 (Wrap-up) can run in parallel or be deferred.



## Phase 0 — Project Scaffold (est. 1h)

| # | Task | Status | Notes |
|---|------|--------|-------|
| T0.1 | Create `requirements.txt` with pinned core dependencies | `[ ]` | See `TECH_STACK.md` dependency list |
| T0.2 | Create source directory structure (`src/graph/`, `src/rag/`, `src/tools/`, `src/prompts/`) | `[ ]` | Including all `__init__.py` files |
| T0.3 | Create `.env.example` and `.streamlit/config.toml` (theme config) | `[ ]` | All required env vars documented |
| T0.4 | Validate environment: `pip install -r requirements.txt` + Streamlit hello world | `[ ]` | Ensure no dependency conflicts |



## Phase 1 — RAG Engine (est. 3h) ⭐

| # | Task | Status | Notes |
|---|------|--------|-------|
| T1.1 | Implement `src/rag/loader.py`: PDF/Markdown loading + text chunking | `[ ]` | `RecursiveCharacterTextSplitter`, chunk_size=1000, overlap=200; inject metadata (subject, year, doc_type) |
| T1.2 | Implement `src/rag/indexer.py`: ChromaDB index building + persistence | `[ ]` | `persist_directory="chroma_store/"`; incremental upsert; collection = "gaokao_docs" |
| T1.3 | Implement `src/rag/retriever.py`: retrieval interface | `[ ]` | top-k=5; metadata where filter (subject/year); distance > 0.7 = miss |
| T1.4 | Prepare MVP test docs: 2-3 past exam PDFs in `data/math/` | `[ ]` | Use recent national exam papers |
| T1.5 | Run `scripts/build_index.py` to verify index building | `[ ]` | Verify: ChromaDB queryable, correct metadata returned |



## Phase 2 — Tools Layer (est. 1.5h)

| # | Task | Status | Notes |
|---|------|--------|-------|
| T2.1 | Implement `src/tools/rag_tool.py`: RAG retrieval wrapped as LangChain `@tool` | `[ ]` | Input: query + subject; Output: doc list + miss flag |
| T2.2 | Implement `src/tools/search_tool.py`: Tavily Web Search tool | `[ ]` | Use `TavilySearchResults`, max_results=3 |
| T2.3 | Unit tests: verify each tool works independently | `[ ]` | |



## Phase 3 — Prompt Engineering (est. 2h)

| # | Task | Status | Notes |
|---|------|--------|-------|
| T3.1 | Write `src/prompts/supervisor.py`: intent routing prompt | `[ ]` | Few-shot (3+ examples per intent); output `{"intent": "academic|planning|emotional"}`; JSON mode |
| T3.2 | Write `src/prompts/academic.py`: subject tutor system prompt | `[ ]` | Persona: senior Gaokao educator; require step-by-step reasoning, knowledge points, common mistakes; cite RAG sources |
| T3.3 | Write `src/prompts/planner.py`: study planner system prompt | `[ ]` | Persona: study planner; output structured Markdown plan (timeline, priorities) |
| T3.4 | Write `src/prompts/emotional.py`: emotional support system prompt | `[ ]` | Persona: experienced homeroom teacher, warm but practical |



## Phase 4 — LangGraph Core (est. 5h) ⭐ Critical Path

| # | Task | Status | Notes |
|---|------|--------|-------|
| T4.1 | Define `src/graph/state.py`: `TutorState` TypedDict | `[ ]` | Fields: messages, intent, subject, keypoints, retrieved_docs, search_results, plan |
| T4.2 | Implement `src/graph/supervisor.py`: Supervisor node | `[ ]` | LLM + structured output; write state.intent; low-confidence fallback to academic |
| T4.3 | Implement `src/graph/academic.py`: SubGraph A | `[ ]` | 4 nodes: extract_keypoints → rag_retrieve → (conditional) web_search → generate_answer |
| T4.4 | Implement `src/graph/planner.py`: SubGraph B | `[ ]` | 3 nodes: init_plan → search_policy → refine_plan |
| T4.5 | Implement `src/graph/emotional.py`: emotional response node | `[ ]` | Single-node direct LLM call |
| T4.6 | Implement `src/graph/stream_adapter.py`: thin streaming adapter | `[ ]` | Bridge LangGraph `stream()` → Streamlit UI components |
| T4.7 | Implement `src/graph/builder.py`: full graph construction & compilation | `[ ]` | Assemble Supervisor + 3 branches; `graph.compile()` |
| T4.8 | Local end-to-end test: run each intent once, verify state flow | `[ ]` | `python -c "from src.graph.builder import graph; ..."` |



## Phase 5 — Streamlit Frontend (est. 3h)

| # | Task | Status | Notes |
|---|------|--------|-------|
| T5.1 | Implement `app.py` base framework | `[ ]` | `st.set_page_config`; session_state init (messages, subject, mode) |
| T5.2 | Implement chat interface: message history rendering + user input | `[ ]` | `st.chat_message` + `st.chat_input`; call `stream_graph_to_streamlit()` |
| T5.3 | Implement sidebar | `[ ]` | Subject selector, mode toggle (Q&A / Planning), clear chat button |
| T5.4 | Implement RAG source citation display | `[ ]` | `st.expander("References")`; show source filename + content excerpt |
| T5.5 | Implement node execution status display | `[ ]` | `st.status()` showing "Searching...", "Generating...", etc. |
| T5.6 | Style: `config.toml` theme colors | `[ ]` | Suggested: primary `#1E88E5` (study blue); dark mode friendly |



## Phase 6 — Integration Test & Deploy (est. 2h)

| # | Task | Status | Notes |
|---|------|--------|-------|
| T6.1 | End-to-end test: 3+ full conversation rounds per intent | `[ ]` | Verify routing accuracy, RAG hits, web search fallback |
| T6.2 | Edge case test: empty input, very long input, irrelevant questions, API timeout | `[ ]` | |
| T6.3 | Push to GitHub (ensure `chroma_store/`, `.env`, `secrets.toml` in .gitignore) | `[ ]` | |
| T6.4 | Deploy to HuggingFace Spaces (primary): create Space (Streamlit SDK), configure Secrets | `[ ]` | Secrets: `DEEPSEEK_API_KEY`, `TAVILY_API_KEY` |
| T6.5 | Verify HF Spaces online: index building, model loading, full conversation flow | `[ ]` | |
| T6.6 | (Optional) Deploy to Streamlit Cloud: bind GitHub, configure Secrets | `[ ]` | Fallback deploy, monitor RAM ≤ 1GB |



## Phase 7 — Wrap Up (est. 1.5h)

| # | Task | Status | Notes |
|---|------|--------|-------|
| T7.1 | Write `README.md` (English) + `README_zh.md` (Chinese) | `[ ]` | Include project intro, tech stack, local setup, deploy guide, screenshots |
| T7.2 | Update `CHANGELOG.md` | `[ ]` | v0.1.0 initial release record |
| T7.3 | Code review: remove debug output, ensure no hardcoded secrets | `[ ]` | |



## Time Budget Summary

| Phase | Description | Est. Hours | Priority |
|-------|------------|-----------|----------|
| 0 | Project Scaffold | 1h | P0 Must |
| 1 | RAG Engine | 3h | P0 Must |
| 2 | Tools Layer | 1.5h | P0 Must |
| 3 | Prompt Engineering | 2h | P0 Must |
| 4 | LangGraph Core | 5h | P0 Critical Path |
| 5 | Streamlit Frontend | 3h | P0 Must |
| 6 | Integration & Deploy | 2h | P1 Deferrable |
| 7 | Wrap Up | 1.5h | P2 Deferrable |
| **Total** | | **~19h** | |



## MVP Cutoff

> If the 24h deadline is tight, minimum deliverable:

- [x] Phase 0 complete
- [x] Phase 1 (T1.1 ~ T1.4) complete
- [x] Phase 2 (T2.1 ~ T2.2) complete
- [x] Phase 3 complete
- [x] Phase 4 (T4.1 ~ T4.7) complete (SubGraph B can simplify to single LLM call)
- [x] Phase 5 (T5.1 ~ T5.2) complete (minimal chat interface)
- Above = demoable MVP 
