# DESIGN — Architecture Overview

**See Also**: [`DIAGRAMS.md`](./DIAGRAMS.md) for visual diagrams



## 1. System Overview

The system is a **multi-agent conversational AI** built on a 4-layer architecture:

1. **Frontend Layer** — Streamlit chat UI (user interaction & streaming display)
2. **Orchestration Layer** — LangGraph StateGraph (intent routing + subgraph execution)
3. **Tool Layer** — ChromaDB RAG Engine + DuckDuckGo Web Search
4. **Model Layer** — DeepSeek-V3 (LLM) + BAAI/bge-m3 via SiliconFlow (Embedding)

The key design principle is **separation of concerns**: prompts, graph logic, RAG engine, and tools are independently modular so any component can be swapped without affecting others.



## 2. High-Level Architecture

| Layer | Component | Responsibility |
|-------|----------|---------------|
| Frontend | Streamlit | Chat UI, streaming rendering, sidebar config, source display |
| Orchestration | LangGraph StateGraph | Intent routing (Supervisor) + subgraph execution (Academic / Planner / Emotional) |
| Tool | ChromaDB RAG + DuckDuckGo | Local knowledge retrieval + online search fallback |
| Model | DeepSeek-V3 + bge-m3 | LLM inference + text vectorization |

Inter-layer communication: Streamlit calls LangGraph via `stream_graph_to_streamlit()` thin adapter. LangGraph nodes share data through `TutorState`. Tools are invoked by nodes on demand.



## 3. LangGraph State Design

The `TutorState` TypedDict is the single source of truth flowing through the entire graph.

```python
class TutorState(TypedDict):
    messages: Annotated[list, add_messages]       # Append-only via reducer
    intent: Literal["academic", "planning", "emotional"]
    subject: str                   # e.g. "math", "chinese"
    keypoints: list[str]           # e.g. ["quadratic functions", "discriminant"]
    retrieved_docs: list[dict]     # [{content, source, score}, ...]
    search_results: list[dict]     # [{title, url, content}, ...]
    plan: str                      # Draft plan in Markdown
```

**State flow**:
```
START
  └─ Supervisor: reads messages[-1], writes intent + subject + keypoints
       ├─ academic → rag_retrieve → [web_search] → generate_answer → END
       ├─ planning → search_policy → generate_plan → END
       └─ emotional → emotional_response → END
```
