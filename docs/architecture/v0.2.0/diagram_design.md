# v0.2.0 Architecture Diagrams

This document contains Mermaid diagrams for the v0.2.0 system architecture.

---

## 1. Full System Architecture

```mermaid
flowchart TD
    User(["👤 User"])

    subgraph Frontend["Frontend — Next.js 16"]
        Chat["Chat Area\n(SSE consumer\nMarkdown renderer)"]
        RightPanel["Right Panel\n(DAG viz / Node Trail\nSystem Logs\nToken Usage)"]
        Sidebar["Left Sidebar\n(Chat History)"]
    end

    subgraph Backend["Backend — FastAPI"]
        API["POST /stream\nStreamingResponse\n(text/event-stream)"]
        SSE["generate_sse()\nastream_events v2"]
    end

    subgraph Graph["LangGraph StateGraph — TutorState"]
        Supervisor["supervisor\nQwen2.5-7B\nSiliconFlow\ntemperature=0.0"]
        subgraph Academic["Academic Branch"]
            AR["academic_router"]
            RAG["rag_retrieve\nHybrid RAG"]
            WS["web_search\nDuckDuckGo"]
            GA["generate_answer\nDeepSeek-V3"]
            EH["evaluate_hallucination\nDeepSeek-V3 structured"]
        end
        subgraph Planner["Planner Branch"]
            SP["search_policy\nDuckDuckGo"]
            GP["generate_plan\nDeepSeek-V3"]
        end
        ER["emotional_response\nDeepSeek-V3"]
    end

    subgraph RAGStack["RAG Stack"]
        ChromaDB[("ChromaDB\nvector store")]
        BM25["BM25 Index\njieba tokenizer"]
        Reranker["BGE Reranker\nSiliconFlow API\nbge-reranker-v2-m3"]
    end

    subgraph Infra["Infrastructure"]
        PG[("PostgreSQL\nLangGraph Checkpointer")]
        Jaeger["Jaeger UI\nlocalhost:16686"]
        SQLite[("SQLite\ntraces.db")]
        OTel["OpenTelemetry\nTracerProvider"]
    end

    User -- "HTTP POST /stream" --> API
    API --> SSE
    SSE -- "astream_events" --> Graph
    SSE -- "SSE: token / node_event / usage" --> Chat
    Chat --> RightPanel

    Supervisor -->|academic| AR
    Supervisor -->|planning| SP
    Supervisor -->|emotional| ER

    AR --> RAG
    AR --> WS
    RAG --> GA
    WS --> GA
    GA --> EH
    EH -->|"retry (count ≤ max_retries)"| AR
    EH -->|end| DONE1(["END"])

    SP --> GP
    GP --> DONE2(["END"])
    ER --> DONE3(["END"])

    RAG --> ChromaDB
    RAG --> BM25
    RAG --> Reranker

    Graph -. "checkpoint" .-> PG
    Graph -- "@traced_node" --> OTel
    OTel --> Jaeger
    OTel --> SQLite
```

---

## 2. LangGraph Node Topology (State Flow)

```mermaid
flowchart TD
    START(["START"])
    END1(["END"])
    END2(["END"])
    END3(["END"])

    START --> supervisor

    supervisor -->|"intent = academic"| academic_router
    supervisor -->|"intent = planning"| search_policy
    supervisor -->|"intent = emotional"| emotional_response

    subgraph fan_out["Fan-out / Fan-in (parallel)"]
        academic_router --> rag_retrieve
        academic_router --> web_search
        rag_retrieve --> generate_answer
        web_search --> generate_answer
    end

    generate_answer --> evaluate_hallucination

    evaluate_hallucination -->|"hallucination_detected=True\nretry_count ≤ max_retries"| academic_router
    evaluate_hallucination -->|"faithful OR retries exhausted"| END1

    search_policy --> generate_plan
    generate_plan --> END2

    emotional_response --> END3

    style fan_out fill:#f0f4f0,stroke:#7a9e7e
```

**State (`TutorState`) key fields and write ownership:**

| Field | Written by | Consumed by |
|-------|-----------|-------------|
| `messages` | supervisor (initial), generate_answer, generate_plan, emotional_response | all nodes |
| `intent` | supervisor | builder (conditional edge) |
| `subject` | supervisor | rag_retrieve (metadata filter) |
| `keypoints` | supervisor | rag_retrieve (query construction) |
| `context` | rag_retrieve, web_search (merged via `operator.add`) | generate_answer |
| `search_results` | search_policy | generate_plan |
| `retry_count` | evaluate_hallucination | should_retry_or_end |
| `hallucination_detected` | evaluate_hallucination | should_retry_or_end |

---

## 3. Hybrid RAG Pipeline

```mermaid
flowchart LR
    Q["User Query\n(joined keypoints)"]

    subgraph Stage1["Stage 1 — Retrieval (parallel)"]
        V["Vector Search\nChromaDB + BGE-M3\ntop_k = 10\nsubject filter applied"]
        B["BM25 Search\njieba tokenize\ntop_k = 10\nno subject filter"]
    end

    subgraph Stage2["Stage 2 — Merge"]
        M["Merge + Dedup\n(MD5 content hash)\nvector results first"]
    end

    subgraph Stage3["Stage 3 — Rerank"]
        R["BGE Reranker\nSiliconFlow API\nbge-reranker-v2-m3\ntop_n = 5"]
    end

    OUT["Top-N docs\n{content, source, score,\nrerank_score, metadata}"]

    FALLBACK["Graceful Degradation:\nreranker API fails → sorted by original score\nBM25 empty → pure vector results\nChromaDB empty → empty result"]

    Q --> V
    Q --> B
    V --> M
    B --> M
    M --> R
    R --> OUT
    R -. "on failure" .-> FALLBACK
    FALLBACK --> OUT
```

**Configuration (`config/settings.yaml`):**

```yaml
rag:
  vector_top_k: 10
  bm25_top_k: 10
  reranker_top_n: 5
  relevance_threshold: 0.3
  reranker_model: "BAAI/bge-reranker-v2-m3"
```

---

## 4. SSE Event Stream Schema

```mermaid
sequenceDiagram
    participant FE as Frontend (Next.js)
    participant BE as Backend (FastAPI)
    participant LG as LangGraph

    FE->>BE: POST /stream {"query": "...", "thread_id": "..."}
    BE->>LG: graph.astream_events(state_input, config, version="v2")

    loop for each graph node
        LG-->>BE: on_chain_start {name, metadata.langgraph_node}
        BE-->>FE: data: {"type":"node_event","status":"start","node":"supervisor"}

        alt LLM node (generate_answer / generate_plan / emotional_response)
            loop token streaming
                LG-->>BE: on_chat_model_stream {chunk.content}
                BE-->>FE: data: {"type":"token","content":"..."}
            end
            LG-->>BE: on_chat_model_end {output.usage_metadata}
            BE-->>FE: data: {"type":"usage","node":"generate_answer","input_tokens":N,"output_tokens":N,"total_tokens":N}
        end

        LG-->>BE: on_chain_end {name, metadata.langgraph_node}
        BE-->>FE: data: {"type":"node_event","status":"end","node":"supervisor","duration_ms":234,"error":null}
    end
```

**Frontend consumption mapping:**

| SSE Event | Frontend Handler |
|-----------|-----------------|
| `node_event` start | `nodeEvents` state: add `{node, status: "running", ts}` |
| `node_event` end | `nodeEvents`: mark `status: "done"`, attach `durationMs`; append `[PERF]` to logs |
| `node_event` end with error | `nodeEvents`: mark done; append `[ERROR]` to logs |
| `token` | Append `content` to current assistant message (streaming typewriter) |
| `usage` | Accumulate into `tokenUsage` state; append `[USAGE]` to logs |

---

## 5. LLM Configuration Architecture

```mermaid
flowchart TD
    subgraph Settings["config/settings.yaml"]
        SUP_CFG["supervisor:\n  model: Qwen/Qwen2.5-7B-Instruct\n  base_url: siliconflow\n  api_key_env: SILICONFLOW_API_KEY\n  temperature: 0.0"]
        AC_CFG["academic:\n  temperature: 0.7\n  (no model override → DEEPSEEK_*)"]
        PL_CFG["planner:\n  temperature: 0.7\n  (no model override → DEEPSEEK_*)"]
        EM_CFG["emotional:\n  temperature: 0.8\n  (no model override → DEEPSEEK_*)"]
    end

    Factory["get_node_llm(node_name, **overrides)\nsrc/graph/llm.py"]

    SUP_CFG --> Factory
    AC_CFG --> Factory
    PL_CFG --> Factory
    EM_CFG --> Factory

    subgraph Env[".env"]
        DS["DEEPSEEK_API_KEY\nDEEPSEEK_BASE_URL\nDEEPSEEK_MODEL"]
        SF["SILICONFLOW_API_KEY\n(shared by: embedding,\nreranker, supervisor,\nfallback)"]
        FB["FALLBACK_MODEL\nFALLBACK_API_KEY\nFALLBACK_BASE_URL\n(→ SiliconFlow + Qwen2.5-7B)"]
    end

    DS --> Factory
    SF --> Factory

    Factory --> Supervisor["Supervisor ChatOpenAI\nQwen2.5-7B @ SiliconFlow"]
    Factory --> Academic["Academic ChatOpenAI\nDeepSeek-V3"]
    Factory --> Planner["Planner ChatOpenAI\nDeepSeek-V3"]
    Factory --> Emotional["Emotional ChatOpenAI\nDeepSeek-V3"]

    FB --> Fallback["Fallback ChatOpenAI\nQwen2.5-7B @ SiliconFlow\n(auto-triggered by invoke_with_fallback)"]
```
