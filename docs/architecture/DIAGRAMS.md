# DIAGRAMS — Architecture & Flow Diagrams



## 1. System Architecture Overview

```mermaid
block-beta
    columns 1

    block:FRONTEND["Frontend Layer"]
        ChatUI["Chat UI"]
        Sidebar["Sidebar Config"]
        SourceDisplay["Source Display"]
    end

    space

    block:ORCHESTRATION["LangGraph Orchestration Layer"]
        Supervisor["Supervisor Node"]
        SubA["SubGraph A: Academic Tutor"]
        SubB["SubGraph B: Study Planner"]
        Direct["Direct Response: Emotional Support"]
    end

    space

    block:TOOLS["Tool Layer"]
        RAG["RAG Engine (ChromaDB)"]
        Search["Tavily Web Search"]
        DocStore["Document Store (gaokao_docs)"]
    end

    space

    block:MODELS["Model Layer"]
        LLM["DeepSeek-V3"]
        Embed["bge-small-zh-v1.5 (Local Embedding)"]
    end

    FRONTEND --> ORCHESTRATION
    ORCHESTRATION --> TOOLS
    TOOLS --> MODELS
```



## 2. LangGraph Main Flow

```mermaid
graph TD
    START([User Input]) --> Supervisor[Supervisor Node]

    Supervisor -->|intent = academic| SubGraphA
    Supervisor -->|intent = planning| SubGraphB
    Supervisor -->|intent = emotional| Emotional[Emotional Response]

    subgraph SubGraphA [SubGraph A: Academic Tutor]
        direction TB
        Extract[Extract Keypoints] --> RAG[ChromaDB RAG Retrieval]
        RAG -->|hit| GenAnswer[Generate Answer]
        RAG -->|miss| WebSearch[Tavily Web Search]
        WebSearch --> GenAnswer
    end

    subgraph SubGraphB [SubGraph B: Study Planner]
        direction TB
        InitPlan[Generate Draft Plan] --> SearchPolicy[Search Gaokao Policies]
        SearchPolicy --> RefinePlan[Refine & Output Task List]
    end

    GenAnswer --> END([Streamlit Streaming Output])
    RefinePlan --> END
    Emotional --> END
```



## 3. SubGraph A — Academic Tutor Internal Flow

```mermaid
graph TD
    A_START([Start]) --> N1["Node 1: extract_keypoints<br/>Output: subject + keypoints"]

    N1 --> N2["Node 2: rag_retrieve<br/>Tool: ChromaDB top-k=5<br/>Output: retrieved_docs"]

    N2 --> COND{distance ≤ 0.7?}

    COND -->|Yes — Hit| N4["Node 4: generate_answer<br/>LLM: DeepSeek-V3 Streaming"]

    COND -->|No — Miss| N3["Node 3: web_search<br/>Tool: Tavily API"]

    N3 --> N4

    N4 --> A_END([End])
```



## 4. SubGraph B — Study Planner Internal Flow

```mermaid
graph TD
    B_START([Start]) --> P1["Node 1: init_plan<br/>Output: draft plan in Markdown"]

    P1 --> P2["Node 2: search_policy<br/>Tool: Tavily API"]

    P2 --> P3["Node 3: refine_plan<br/>Output: final Markdown task list"]

    P3 --> B_END([End])
```



## 5. State Flow Diagram

```mermaid
graph LR
    subgraph TutorState
        messages["messages (append-only)"]
        intent["intent"]
        subject["subject"]
        keypoints["keypoints"]
        retrieved_docs["retrieved_docs"]
        search_results["search_results"]
        plan["plan"]
    end

    Supervisor -->|writes| intent
    extract_keypoints -->|writes| subject
    extract_keypoints -->|writes| keypoints
    rag_retrieve -->|writes| retrieved_docs
    web_search -->|writes| search_results
    search_policy -->|writes| search_results
    init_plan -->|writes| plan
    ALL_NODES["All terminal nodes"] -->|append| messages
```



## 6. Deployment Pipeline

```mermaid
graph TD
    DEV[Developer Machine] -->|git push origin main| GH[GitHub Repository]

    GH -->|Auto-sync| HF[HuggingFace Spaces]
    GH -->|Auto-deploy| SC[Streamlit Cloud]

    subgraph HF_STARTUP [HF Spaces Startup Sequence]
        direction TB
        S1[pip install requirements]
        S1 --> S2{chroma_store/ exists?}
        S2 -->|No| S3[Run build_index.py]
        S2 -->|Yes| S4[Load ChromaDB]
        S3 --> S4
        S4 --> S5[Load bge-small-zh]
        S5 --> S6[Compile LangGraph]
        S6 --> S7[App Ready]
    end

    HF --> HF_STARTUP

    subgraph SECRETS [Required Secrets]
        K1[DEEPSEEK_API_KEY]
        K2[TAVILY_API_KEY]
    end

    SECRETS -.-> HF
    SECRETS -.-> SC
```
