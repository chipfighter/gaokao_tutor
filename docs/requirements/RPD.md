# RPD — Requirements Analysis

## Functional Requirements

### FR-1: Intent Recognition & Routing

The Supervisor Node uses an LLM with few-shot prompting and structured JSON output to classify user input into one of three intents and route accordingly.

| Intent | Route Target | Trigger Examples |
| ------ | ------------ | ---------------- |
| `academic` | SubGraph A (Subject Tutor) | "How do I solve quadratic functions?", "Analyze this reading comprehension" |
| `planning` | SubGraph B (Study Planner) | "Make me a review plan for next week", "When is this year's Gaokao?" |
| `emotional` | Direct emotional support reply | "I'm so anxious", "I feel like I can't learn anything" |

- **Prompt strategy**: Few-shot (3+ examples per intent) + structured output (JSON mode)
- **Target accuracy**: > 90%
- **Fallback**: Low confidence routes to `academic` by default

### FR-2: Academic Tutor Workflow (SubGraph A)

Handles subject knowledge questions and past exam problem analysis with RAG-first retrieval and web search fallback.

- **Keypoint extraction**: Extract subject, knowledge points, question type from user input (LLM structured output)
- **RAG retrieval**: top-k=5, metadata filtering (subject, year, type), source citations; relevance < 0.3 triggers web search
- **Web search fallback**: DuckDuckGo API returns latest relevant content
- **Answer generation**: Detailed solution with step-by-step reasoning, knowledge point connections, common mistakes; streaming output

### FR-3: Study Planner Workflow (SubGraph B)

Generates personalized study plans combining the student's goals with the latest Gaokao policy information retrieved via web search.

**Output format**: Structured Markdown task list with timeline (daily/weekly), priority labels, and subject distribution.

### FR-4: Emotional Support

Detects emotional distress signals (anxiety, frustration, burnout) and responds with warm, practical encouragement in the persona of an experienced homeroom teacher.

- Balances emotional support with practical study advice
- If severe distress is detected, gently suggests professional counseling

### FR-5: Streamlit Frontend

| Component | Specification |
| --------- | ------------- |
| Chat interface | `st.chat_message` + `st.chat_input`, renders message history |
| Streaming output | `st.empty()` + LangGraph `stream()`, progressive rendering |
| Source citations | `st.expander` showing RAG-retrieved document snippets + source filenames |
| Sidebar | Usage guide, clear chat button |
| Node status | `st.status()` showing current execution node |

## Non-Functional Requirements

| Dimension | Requirement | Implementation |
| --------- | ----------- | -------------- |
| Response speed | First token < 2s (perceived via streaming) | LLM streaming + LangGraph stream mode |
| Security | API keys never hardcoded or in Git history | `.env` (local) + platform secrets (deploy) |
| Extensibility | Add subject = add `data/` docs + adjust prompt | ChromaDB metadata design + `prompts/` separation |
| Maintainability | Prompts fully decoupled from business logic | `src/prompts/` independent directory |
| Cost control | Minimize API costs | DeepSeek-V3 + SiliconFlow embedding API |
| Observability | Key node execution status visible | `st.status()` + LangGraph node names |
