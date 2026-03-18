# Gaokao Tutor

<p align="center"><a href="README_zh.md">中文文档</a></p>

A multi-agent conversational AI that helps high school students prepare for the Chinese College Entrance Examination (Gaokao). Built with **LangGraph** and **Streamlit**, it routes user queries to provide subject tutoring, personalized study plans, and emotional support.

## Core Features

- **Academic Q&A** — RAG-powered subject knowledge retrieval with DuckDuckGo web search fallback, step-by-step solutions, and source citations
- **Study Planner** — Generates personalized revision schedules enriched with the latest Gaokao policy data
- **Emotional Support** — Empathetic responses in the persona of an experienced homeroom teacher
- **Intent Routing** — LLM-based supervisor classifies user intent and dispatches to the appropriate workflow

## Tech Stack

| Layer | Choice | Purpose |
| ----- | ------ | ------- |
| Frontend | Streamlit | Chat UI with streaming output |
| Orchestration | LangGraph | Multi-agent state graph |
| LLM | DeepSeek-V3 | Chat completion |
| Vector Store | ChromaDB | Local knowledge retrieval |
| Embedding | BAAI/bge-m3 (SiliconFlow) | Text vectorization |
| PDF Parser | PyMuPDF | Document ingestion |
| Web Search | DuckDuckGo | Online fallback retrieval |

## Quick Start

### Prerequisites

- Python 3.11+
- [Conda](https://docs.conda.io/) (recommended)

### Setup

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
```

### Build Knowledge Base

```bash
python scripts/build_index.py
```

### Run

```bash
streamlit run app.py
```

## Project Structure

```
gaokao_tutor/
├── app.py                    # Streamlit entry point
├── src/
│   ├── graph/                # LangGraph orchestration
│   │   ├── supervisor.py     # Intent routing + keypoint extraction
│   │   ├── academic.py       # RAG retrieval + answer generation
│   │   ├── planner.py        # Study plan generation
│   │   ├── emotional.py      # Emotional support
│   │   ├── builder.py        # Graph construction
│   │   ├── state.py          # TutorState definition
│   │   └── stream_adapter.py # LangGraph → Streamlit bridge
│   ├── rag/                  # RAG engine (loader, indexer, retriever)
│   ├── tools/                # Search + RAG tool wrappers
│   └── prompts/              # Prompt templates (decoupled)
├── data/                     # Knowledge base source documents
├── tests/                    # Unit and integration tests
└── docs/                     # Architecture and requirements docs
```

## License

MIT
