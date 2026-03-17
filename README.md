# Gaokao_Tutor

<p align="center"><a href="README_zh.md">中文文档</a></p>

## Introduction

Gaokao Tutor is a multi-agent conversational AI designed to assist high school students preparing for the Chinese College Entrance Examination (Gaokao). Built with LangGraph and Streamlit, it intelligently routes user queries to provide targeted subject knowledge, generate personalized study plans, and offer emotional support. 

The core architectural principle of this project is the ***separation of concerns***—— prompts, routing logic, Retrieval-Augmented Generation (RAG) engines, and tools are decoupled, ensuring any component can be securely swapped or upgraded without affecting the entire system.



---

## Core Features



---

## Tech Decision

| Layer         | Component        | Choice             | Alternatives Considered                                      |
| ------------- | ---------------- | ------------------ | ------------------------------------------------------------ |
| Frontend      | UI Framework     | **Streamlit**      | Gradio, FastAPI + Vue                                        |
| Orchestration | Agent Framework  | **LangGraph**      | CrewAI, AutoGen                                              |
| LLM           | Chat Model       | **DeepSeek-V3**    | GPT-4o-mini, Claude 3.5 Haiku                                |
| RAG           | Vector Store     | **ChromaDB**       | FAISS, Weaviate, Pinecone                                    |
| RAG           | Embedding Model  | **BAAI/bge-m3**    | bge-small-zh-v1.5                                            |
| RAG           | PDF Parser       | **PyMuPDF (fitz)** | pdfplumber, pypdf                                            |
| Search        | Web Search API   | **DuckDuckGo**     | Tavily, SerpAPI, Bing                                        |
| Deployment    | Primary Platform | **Aliyun**         | Tencent Cloud, Huggingface, Streamlit Cloud, Railway, Render |



---

## Demo (Video & Screenshot)





---

## How to start

