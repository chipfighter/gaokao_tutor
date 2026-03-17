# 高考辅导
<p align="center"><a href="README.md"> English README</a></p>

## 项目简介
Gaokao Tutor 是一个多 Agent 对话式人工智能平台，专为备考高考的高中生设计。项目基于 LangGraph 和 Streamlit 构建，通过智能意图路由，为学生提供包含**学科知识目标解答**、**个性化复习计划制定**以及**考前情绪疏导**功能的支持。

本项目遵循的核心架构原则是**关注点分离** —— 提示词、路由逻辑、检索增强 (RAG) 引擎以及外部工具的调用都是完全解耦的，从而确保系统中的任意组件都能在不影响大局的情况下被安全地热插拔或升级。



---

## 核心功能



---

## 技术选型
| 层级       | 组件           | 项目选择            | 备选方案                                              |
| ---------- | -------------- | ------------------- | ----------------------------------------------------- |
| 前端 UI    | UI 框架        | **Streamlit**       | Gradio, FastAPI + Vue                                 |
| 核心编排   | Agent 框架     | **LangGraph**       | CrewAI, AutoGen                                       |
| 大语言模型 | 对话模型       | **DeepSeek-V3**     | GPT-4o-mini, Claude 3.5 Haiku                         |
| 检索增强   | 向量数据库     | **ChromaDB**        | FAISS, Weaviate, Pinecone                             |
| 检索增强   | 文本嵌入模型   | **BAAI/bge-m3**     | bge-small-zh-v1.5                                     |
| 检索增强   | PDF 解析库     | **PyMuPDF (fitz)**  | pdfplumber, pypdf                                     |
| 搜索工具   | 搜索引擎 API   | **DuckDuckGo**      | Tavily, SerpAPI, Bing                                 |
| 部署方式   | 首选云计算平台 | **阿里云 (Aliyun)** | 腾讯云, Huggingface, Streamlit Cloud, Railway, Render |


---
## 效果演示（视频 & 截图）



---

## 如何启动

### 