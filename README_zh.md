# 高考辅导

<p align="center"><a href="README.md">English README</a></p>

## 项目简介

Gaokao Tutor 是一个多 Agent 对话式人工智能平台，专为备考高考的高中生设计。项目基于 **LangGraph**（编排）、**FastAPI**（SSE 流式后端）和 **Next.js**（响应式前端）构建，通过智能意图路由，为学生提供包含**学科知识问题解答**、**个性化复习计划制定**以及**考前情绪疏导**功能的支持。

本项目遵循的核心架构原则是**关注点分离** —— 提示词（XML 模板）、运行参数（YAML 配置）、路由逻辑、检索增强 (RAG) 引擎以及外部工具的调用都是完全解耦的，从而确保系统中的任意组件都能在不影响大局的情况下被安全地热插拔或升级。

---

## 核心功能

- **学科问答** — 并行 RAG + 网络搜索检索（Fan-out/Fan-in），逐步推理解答与来源引用，幻觉评估自动重试循环
- **学习规划** — 结合最新高考政策数据生成个性化复习计划
- **情绪支持** — 以经验丰富的班主任角色提供共情回应
- **意图路由** — 基于 LLM 的 Supervisor 分类用户意图并分发至对应工作流
- **LLM 容灾** — 主 API 超时或返回 5xx 时自动切换到备用 LLM 提供者
- **分布式追踪** — OpenTelemetry 全链路追踪，支持 Jaeger 导出 + SQLite 兜底
- **状态持久化** — PostgreSQL Checkpointer 实现多轮对话记忆
- **配置驱动** — YAML 运行参数 + XML 提示词注册表，无需改代码即可调整行为
- **实时可观测** — 前端推理路径可视化 + 系统日志（基于 SSE 节点生命周期事件）
- **Markdown 渲染** — 助手回复支持完整 GFM Markdown（表格、代码块、列表）

---

## 技术选型

| 层级 | 组件 | 项目选择 | 备选方案 |
| ---- | ---- | -------- | -------- |
| 前端 UI | UI 框架 | **Next.js 16 + Tailwind CSS 4** | Gradio, Streamlit |
| 后端 API | 流式接口 | **FastAPI + Uvicorn (SSE)** | WebSocket, gRPC |
| 核心编排 | Agent 框架 | **LangGraph** | CrewAI, AutoGen |
| 大语言模型 | 对话模型 | **DeepSeek-V3**（+ 可配置备用） | GPT-4o-mini, Claude 3.5 Haiku |
| 检索增强 | 向量数据库 | **ChromaDB** | FAISS, Weaviate, Pinecone |
| 检索增强 | 文本嵌入模型 | **BAAI/bge-m3**（SiliconFlow API） | bge-small-zh-v1.5 |
| 检索增强 | PDF 解析库 | **PyMuPDF (fitz)** | pdfplumber, pypdf |
| 搜索工具 | 搜索引擎 API | **DuckDuckGo** | Tavily, SerpAPI, Bing |
| 状态持久化 | 数据库 | **PostgreSQL (psycopg)** | SQLite, Redis |
| 可观测性 | 追踪 | **OpenTelemetry + Jaeger + SQLite** | Zipkin, Datadog |
| 配置管理 | 运行参数/提示词 | **YAML + XML** | TOML, JSON |

---

## 效果演示（视频 & 截图）

---

## 如何启动

### 环境要求

- Python 3.11+
- Node.js 18+ 和 npm
- [Conda](https://docs.conda.io/)（推荐）
- PostgreSQL（可选，用于状态持久化）
- Jaeger 2.x（可选，用于追踪可视化）

### 后端安装

```bash
# 克隆仓库
git clone https://github.com/<your-username>/gaokao_tutor.git
cd gaokao_tutor

# 创建并激活 Conda 环境
conda create -n gaokao_tutor python=3.11 -y
conda activate gaokao_tutor

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API 密钥:
#   DEEPSEEK_API_KEY=<your-key>
#   SILICONFLOW_API_KEY=<your-key>
#   DB_URI=postgresql://postgres:<password>@localhost:5432/gaokao_tutor  (可选)
```

### 构建知识库

```bash
python scripts/build_index.py
```

### 前端安装

```bash
cd frontend
npm install
```

### 启动

```bash
# 终端 1 — 后端（SSE 流式 API）
uvicorn app:app --reload --port 8000

# 终端 2 — 前端（Next.js 开发服务器）
cd frontend
npm run dev
```

前端运行在 `http://localhost:3000`，后端 API 运行在 `http://localhost:8000`。

### 可选：Jaeger 追踪

```bash
# 启动 Jaeger（追踪界面在 http://localhost:16686）
./local_tools/jaeger-2.15.0-windows-amd64/jaeger.exe
```

---

## 测试

```bash
# 运行所有单元测试（排除需要在线 API 的集成测试）
python -m pytest tests/ --ignore=tests/test_integration.py -v

# 运行集成测试（需要 .env 中配置有效的 API 密钥 + chroma_store/）
python -m pytest tests/test_integration.py -v
```

---

## 许可证

MIT
