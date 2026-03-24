# 高考辅导 AI

<p align="center">
  <a href="README_en.md">English README</a> ·
  <a href="docs/architecture/v0.2.0/diagram_design.md">架构图</a> ·
  <a href="CHANGELOG.md">更新日志</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-v0.2.0-orange" alt="version" />
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="license" />
  <img src="https://github.com/<your-username>/gaokao_tutor/actions/workflows/ci.yml/badge.svg" alt="CI" />
</p>

一个面向生产场景的高考备考多智能体对话 AI 系统，基于 **LangGraph**（有状态编排）、**FastAPI**（SSE 流式传输）和 **Next.js**（响应式前端）构建。一个轻量级的 Qwen2.5-7B 路由 Agent 将用户的问题分发给三个专项 Agent：学科辅导、学习规划和情绪疏导，每个分支都具备完整的可观测性和容错机制。

---

## 效果演示

<img src="./assets/v0.2.0/3a9a28cc-86fd-49ad-bf46-29c3b38f8b38.png" alt="demo-screenshot-1" style="zoom:40%;" />

<img src="./assets/v0.2.0/793acbbf-c209-4572-a108-98e6a80527cf.png" alt="demo-screenshot-2" style="zoom:40%;" />

---

## v0.2.0 新特性

- **混合检索 RAG**：三阶段检索流水线——ChromaDB 向量检索 + BM25 关键词检索（jieba 分词）+ BGE Reranker（SiliconFlow API），召回率显著优于纯向量检索
- **增强 SSE 事件流**：`node_event` 现在携带 `duration_ms` 和 `error` 字段，新增 `usage` 事件类型，实时上报各节点的 Token 消耗
- **图拓扑可视化**：右侧面板支持选项卡切换——"节点列表"（顺序展示）与"图视图"（完整 DAG 拓扑 + 实时节点状态）
- **Token 用量展示**：每次会话的 Token 累计计数（输入 / 输出 / 总计）实时显示在右侧面板
- **统一 LLM 工厂**：`get_node_llm()` 将四个图模块的 LLM 构建逻辑收敛为一处。Supervisor 改用 **SiliconFlow 上的 Qwen2.5-7B**（低延迟路由），生成节点保留 DeepSeek-V3
- **跨厂商容灾降级**：Fallback 指向 SiliconFlow + Qwen2.5-7B，实现真正的跨基础设施故障转移

---

## 核心功能

- **学科问答** — 混合 RAG（向量 + BM25 + Reranker）并行 Fan-out/Fan-in 检索，幻觉评估 + 自动重试闭环
- **学习规划** — 结合实时高考政策搜索数据，生成个性化复习计划
- **情绪支持** — 以经验丰富的班主任身份，提供温暖而实用的回应
- **意图路由** — Qwen2.5-7B Supervisor 低延迟分类用户意图，精准分发到对应工作流
- **LLM 容灾** — DeepSeek 主 API 超时或返回 5xx 时，自动切换到 SiliconFlow（Qwen2.5-7B）
- **分布式追踪** — OpenTelemetry 全链路埋点，导出到 Jaeger（OTLP）并有 SQLite 兜底
- **状态持久化** — PostgreSQL 驱动的 LangGraph Checkpointer 实现多轮对话记忆；无数据库时自动降级为无状态运行
- **配置驱动** — YAML 运行参数 + XML 提示词注册表，修改行为无需动代码
- **实时可观测** — SSE 驱动的推理路径（节点列表或 DAG 视图）、节点耗时、错误流和 Token 用量展示
- **Markdown 渲染** — 完整 GFM 支持：表格、代码块、LaTeX 公式、列表

---

## 系统架构

```text
用户 ──► Next.js (SSE 消费端) ──► FastAPI /stream ──► LangGraph StateGraph
                                                            │
                         ┌──────────────────────────────────┼───────────────────────┐
                         ▼                                  ▼                       ▼
                    [学科辅导]                           [学习规划]             [情绪支持]
                  ┌──────┴──────┐                           │                       │
                  ▼             ▼                            ▼                       ▼
           rag_retrieve    web_search               search_policy        emotional_response
           (向量+BM25        (DuckDuckGo)                  │
            +Reranker)          │                           ▼
                  └──────┬──────┘                    generate_plan ──► END
                         ▼
                   generate_answer
                         │
                         ▼
              evaluate_hallucination
                    ┌────┴────┐
                 [重试]      [结束]
                    │
               academic_router（循环）
```

横切关注点：所有节点上的 `@traced_node` → OpenTelemetry → Jaeger UI / SQLite

详细 Mermaid 架构图见 [`docs/architecture/v0.2.0/diagram_design.md`](docs/architecture/v0.2.0/diagram_design.md)

---

## 技术选型

| 层级 | 组件 | 说明 |
| ---- | ---- | ---- |
| 前端 | Next.js 16 + Tailwind CSS 4 | 响应式聊天 UI、SSE 消费端、Markdown 渲染 |
| 后端 API | FastAPI + Uvicorn | SSE 端点（`POST /stream`）、CORS、OTel 自动埋点 |
| 编排 | LangGraph | 含 Fan-out/Fan-in、条件边和重试循环的 StateGraph |
| 路由 LLM | Qwen2.5-7B（SiliconFlow） | 轻量意图分类器（temperature=0.0） |
| 生成 LLM | DeepSeek-V3 | 学科解答、学习计划、情绪支持 |
| LLM 容灾 | Qwen2.5-7B（SiliconFlow） | 跨厂商故障转移 |
| 向量数据库 | ChromaDB | 本地知识库检索（L2→相关度归一化） |
| 文本嵌入 | BAAI/bge-m3（SiliconFlow） | RAG 向量化 |
| 关键词检索 | rank-bm25 + jieba | 中文感知 BM25 检索 |
| 重排序 | BAAI/bge-reranker-v2-m3（SiliconFlow） | 合并候选集的精排 |
| 网络搜索 | DuckDuckGo | 学习规划及学科问答的在线补充 |
| 状态持久化 | PostgreSQL（psycopg） | LangGraph Checkpointer 多轮对话记忆 |
| 可观测性 | OpenTelemetry + Jaeger + SQLite | 所有图节点的分布式链路追踪 |
| 配置管理 | YAML + XML | 运行参数与提示词模板 |

---

## 快速启动

### 环境要求

- Python 3.11+
- Node.js 18+ 和 npm
- [Conda](https://docs.conda.io/)（推荐）
- PostgreSQL（可选，用于状态持久化；不配置时自动降级为无状态）
- Jaeger 2.x（可选，用于追踪可视化）

### 后端安装

```bash
git clone https://github.com/<your-username>/gaokao_tutor.git
cd gaokao_tutor

conda create -n gaokao_tutor python=3.11 -y
conda activate gaokao_tutor

pip install -r requirements.txt

cp .env.example .env
# 编辑 .env 填入以下配置：
#   DEEPSEEK_API_KEY        — DeepSeek API 密钥（主力 LLM）
#   SILICONFLOW_API_KEY     — SiliconFlow 密钥（嵌入、重排、路由 LLM、容灾）
#   DB_URI                  — PostgreSQL URI（可选）
```

### 构建知识库

将高考试卷的 `.txt` / `.pdf` 文件放入 `data/chinese/` 或 `data/math/` 目录，然后：

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
# 终端 1 — 后端
uvicorn app:app --reload --port 8000

# 终端 2 — 前端
cd frontend
npm run dev
```

前端地址：`http://localhost:3000` · 后端 API：`http://localhost:8000`

### 可选：Jaeger 追踪

```bash
# 追踪界面：http://localhost:16686
./local_tools/jaeger-2.15.0-windows-amd64/jaeger.exe
```

---

## 项目结构

```text
gaokao_tutor/
├── app.py                        # FastAPI SSE 端点 + lifespan（追踪、Checkpointer、图初始化）
├── config/
│   ├── settings.yaml             # 运行参数（温度、超时、重试上限）
│   └── prompts/                  # XML 提示词模板（8 个文件）
├── src/
│   ├── graph/
│   │   ├── builder.py            # 图构建与编译
│   │   ├── state.py              # TutorState TypedDict
│   │   ├── supervisor.py         # 意图路由 + 关键词提取（Qwen2.5-7B）
│   │   ├── academic.py           # 并行检索、答案生成、幻觉评估
│   │   ├── planner.py            # 政策搜索 + 学习计划生成
│   │   ├── emotional.py          # 情绪支持
│   │   └── llm.py                # 统一 LLM 工厂：get_node_llm()、invoke_with_fallback()
│   ├── rag/
│   │   ├── loader.py             # PDF/TXT → 分块 Document
│   │   ├── indexer.py            # ChromaDB 索引构建（去重）
│   │   ├── retriever.py          # 混合检索：向量 + BM25 + Reranker
│   │   └── reranker.py           # SiliconFlow BGE Reranker API 封装
│   ├── tools/                    # 图节点的搜索和 RAG 工具封装
│   ├── config/                   # YAML 配置加载 + XML 提示词缓存（线程安全）
│   ├── database/                 # PostgreSQL Checkpointer 生命周期管理
│   ├── tracing/                  # OTel 初始化、@traced_node 装饰器、SQLite 导出器
│   └── schemas.py                # Pydantic 请求模型
├── frontend/
│   ├── app/page.tsx              # 主页面：SSE 消费、状态管理
│   └── components/
│       ├── chat-area.tsx         # 消息气泡 + Markdown 渲染
│       ├── right-panel.tsx       # 推理路径（节点列表 + DAG）、系统日志、Token 用量
│       └── left-sidebar.tsx      # 对话历史
├── data/
│   ├── chinese/                  # 语文试卷：2024 新课标 I/II、2025 全国 I/II
│   └── math/                     # 数学试卷（按需扩充）
├── scripts/
│   └── build_index.py            # 离线索引构建脚本
├── tests/                        # 18 个测试模块，250+ 测试用例，全部 Mock
└── docs/
    └── architecture/
        └── v0.2.0/diagram_design.md  # v0.2.0 Mermaid 架构图
```

---

## 测试

```bash
# 单元测试 — 无需在线 API（全部 Mock）
$env:OTEL_TRACING_ENABLED="false"
python -m pytest tests/ --ignore=tests/test_integration.py -v --tb=short

# 集成测试 — 需要有效的 .env + 已构建的 chroma_store/
python -m pytest tests/test_integration.py -v

# 前端构建检查
cd frontend && npm run build
```

---

## 已知限制

- **Supervisor 知识截止日期**：Qwen2.5-7B 的训练数据截止于 2024 年 6 月之前，查询已发生考试题目时（如"2024 年高考作文标题"）可能被误判为备考规划意图。v0.3.0 将通过改进 few-shot 示例修复。
- **RAG 分块策略**：试卷按字符数分块，作文（写作）题目可能与前序题目混入同一 chunk，导致检索精度下降。v0.3.0 计划引入感知节标题的分块策略。

---

## 版本路线图

### v0.3.0 — 多智能体规划 & RAG 成熟度

| 特性 | 优先级 | 描述 |
| ---- | ------ | ---- |
| **对抗式计划生成** | 高 | 学习计划子图引入多智能体博弈：一个"起草者"生成高强度初版计划，"学情审查员"和"情绪审查员"并行审阅，全票通过才放行，否则打回重写——通过对抗循环产出真正平衡、体贴学生的计划 |
| **人工介入计划审批** | 高 | 对抗循环收敛后，图执行挂起（LangGraph `interrupt`），将草稿推送到前端供用户手动编辑，确认后再恢复执行 |
| **感知节标题的 RAG 分块** | 高 | 用节标题感知策略（按"一、现代文阅读"/"四、写作"等切割）替代 `RecursiveCharacterTextSplitter`，避免作文题被稀释在混合 chunk 中 |
| **Supervisor Few-Shot 修复** | 中 | 为"查询历届真题"类查询补充 few-shot 示例，避免 Qwen2.5-7B 因知识截止日期导致误判 |
| **对话级回滚** | 低 | 利用 LangGraph Checkpointer 列出各轮对话快照，允许用户回退到之前的对话状态 |

---

## 许可证

MIT
