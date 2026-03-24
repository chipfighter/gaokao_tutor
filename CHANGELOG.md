# 变更日志

本文档记录项目所有重要变更。
格式遵循 [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) 规范。

---

## [0.2.1] — 2026-03-24

### 文档

- 将 `README.md` 主体替换为中文版（原英文内容迁移至 `README_en.md` 存档）
- 删除冗余的 `README_zh.md`，导航链接更新为 `README.md` ↔ `README_en.md` 互指
- 翻译 `docs/architecture/v0.2.0/diagram_design.md` 散文部分为中文（Mermaid 节点标签保持英文）
- 翻译本 `CHANGELOG.md` 为中文
- 新增 `docs/requirements/backlog_drafts.md` 中文占位说明

---

## [0.2.0] — 2026-03-23

### 新增

**混合检索 RAG**
- 三阶段检索流水线：ChromaDB 向量检索 → BM25 关键词检索 → BGE Reranker（`BAAI/bge-reranker-v2-m3`，通过 SiliconFlow API）
- BM25 使用 `jieba` 中文分词；语料库在首次查询时从 ChromaDB 延迟构建
- 新增 `src/rag/reranker.py`：SiliconFlow Reranking API 封装，含降级策略（API 失败时按原始顺序返回结果）
- `requirements.txt` 新增 `rank-bm25` 和 `jieba` 依赖
- `config/settings.yaml` 新增 `rag:` 配置块：`vector_top_k`、`bm25_top_k`、`reranker_top_n`、`reranker_model`

**统一 LLM 工厂 + Supervisor 模型切换**
- `src/graph/llm.py` 新增 `get_node_llm(node_name, **overrides)`：从 `settings.yaml` 按节点读取 `model`、`base_url`、`api_key_env`、`temperature`；未覆盖时回退至 `DEEPSEEK_*` 环境变量
- 移除 `supervisor.py`、`academic.py`、`planner.py`、`emotional.py` 中四个重复的 `_get_llm()` 工厂函数
- Supervisor 改用 **SiliconFlow 上的 Qwen2.5-7B-Instruct**（`temperature=0.0`）实现低延迟意图路由；生成节点保留 DeepSeek-V3
- `settings.yaml` 的 `supervisor:` 节新增 `model`、`base_url`、`api_key_env` 字段

**跨厂商 LLM 容灾**
- Fallback 改为指向 **SiliconFlow + Qwen2.5-7B-Instruct**（真正的跨基础设施故障转移）
- 更新 `.env.example`：取消注释并填充 `FALLBACK_MODEL`、`FALLBACK_API_KEY`、`FALLBACK_BASE_URL`；新增 `RERANKER_MODEL` 占位；移除过时的 `TAVILY_API_KEY`

**增强 SSE 事件流**
- `node_event` 结束载荷新增 `duration_ms`（后端单调时钟计算）和 `error`（字符串或 null）
- 新增 SSE 事件类型 `usage`：`{"type": "usage", "node": "...", "input_tokens": N, "output_tokens": N, "total_tokens": N}`，在每个 LLM 节点调用后发出（当模型返回 `usage_metadata` 时）
- 在 `generate_sse()` 内使用 `time.monotonic()` 进行内存级节点计时

**前端 — Token 用量展示**
- `RightPanel` 新增会话级 Token 用量计数器（输入 / 输出 / 总计），新对话时重置
- 系统日志新增 `[PERF]` 条目（显示节点耗时 `duration_ms`）、`[ERROR]` 条目（展示后端节点错误）、`[USAGE]` 条目（记录每节点 Token 消耗）
- `LogEntry.type` 联合类型扩展：在原有 `"info" | "error" | "warning"` 基础上新增 `"perf" | "usage"`

**前端 — Graph DAG 可视化**
- 推理路径区域新增选项卡切换："节点轨迹"（原有顺序视图）与"图视图"（新增）
- 图视图将完整 9 节点 LangGraph 拓扑渲染为静态 DAG（手工布局的 SVG 边 + CSS 定位节点）
- 节点状态：`idle`（灰色/虚线）→ `running`（橙色 + 脉冲动画）→ `done`（绿色 + 耗时徽标）
- `NodeEvent` 接口扩展 `durationMs?: number`，来自增强后的 SSE 载荷

### 变更

- `config/prompts/academic_answer.xml`：移除披露内部检索来源的指令，改为引导模型自然作答，不提及参考资料或检索过程
- `config/prompts/academic_system.xml`：将"引用来源"要求替换为"自然作答"指引

### 修复

- `src/rag/retriever.py`：将默认 `vector_top_k` 从 5 提升至 10，改善 Reranker 前的召回率

### 已知问题

- Supervisor（Qwen2.5-7B）因训练数据截止日期，对特定历届真题查询可能产生意图误判，计划在 v0.3.0 修复。
- 试卷 RAG 分块基于字符数量，作文板块内容可能与其他题型混入同一 chunk，节标题感知分块计划在 v0.3.0 引入。

---

## [0.1.0] — 2026-03-18

### 新增

- 基于 LangGraph `StateGraph` 的多智能体系统，含三条分支：学科辅导（Academic）、学习规划（Planner）、情绪支持（Emotional）
- 基于 LLM 的 Supervisor 节点，单次 API 调用完成意图分类与知识点提取
- 学科辅导分支：`rag_retrieve` + `web_search` 并行 fan-out，`generate_answer` fan-in，含幻觉评估重试闭环（重试次数由 config 中 `max_retries` 控制）
- 学习规划分支：`search_policy`（DuckDuckGo）→ `generate_plan`
- 情绪支持分支：单次 LLM 调用，采用班主任人设
- FastAPI `POST /stream` SSE 端点，使用 `graph.astream_events(version="v2")`
- 通过 SSE 下发节点生命周期事件（`node_event`）和 Token 流式输出（`token`）
- Next.js 16 + Tailwind CSS 前端：对话区、推理路径面板（线性节点轨迹）、系统日志、左侧边栏
- ChromaDB 向量存储，使用 BAAI/bge-m3 嵌入（SiliconFlow API），L2 → 相关性分数归一化
- DuckDuckGo 网络搜索，含超时配置与优雅降级
- OpenTelemetry 分布式追踪：所有节点使用 `@traced_node` 装饰器，`traced_llm_call` / `traced_retrieval` / `traced_search` 上下文管理器，OTLP → Jaeger + SQLite 兜底导出
- PostgreSQL 支持的 LangGraph Checkpointer，实现多轮对话记忆；`DB_URI` 未设置时优雅降级为无状态模式
- 配置系统：`config/settings.yaml`（YAML，点号访问）+ `config/prompts/*.xml`（8 个 XML 提示词模板），线程安全缓存与失效机制
- LLM 容灾机制（`invoke_with_fallback`），捕获 6 种 OpenAI 错误类型
- 通过 `opentelemetry-instrumentation-fastapi` 对 FastAPI 自动埋点
- 17 个测试模块，约 250 个测试用例（全量 Mock，无实时 API 依赖）
- GitHub Actions CI：单元测试（Python 3.11/3.12/3.13 矩阵）+ 安全审计任务
