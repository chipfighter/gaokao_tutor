"""Thin streaming adapter: LangGraph stream() → Streamlit UI components."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

NODE_STATUS_LABELS = {
    "supervisor": "🔍 正在分析你的问题...",
    "rag_retrieve": "📚 在知识库中检索...",
    "web_search": "🌐 搜索网络获取最新信息...",
    "generate_answer": "✍️ 正在生成解答...",
    "search_policy": "🔎 查询最新高考政策...",
    "generate_plan": "📋 正在制定学习计划...",
    "emotional_response": "💬 正在回复...",
}


def stream_graph_to_streamlit(
    graph: CompiledStateGraph,
    user_input: dict[str, list],
    config: dict | None = None,
) -> dict:
    """Run the graph in streaming mode and render updates to Streamlit.

    ``stream_mode="updates"`` yields ``{node_name: node_output_dict}`` per step.
    We accumulate all outputs so the caller can extract retrieved_docs for citations.
    """
    accumulated: dict = {}

    status_placeholder = st.status("🚀 处理中...", expanded=False)
    message_placeholder = st.empty()

    for chunk in graph.stream(
        user_input,
        config=config,
        stream_mode="updates",
    ):
        if not isinstance(chunk, dict):
            continue

        for node_name, update in chunk.items():
            if not isinstance(update, dict):
                continue

            label = NODE_STATUS_LABELS.get(node_name, f"⚙️ {node_name}...")
            status_placeholder.update(label=label)

            accumulated.update(update)

            if "messages" in update and update["messages"]:
                last_ai = update["messages"][-1]
                content = (
                    last_ai.content if hasattr(last_ai, "content") else str(last_ai)
                )
                if content:
                    message_placeholder.markdown(content)

    status_placeholder.update(label="✅ 完成", state="complete")

    return accumulated
