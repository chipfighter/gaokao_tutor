"""Gaokao Tutor — AI-powered tutoring assistant for Chinese Gaokao preparation."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.builder import get_compiled_graph
from src.graph.stream_adapter import stream_graph_to_streamlit

# ── Page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="高考辅导 AI 助手",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Noto Sans SC', sans-serif;
}

/* Hide default Streamlit header/footer */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Chat container spacing */
.stChatMessage {
    padding: 0.8rem 1rem;
    border-radius: 12px;
    margin-bottom: 0.5rem;
}

/* Sidebar styling — Gemini-inspired light palette */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f0f4f9 0%, #e8ecf1 100%);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label {
    font-weight: 500;
    color: #333 !important;
}

/* Expander (citations) styling */
.streamlit-expanderHeader {
    font-size: 0.9rem;
    font-weight: 500;
    border-radius: 8px;
}

/* Status component */
div[data-testid="stStatusWidget"] {
    border-radius: 8px;
}

/* Wider chat input */
.stChatInput {
    border-radius: 12px;
}

/* Hero section */
.hero-container {
    text-align: center;
    padding: 2rem 1rem 1rem 1rem;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #1E88E5 0%, #64B5F6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero-subtitle {
    color: #888;
    font-size: 0.95rem;
    letter-spacing: 0.5px;
}

/* Feature pills for empty state */
.feature-pills {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.pill {
    background: #f0f4ff;
    color: #1E88E5;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 500;
}

/* Quick action cards */
.quick-actions {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
    margin: 1.5rem auto;
    max-width: 500px;
}
.action-card {
    background: #fafbfd;
    border: 1px solid #e8ecf1;
    border-radius: 10px;
    padding: 0.8rem;
    text-align: center;
    transition: all 0.2s;
}
.action-card:hover {
    border-color: #1E88E5;
    box-shadow: 0 2px 8px rgba(30,136,229,0.1);
}
.action-icon {
    font-size: 1.4rem;
    margin-bottom: 0.2rem;
}
.action-text {
    font-size: 0.8rem;
    color: #555;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state init ───────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:
    with st.spinner("正在加载模型与知识库，首次启动约需 30 秒..."):
        st.session_state.graph = get_compiled_graph()

# ── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
<div style="text-align:center; padding: 1rem 0;">
    <span style="font-size: 2.5rem;">🎓</span>
    <h2 style="margin: 0.3rem 0 0 0; font-size: 1.3rem; color: #1a1a2e;">高考辅导 AI</h2>
    <p style="font-size: 0.75rem; color: #666; margin-top: 0.2rem;">Powered by DeepSeek + LangGraph</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()

    subject = st.selectbox(
        "📖 科目",
        options=["自动识别", "数学", "语文"],
        index=0,
    )

    mode = st.radio(
        "💡 模式",
        options=["智能问答", "学习规划"],
        index=0,
        horizontal=True,
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 清空", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 重载", use_container_width=True, help="重新加载 LangGraph"):
            del st.session_state["graph"]
            st.rerun()

    st.divider()

    with st.expander("📋 使用指南"):
        st.markdown(
            """
**三种能力：**

🔬 **学科答疑** — 知识点讲解、真题解析、解题思路

📅 **学习规划** — 个性化复习计划、时间管理建议

💪 **心理辅导** — 缓解焦虑、提振信心、调整心态
""",
        )

    st.caption("v0.1.0 · Made with LangGraph")

# ── Main area ────────────────────────────────────────────────────────

_SUBJECT_MAP = {"自动识别": None, "数学": "math", "语文": "chinese"}

if not st.session_state.messages:
    st.markdown(
        """
<div class="hero-container">
    <div class="hero-title">高考辅导 AI 助手</div>
    <div class="hero-subtitle">学科答疑 · 学习规划 · 情绪支持</div>
    <div class="feature-pills">
        <span class="pill">📚 RAG 知识检索</span>
        <span class="pill">🌐 实时网络搜索</span>
        <span class="pill">🧠 多智能体协作</span>
        <span class="pill">💬 流式输出</span>
    </div>
</div>
<div class="quick-actions">
    <div class="action-card">
        <div class="action-icon">📐</div>
        <div class="action-text">二次函数判别式怎么用？</div>
    </div>
    <div class="action-card">
        <div class="action-icon">📝</div>
        <div class="action-text">制定下周复习计划</div>
    </div>
    <div class="action-card">
        <div class="action-icon">📖</div>
        <div class="action-text">文言文阅读技巧</div>
    </div>
    <div class="action-card">
        <div class="action-icon">💪</div>
        <div class="action-text">感觉压力好大...</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

# ── Chat history display ─────────────────────────────────────────────

for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ── User input ───────────────────────────────────────────────────────

if prompt := st.chat_input("输入你的问题..."):

    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        graph_input = {"messages": st.session_state.messages}

        accumulated = stream_graph_to_streamlit(
            graph=st.session_state.graph,
            user_input=graph_input,
        )

        if accumulated and "messages" in accumulated:
            ai_msgs = accumulated["messages"]
            if ai_msgs:
                last_ai = ai_msgs[-1]
                if isinstance(last_ai, AIMessage):
                    st.session_state.messages.append(last_ai)

        docs = accumulated.get("retrieved_docs", []) if accumulated else []
        if docs:
            with st.expander("📚 参考来源", expanded=False):
                for i, doc in enumerate(docs, 1):
                    source = doc.get("source", "未知来源")
                    score = doc.get("score", "N/A")
                    content = doc.get("content", "")
                    st.markdown(f"**[{i}] {source}**（相关度：{score}）")
                    st.caption(
                        content[:300] + "..." if len(content) > 300 else content
                    )
                    if i < len(docs):
                        st.divider()
