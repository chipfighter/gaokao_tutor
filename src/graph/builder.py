"""Graph construction — assemble Supervisor + 3 branches, compile."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.graph.academic import (
    extract_keypoints,
    generate_answer,
    rag_retrieve,
    should_web_search,
    web_search,
)
from src.graph.emotional import emotional_response
from src.graph.planner import init_plan, refine_plan, search_policy
from src.graph.state import TutorState
from src.graph.supervisor import route_by_intent, supervisor_node


def build_graph() -> StateGraph:
    """Construct the full LangGraph StateGraph (uncompiled)."""

    graph = StateGraph(TutorState)

    # ── Nodes ────────────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor_node)

    # SubGraph A — Academic
    graph.add_node("extract_keypoints", extract_keypoints)
    graph.add_node("rag_retrieve", rag_retrieve)
    graph.add_node("web_search", web_search)
    graph.add_node("generate_answer", generate_answer)

    # SubGraph B — Planner
    graph.add_node("init_plan", init_plan)
    graph.add_node("search_policy", search_policy)
    graph.add_node("refine_plan", refine_plan)

    # Emotional
    graph.add_node("emotional_response", emotional_response)

    # ── Edges ────────────────────────────────────────────────────────

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_by_intent,
        {
            "academic": "extract_keypoints",
            "planning": "init_plan",
            "emotional": "emotional_response",
        },
    )

    # Academic flow
    graph.add_edge("extract_keypoints", "rag_retrieve")
    graph.add_conditional_edges(
        "rag_retrieve",
        should_web_search,
        {
            "web_search": "web_search",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("web_search", "generate_answer")
    graph.add_edge("generate_answer", END)

    # Planner flow
    graph.add_edge("init_plan", "search_policy")
    graph.add_edge("search_policy", "refine_plan")
    graph.add_edge("refine_plan", END)

    # Emotional — direct to END
    graph.add_edge("emotional_response", END)

    return graph


def get_compiled_graph():
    """Build and compile the graph, ready for invocation."""
    return build_graph().compile()
