"""Graph construction — assemble Supervisor + 3 branches, compile."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.graph.academic import (
    academic_router,
    evaluate_hallucination,
    generate_answer,
    rag_retrieve,
    should_retry_or_end,
    web_search,
)
from src.graph.emotional import emotional_response
from src.graph.planner import generate_plan, search_policy
from src.graph.state import TutorState
from src.graph.supervisor import route_by_intent, supervisor_node


def build_graph() -> StateGraph:
    """Construct the full LangGraph StateGraph (uncompiled)."""

    # Build graph
    graph = StateGraph(TutorState)

    # ── Nodes ────────────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor_node)

    # SubGraph A — Academic (parallel retrieval + answer generation)
    graph.add_node("academic_router", academic_router)
    graph.add_node("rag_retrieve", rag_retrieve)
    graph.add_node("web_search", web_search)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("evaluate_hallucination", evaluate_hallucination)

    # SubGraph B — Planner (search first, then single-call plan)
    graph.add_node("search_policy", search_policy)
    graph.add_node("generate_plan", generate_plan)

    # Emotional
    graph.add_node("emotional_response", emotional_response)

    # ── Edges ────────────────────────────────────────────────────────
    graph.set_entry_point("supervisor")

    # Conditional fork edges
    graph.add_conditional_edges(
        "supervisor",
        route_by_intent,    # judge users intent
        {
            "academic": "academic_router",
            "planning": "search_policy",
            "emotional": "emotional_response",
        },
    )

    # Academic flow — fan-out/fan-in parallel retrieval
    graph.add_edge("academic_router", "rag_retrieve")
    graph.add_edge("academic_router", "web_search")

    # Fan-in: both converge at generate_answer
    graph.add_edge("rag_retrieve", "generate_answer")
    graph.add_edge("web_search", "generate_answer")

    # Hallucination evaluation with retry loop
    graph.add_edge("generate_answer", "evaluate_hallucination")
    graph.add_conditional_edges(
        "evaluate_hallucination",
        should_retry_or_end,
        {
            "retry": "academic_router",
            "end": END,
        },
    )

    # Planner flow (search → generate in 2 steps)
    graph.add_edge("search_policy", "generate_plan")
    graph.add_edge("generate_plan", END)

    # Emotional — direct to END
    graph.add_edge("emotional_response", END)

    return graph


def get_compiled_graph(checkpointer=None):
    """Build and compile the graph, ready for invocation.

    Args:
        checkpointer: Optional LangGraph checkpointer for persistent state.
                      When provided, the graph saves/restores state per thread_id.
    """
    return build_graph().compile(checkpointer=checkpointer)
