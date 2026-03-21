"""Gaokao Tutor — AI-powered tutoring assistant for Chinese Gaokao preparation."""

from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

load_dotenv(Path(__file__).parent / ".env")

from src.database.checkpointer import get_db_uri, make_thread_config
from src.graph.builder import get_compiled_graph
from src.schemas import ChatRequest
from src.tracing import setup_tracing, shutdown_tracing

logger = logging.getLogger(__name__)

# Module-level reference set during lifespan
graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage async resources: tracing, PostgreSQL checkpointer, graph."""
    global graph

    setup_tracing()

    async with AsyncExitStack() as stack:
        checkpointer = None
        db_uri = get_db_uri()

        if db_uri:
            try:
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

                checkpointer = await stack.enter_async_context(
                    AsyncPostgresSaver.from_conn_string(db_uri)
                )
                await checkpointer.setup()
                logger.info("PostgreSQL checkpointer initialized")
            except Exception:
                logger.exception(
                    "Failed to initialize PostgreSQL checkpointer, running stateless"
                )
                checkpointer = None
        else:
            logger.info("DB_URI not set, running without persistent state")

        graph = get_compiled_graph(checkpointer=checkpointer)
        yield

    shutdown_tracing()


app = FastAPI(title="Gaokao Tutor API", lifespan=lifespan)

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

FastAPIInstrumentor.instrument_app(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


ALLOWED_NODES = {"generate_answer", "generate_plan", "emotional_response"}

# All graph nodes whose lifecycle (start/end) we broadcast to the frontend.
GRAPH_NODES = {
    "supervisor",
    "rag_retrieve",
    "web_search",
    "generate_answer",
    "evaluate_hallucination",
    "search_policy",
    "generate_plan",
    "emotional_response",
}


async def generate_sse(
    query: str,
    thread_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream LangGraph events as Server-Sent Events (SSE).

    Yields two SSE payload types:

    * ``{"type": "node_event", "status": "start"|"end", "node": "<name>"}``
      — emitted when a graph node begins or finishes execution.
    * ``{"type": "token", "content": "<text>"}``
      — emitted for each streamed token from an allowed LLM node.

    Args:
        query: The user-provided string to be processed by the graph.
        thread_id: Optional session ID for multi-turn memory. Auto-generated if None.
    """
    config = make_thread_config(thread_id)
    state_input = {"messages": [HumanMessage(content=query)]}

    async for event in graph.astream_events(state_input, config=config, version="v2"):
        event_type = event["event"]

        # ── Node lifecycle events ──────────────────────────────────────
        if event_type in ("on_chain_start", "on_chain_end"):
            node_name = event.get("name")
            meta_node = event.get("metadata", {}).get("langgraph_node")
            # Only emit for top-level graph nodes (name matches metadata),
            # not for internal sub-chains (RunnableSequence, etc.).
            if node_name and node_name == meta_node and node_name in GRAPH_NODES:
                status = "start" if event_type == "on_chain_start" else "end"
                payload = json.dumps(
                    {"type": "node_event", "status": status, "node": node_name},
                    ensure_ascii=False,
                )
                yield f"data: {payload}\n\n"

        # ── Token streaming ────────────────────────────────────────────
        elif event_type == "on_chat_model_stream":
            node_name = event.get("metadata", {}).get("langgraph_node")
            if node_name in ALLOWED_NODES:
                chunk = event["data"]["chunk"]
                if chunk.content:
                    payload = json.dumps(
                        {"type": "token", "content": chunk.content},
                        ensure_ascii=False,
                    )
                    yield f"data: {payload}\n\n"


@app.post("/stream")
async def stream_endpoint(request: ChatRequest):
    return StreamingResponse(
        generate_sse(request.query, thread_id=request.thread_id),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
