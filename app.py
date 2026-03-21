"""Gaokao Tutor — AI-powered tutoring assistant for Chinese Gaokao preparation."""

import json
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

load_dotenv(Path(__file__).parent / ".env")

from src.graph.builder import get_compiled_graph

app = FastAPI(title="Gaokao Tutor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = get_compiled_graph()

class ChatRequest(BaseModel):
    query: str

async def generate_sse(query: str) -> AsyncGenerator[str, None]:
    """
    Streams LangGraph events as Server-Sent Events (SSE).

    This generator consumes asynchronous events from a LangGraph execution 
    and formats specific chat model stream chunks into the SSE protocol 
    format for real-time frontend consumption.

    Args:
        query: The user-provided string to be processed by the graph.

    Yields:
        A string formatted according to the SSE protocol (data: <json>\n\n),
        containing the streaming tokens or a completion marker.

    Raises:
        Exception: If the graph execution fails, the error is caught and 
            yielded as a JSON error payload before closing the stream.
    """
    state_input = {"messages": [HumanMessage(content=query)]}
    
    ALLOWED_NODES = {"generate_answer", "generate_plan", "emotional_response"}
    
    # We use astream_events to catch the LLM streaming tokens
    async for event in graph.astream_events(state_input, version="v2"):
        if event["event"] == "on_chat_model_stream":
            node_name = event.get("metadata", {}).get("langgraph_node")
            if node_name in ALLOWED_NODES:
                chunk = event["data"]["chunk"]
                if chunk.content:
                    # SSE format: data: {"type": "token", "content": "..."}\n\n
                    payload = json.dumps({"type": "token", "content": chunk.content}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"

@app.post("/stream")
async def stream_endpoint(request: ChatRequest):
    return StreamingResponse(
        generate_sse(request.query),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
