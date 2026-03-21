"""Gaokao Tutor — AI-powered tutoring assistant for Chinese Gaokao preparation."""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from src.graph.builder import get_compiled_graph

if __name__ == "__main__":
    # Initialize the core backend logic
    graph = get_compiled_graph()
    print("Gaokao Tutor Backend Initialized")
