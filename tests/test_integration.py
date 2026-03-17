"""Local integration tests for Gaokao Tutor.

Runs the compiled LangGraph end-to-end against real APIs.
Requires a valid .env with DEEPSEEK_API_KEY and SILICONFLOW_API_KEY.
Also requires chroma_store/ to exist (run scripts/build_index.py first).

Usage:
    python -m tests.test_integration          # run all tests
    python -m tests.test_integration --quick   # skip slow web-search tests
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

# In a VPN environment, httpx (used by ChatOpenAI) needs proxy settings.
# Try to infer it or let the user configure HTTPS_PROXY.
import urllib.request
proxies = urllib.request.getproxies()
if 'http' in proxies and not os.getenv("HTTP_PROXY"):
    os.environ["HTTP_PROXY"] = proxies['http']
if 'https' in proxies and not os.getenv("HTTPS_PROXY"):
    os.environ["HTTPS_PROXY"] = proxies['https']

from langchain_core.messages import AIMessage, HumanMessage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭️ SKIP"

results: list[dict] = []


def _run_test(name: str, fn, *, skip: bool = False) -> None:
    """Execute a single test, record result, print status."""
    if skip:
        results.append({"name": name, "status": SKIP, "time": 0, "detail": "skipped"})
        print(f"  {SKIP}  {name} (skipped)")
        return

    start = time.time()
    try:
        detail = fn()
        elapsed = time.time() - start
        results.append({"name": name, "status": PASS, "time": elapsed, "detail": detail or ""})
        print(f"  {PASS}  {name} ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - start
        results.append({"name": name, "status": FAIL, "time": elapsed, "detail": str(e)})
        print(f"  {FAIL}  {name} ({elapsed:.1f}s) — {e}")
        traceback.print_exc()


def _assert(condition: bool, msg: str = "Assertion failed") -> None:
    if not condition:
        raise AssertionError(msg)


def _invoke(graph, content: str) -> dict:
    """Helper: invoke the graph with a single user message."""
    return graph.invoke({"messages": [HumanMessage(content=content)]})


def _extract_ai_text(result: dict) -> str:
    """Extract the last AIMessage text from graph result."""
    msgs = result.get("messages", [])
    for msg in reversed(msgs):
        if isinstance(msg, AIMessage):
            return msg.content
    return ""


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

def test_supervisor_routing(graph):
    """Verify the supervisor correctly classifies intents."""

    cases = [
        ("二次函数的判别式怎么用？", "academic"),
        ("帮我制定下周复习计划", "planning"),
        ("我好焦虑，感觉学不会", "emotional"),
        ("你好", "emotional"),
    ]

    from src.graph.supervisor import supervisor_node

    for text, expected in cases:
        state = {"messages": [HumanMessage(content=text)]}
        out = supervisor_node(state)
        actual = out.get("intent", "???")
        _assert(
            actual == expected,
            f"Input '{text}': expected '{expected}', got '{actual}'",
        )

    return f"Tested {len(cases)} routing cases"


def test_academic_scenario(graph):
    """Core scenario 1: academic Q&A end-to-end."""
    result = _invoke(graph, "请解释一下二次函数判别式 Δ=b²-4ac 的作用和用法")
    intent = result.get("intent")
    ai_text = _extract_ai_text(result)

    _assert(intent == "academic", f"Expected intent 'academic', got '{intent}'")
    _assert(len(ai_text) > 50, f"Response too short ({len(ai_text)} chars)")
    _assert("判别" in ai_text or "Δ" in ai_text or "delta" in ai_text.lower(),
            "Response doesn't mention discriminant")

    return f"intent={intent}, response_len={len(ai_text)}"


def test_planning_scenario(graph):
    """Core scenario 2: study planning end-to-end."""
    result = _invoke(graph, "我是高三学生，距离高考还有3个月，帮我制定一个每周复习计划")
    intent = result.get("intent")
    ai_text = _extract_ai_text(result)

    _assert(intent == "planning", f"Expected intent 'planning', got '{intent}'")
    _assert(len(ai_text) > 100, f"Response too short ({len(ai_text)} chars)")

    return f"intent={intent}, response_len={len(ai_text)}"


def test_emotional_scenario(graph):
    """Core scenario 3: emotional support end-to-end."""
    result = _invoke(graph, "我最近压力好大，每天晚上都睡不着，觉得自己什么都学不会")
    intent = result.get("intent")
    ai_text = _extract_ai_text(result)

    _assert(intent == "emotional", f"Expected intent 'emotional', got '{intent}'")
    _assert(len(ai_text) > 50, f"Response too short ({len(ai_text)} chars)")

    return f"intent={intent}, response_len={len(ai_text)}"


def test_short_greeting(graph):
    """Edge case: very short greeting input."""
    result = _invoke(graph, "你好")
    intent = result.get("intent")
    ai_text = _extract_ai_text(result)

    _assert(intent == "emotional", f"Greeting should route to emotional, got '{intent}'")
    _assert(len(ai_text) > 5, "Response is empty for greeting")

    return f"intent={intent}, response_len={len(ai_text)}"


def test_empty_input(graph):
    """Edge case: empty input."""
    result = _invoke(graph, "")
    intent = result.get("intent")
    ai_text = _extract_ai_text(result)

    _assert(intent is not None, "Intent should not be None even for empty input")
    _assert(len(ai_text) > 0, "Response should not be empty")

    return f"intent={intent}, response_len={len(ai_text)}"


def test_long_input(graph):
    """Edge case: long input (simulated verbose question)."""
    long_text = (
        "我想问一个关于高考数学的问题。"
        "在解析几何中，椭圆的标准方程为 x²/a² + y²/b² = 1，其中 a > b > 0。"
        "已知椭圆经过点 (1, 3/2)，且离心率 e = √2/2。"
        "请问：(1) 求椭圆的标准方程；"
        "(2) 若直线 l 过椭圆的右焦点且与椭圆交于 A、B 两点，"
        "求 △AOB 面积的最大值（O 为坐标原点）。"
        "请详细写出解题过程，包括每一步的推导。"
    )
    result = _invoke(graph, long_text)
    intent = result.get("intent")
    ai_text = _extract_ai_text(result)

    _assert(intent == "academic", f"Math problem should route to academic, got '{intent}'")
    _assert(len(ai_text) > 100, f"Response too short for detailed math problem ({len(ai_text)} chars)")

    return f"intent={intent}, response_len={len(ai_text)}"


def test_rag_miss_fallback(graph):
    """Edge case: query unlikely to match any indexed documents (triggers web search)."""
    result = _invoke(graph, "请解释量子纠缠的原理以及它在量子计算中的应用")
    intent = result.get("intent")
    ai_text = _extract_ai_text(result)

    _assert(intent == "academic", f"Expected academic, got '{intent}'")
    _assert(len(ai_text) > 50, "Should still generate an answer even on RAG miss")

    search_results = result.get("search_results", [])
    return f"intent={intent}, search_results={len(search_results)}, response_len={len(ai_text)}"


def test_search_unavailable(graph):
    """Edge case: simulate DuckDuckGo unavailable by resetting the search tool singleton."""
    import src.tools.search_tool as st_mod

    # Save and replace the tool with None to force re-init
    original_tool = st_mod._search_tool
    st_mod._search_tool = None

    try:
        result = _invoke(graph, "2026年高考最新政策有什么变化？帮我做个规划")
        ai_text = _extract_ai_text(result)
        _assert(len(ai_text) > 20, "Should still generate response when search is unavailable")
        return f"response_len={len(ai_text)} (graceful degradation)"
    finally:
        st_mod._search_tool = original_tool


# ---------------------------------------------------------------------------
# Security checks (offline, no API calls)
# ---------------------------------------------------------------------------

def test_no_hardcoded_secrets():
    """Security: no hardcoded API keys in source code."""
    import re

    secret_patterns = [
        re.compile(r"sk-[a-zA-Z0-9]{20,}"),
        re.compile(r"(DEEPSEEK_API_KEY|SILICONFLOW_API_KEY)\s*=\s*[\"'][a-zA-Z0-9]"),
    ]

    src_dir = project_root / "src"
    violations = []

    for py_file in src_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for pattern in secret_patterns:
            matches = pattern.findall(content)
            if matches:
                violations.append(f"{py_file.name}: {matches}")

    app_content = (project_root / "app.py").read_text(encoding="utf-8")
    for pattern in secret_patterns:
        matches = pattern.findall(app_content)
        if matches:
            violations.append(f"app.py: {matches}")

    _assert(not violations, f"Hardcoded secrets found: {violations}")
    return "No hardcoded secrets in src/ and app.py"


def test_gitignore_coverage():
    """Security: .env, secrets.toml, chroma_store/ are in .gitignore."""
    gitignore = (project_root / ".gitignore").read_text(encoding="utf-8")

    required = [".env", "secrets.toml", "chroma_store"]
    missing = [r for r in required if r not in gitignore]

    _assert(not missing, f"Missing from .gitignore: {missing}")
    return f"All {len(required)} sensitive paths covered in .gitignore"


def test_chroma_not_tracked():
    """Security: chroma_store/ is not tracked by git."""
    import subprocess

    result = subprocess.run(
        ["git", "ls-files", "chroma_store/"],
        capture_output=True,
        text=True,
        cwd=str(project_root),
    )
    tracked = result.stdout.strip()
    _assert(not tracked, f"chroma_store/ is tracked by git: {tracked}")
    return "chroma_store/ not in git index"


def test_env_not_tracked():
    """Security: .env is not tracked by git."""
    import subprocess

    result = subprocess.run(
        ["git", "ls-files", ".env"],
        capture_output=True,
        text=True,
        cwd=str(project_root),
    )
    tracked = result.stdout.strip()
    _assert(not tracked, f".env is tracked by git: {tracked}")
    return ".env not in git index"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    quick = "--quick" in sys.argv

    print("=" * 60)
    print("  Gaokao Tutor — Integration Tests (v0.1)")
    print("=" * 60)

    # ── Pre-flight checks ─────────────────────────────────────
    print("\n[Pre-flight] Checking environment...")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("  ❌ DEEPSEEK_API_KEY not set. Please configure .env")
        sys.exit(1)
    print(f"  ✅ DEEPSEEK_API_KEY found (ends with ...{api_key[-4:]})")

    sf_key = os.getenv("SILICONFLOW_API_KEY")
    if sf_key:
        print(f"  ✅ SILICONFLOW_API_KEY found (ends with ...{sf_key[-4:]})")
    else:
        print("  ⚠️ SILICONFLOW_API_KEY not set — embedding/RAG tests may fail")

    chroma_dir = project_root / os.getenv("CHROMA_PERSIST_DIR", "chroma_store")
    if chroma_dir.is_dir():
        print(f"  ✅ ChromaDB index found at {chroma_dir}")
    else:
        print(f"  ⚠️ ChromaDB index not found at {chroma_dir} — RAG tests may not work as expected")

    print("\n[Network Context]")
    print(f"  - HTTP_PROXY : {os.getenv('HTTP_PROXY', 'Not Set')}")
    print(f"  - HTTPS_PROXY: {os.getenv('HTTPS_PROXY', 'Not Set')}")

    # ── Security tests (offline) ──────────────────────────────
    print("\n[Security Audit]")
    _run_test("No hardcoded secrets", test_no_hardcoded_secrets)
    _run_test(".gitignore coverage", test_gitignore_coverage)
    _run_test("chroma_store/ not tracked", test_chroma_not_tracked)
    _run_test(".env not tracked", test_env_not_tracked)

    # ── Build graph ───────────────────────────────────────────
    print("\n[Graph Build]")
    try:
        t0 = time.time()
        from src.graph.builder import get_compiled_graph
        graph = get_compiled_graph()
        print(f"  ✅ Graph compiled ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"  ❌ Graph compilation failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── Supervisor routing tests ──────────────────────────────
    print("\n[Supervisor Routing]")
    _run_test("Intent classification (4 cases)", lambda: test_supervisor_routing(graph))

    # ── Core scenario tests ───────────────────────────────────
    print("\n[Core Scenarios]")
    _run_test("Academic Q&A (数学-判别式)", lambda: test_academic_scenario(graph))
    _run_test("Study Planning (复习计划)", lambda: test_planning_scenario(graph))
    _run_test("Emotional Support (压力焦虑)", lambda: test_emotional_scenario(graph))

    # ── Edge case tests ───────────────────────────────────────
    print("\n[Edge Cases]")
    _run_test("Empty input", lambda: test_empty_input(graph))
    _run_test("Short greeting ('你好')", lambda: test_short_greeting(graph))
    _run_test("Long input (解析几何大题)", lambda: test_long_input(graph), skip=quick)
    _run_test("RAG miss fallback (量子纠缠)", lambda: test_rag_miss_fallback(graph), skip=quick)
    _run_test("Search unavailable (DuckDuckGo)", lambda: test_search_unavailable(graph), skip=quick)

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)

    passed = sum(1 for r in results if r["status"] == PASS)
    failed = sum(1 for r in results if r["status"] == FAIL)
    skipped = sum(1 for r in results if r["status"] == SKIP)
    total_time = sum(r["time"] for r in results)

    for r in results:
        print(f"  {r['status']}  {r['name']}")
        if r["detail"] and r["status"] != SKIP:
            print(f"       → {r['detail']}")

    print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped ({total_time:.1f}s)")

    if failed:
        print("\n  ⚠️ Some tests FAILED — review errors above before proceeding.")
        sys.exit(1)
    else:
        print("\n  🎉 All tests passed! Ready for v0.1 release.")
        sys.exit(0)


if __name__ == "__main__":
    main()
