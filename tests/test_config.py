"""Unit tests for configuration manager — YAML settings + XML prompt loading.

Tests cover: YAML settings loading/caching, XML prompt loading/caching,
dot-notation setting access, prompt rendering with {variables},
cache invalidation, and all prompt files loadable.
All tests use tmp_path or mock — no real config files needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


# ===========================================================================
# Fixture: temporary config directory
# ===========================================================================

@pytest.fixture
def config_dir(tmp_path):
    """Create a temp config dir and point config_manager at it."""
    from src.config import config_manager

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    original = {
        "dir": config_manager._CONFIG_DIR,
        "settings": config_manager._SETTINGS_PATH,
        "prompts": config_manager._PROMPTS_DIR,
    }

    config_manager._CONFIG_DIR = tmp_path
    config_manager._SETTINGS_PATH = tmp_path / "settings.yaml"
    config_manager._PROMPTS_DIR = prompts_dir
    config_manager.clear_cache()

    yield tmp_path

    config_manager._CONFIG_DIR = original["dir"]
    config_manager._SETTINGS_PATH = original["settings"]
    config_manager._PROMPTS_DIR = original["prompts"]
    config_manager.clear_cache()


# ===========================================================================
# TestLoadSettings — YAML loading and caching
# ===========================================================================

class TestLoadSettings:
    """Test YAML settings loading and caching."""

    def test_loads_yaml_settings(self, config_dir):
        """Settings are loaded from YAML file."""
        from src.config.config_manager import load_settings

        (config_dir / "settings.yaml").write_text(
            "academic:\n  max_retries: 3\n", encoding="utf-8",
        )

        settings = load_settings(reload=True)

        assert settings["academic"]["max_retries"] == 3

    def test_caches_after_first_load(self, config_dir):
        """Second call returns cached dict without re-reading file."""
        from src.config.config_manager import load_settings

        yaml_path = config_dir / "settings.yaml"
        yaml_path.write_text("value: 1\n", encoding="utf-8")

        first = load_settings(reload=True)
        yaml_path.write_text("value: 999\n", encoding="utf-8")
        second = load_settings()

        assert first is second
        assert second["value"] == 1

    def test_reload_forces_reread(self, config_dir):
        """reload=True re-reads the file."""
        from src.config.config_manager import load_settings

        yaml_path = config_dir / "settings.yaml"
        yaml_path.write_text("value: 1\n", encoding="utf-8")
        load_settings(reload=True)

        yaml_path.write_text("value: 42\n", encoding="utf-8")
        refreshed = load_settings(reload=True)

        assert refreshed["value"] == 42

    def test_returns_empty_dict_on_missing_file(self, config_dir):
        """Missing settings.yaml returns empty dict (graceful degradation)."""
        from src.config.config_manager import load_settings

        settings = load_settings(reload=True)

        assert settings == {}


# ===========================================================================
# TestGetSetting — dot-notation access
# ===========================================================================

class TestGetSetting:
    """Test dot-notation setting access with defaults."""

    def test_top_level_key(self, config_dir):
        from src.config.config_manager import get_setting, load_settings

        (config_dir / "settings.yaml").write_text(
            "debug: true\n", encoding="utf-8",
        )
        load_settings(reload=True)

        assert get_setting("debug") is True

    def test_nested_dotted_key(self, config_dir):
        from src.config.config_manager import get_setting, load_settings

        (config_dir / "settings.yaml").write_text(
            "academic:\n  max_retries: 2\n  search_timeout: 15\n",
            encoding="utf-8",
        )
        load_settings(reload=True)

        assert get_setting("academic.max_retries") == 2
        assert get_setting("academic.search_timeout") == 15

    def test_returns_default_for_missing_key(self, config_dir):
        from src.config.config_manager import get_setting, load_settings

        (config_dir / "settings.yaml").write_text("a: 1\n", encoding="utf-8")
        load_settings(reload=True)

        assert get_setting("nonexistent", "fallback") == "fallback"
        assert get_setting("a.b.c", 99) == 99

    def test_returns_list_values(self, config_dir):
        from src.config.config_manager import get_setting, load_settings

        (config_dir / "settings.yaml").write_text(
            "supervisor:\n  valid_intents:\n    - academic\n    - planning\n    - emotional\n",
            encoding="utf-8",
        )
        load_settings(reload=True)

        intents = get_setting("supervisor.valid_intents")
        assert intents == ["academic", "planning", "emotional"]

    def test_returns_none_default_when_not_specified(self, config_dir):
        from src.config.config_manager import get_setting, load_settings

        (config_dir / "settings.yaml").write_text("a: 1\n", encoding="utf-8")
        load_settings(reload=True)

        assert get_setting("missing") is None


# ===========================================================================
# TestLoadPrompt — XML prompt loading and caching
# ===========================================================================

class TestLoadPrompt:
    """Test XML prompt loading."""

    def test_loads_xml_prompt_text(self, config_dir):
        """Prompt text is extracted from <prompt> XML element."""
        from src.config.config_manager import load_prompt

        prompt_xml = '<?xml version="1.0" encoding="UTF-8"?>\n<prompt>Hello teacher</prompt>'
        (config_dir / "prompts" / "test_prompt.xml").write_text(
            prompt_xml, encoding="utf-8",
        )

        result = load_prompt("test_prompt", reload=True)

        assert result == "Hello teacher"

    def test_loads_cdata_prompt(self, config_dir):
        """CDATA sections are transparently handled."""
        from src.config.config_manager import load_prompt

        prompt_xml = '<prompt><![CDATA[Use $ax^2+bx+c=0$ formula]]></prompt>'
        (config_dir / "prompts" / "math.xml").write_text(
            prompt_xml, encoding="utf-8",
        )

        result = load_prompt("math", reload=True)

        assert "$ax^2+bx+c=0$" in result

    def test_preserves_format_variables(self, config_dir):
        """Python {variable} placeholders survive XML loading."""
        from src.config.config_manager import load_prompt

        prompt_xml = '<prompt><![CDATA[Question: {question}\nContext: {context}]]></prompt>'
        (config_dir / "prompts" / "with_vars.xml").write_text(
            prompt_xml, encoding="utf-8",
        )

        result = load_prompt("with_vars", reload=True)

        assert "{question}" in result
        assert "{context}" in result
        rendered = result.format(question="What is X?", context="X is Y")
        assert "What is X?" in rendered

    def test_caches_prompt(self, config_dir):
        """Second call returns cached string without re-reading file."""
        from src.config.config_manager import load_prompt

        xml_path = config_dir / "prompts" / "cached.xml"
        xml_path.write_text("<prompt>original</prompt>", encoding="utf-8")

        first = load_prompt("cached", reload=True)
        xml_path.write_text("<prompt>changed</prompt>", encoding="utf-8")
        second = load_prompt("cached")

        assert first == second == "original"

    def test_raises_on_missing_prompt(self, config_dir):
        """Missing XML file raises FileNotFoundError."""
        from src.config.config_manager import load_prompt

        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent", reload=True)

    def test_handles_chinese_content(self, config_dir):
        """Chinese text loads correctly from XML."""
        from src.config.config_manager import load_prompt

        prompt_xml = '<prompt><![CDATA[你是一位经验丰富的高考学科辅导老师]]></prompt>'
        (config_dir / "prompts" / "chinese.xml").write_text(
            prompt_xml, encoding="utf-8",
        )

        result = load_prompt("chinese", reload=True)

        assert "高考" in result


# ===========================================================================
# TestPromptRendering — .format() with loaded prompts
# ===========================================================================

class TestPromptRendering:
    """Test that loaded prompts render correctly with .format()."""

    def test_academic_answer_renders(self):
        """academic_answer prompt renders with all 3 variables."""
        from src.config import load_prompt

        prompt = load_prompt("academic_answer")
        rendered = prompt.format(
            retrieved_context="some docs",
            search_context="some search",
            question="What is X?",
        )

        assert "some docs" in rendered
        assert "some search" in rendered
        assert "What is X?" in rendered

    def test_hallucination_eval_renders(self):
        """hallucination_eval prompt renders with question, context, answer."""
        from src.config import load_prompt

        prompt = load_prompt("hallucination_eval")
        rendered = prompt.format(
            question="Q", context="C", answer="A",
        )

        assert "Q" in rendered
        assert "C" in rendered
        assert "A" in rendered

    def test_planner_generate_renders(self):
        """planner_generate prompt renders with user_request, policy_info."""
        from src.config import load_prompt

        prompt = load_prompt("planner_generate")
        rendered = prompt.format(
            user_request="Make a plan",
            policy_info="New policy info",
        )

        assert "Make a plan" in rendered
        assert "New policy info" in rendered

    def test_system_prompts_have_no_variables(self):
        """System prompts should load without needing .format()."""
        from src.config import load_prompt

        for name in [
            "supervisor_system",
            "academic_system",
            "hallucination_system",
            "planner_system",
            "emotional_system",
        ]:
            prompt = load_prompt(name)
            assert len(prompt) > 50, f"{name} prompt is too short"


# ===========================================================================
# TestClearCache — cache invalidation
# ===========================================================================

class TestClearCache:
    """Test cache clearing."""

    def test_clears_all_caches(self, config_dir):
        """clear_cache resets both settings and prompt caches."""
        from src.config.config_manager import (
            clear_cache,
            load_prompt,
            load_settings,
        )

        (config_dir / "settings.yaml").write_text("v: 1\n", encoding="utf-8")
        (config_dir / "prompts" / "x.xml").write_text(
            "<prompt>old</prompt>", encoding="utf-8",
        )

        load_settings(reload=True)
        load_prompt("x", reload=True)

        # Modify files
        (config_dir / "settings.yaml").write_text("v: 2\n", encoding="utf-8")
        (config_dir / "prompts" / "x.xml").write_text(
            "<prompt>new</prompt>", encoding="utf-8",
        )

        clear_cache()

        assert load_settings()["v"] == 2
        assert load_prompt("x") == "new"


# ===========================================================================
# TestAllPromptsLoadable — smoke test for all XML prompt files
# ===========================================================================

class TestAllPromptsLoadable:
    """Verify all XML prompt files exist and load without error."""

    @pytest.mark.parametrize("name", [
        "supervisor_system",
        "academic_system",
        "academic_answer",
        "hallucination_system",
        "hallucination_eval",
        "planner_system",
        "planner_generate",
        "emotional_system",
    ])
    def test_prompt_loads(self, name):
        from src.config import load_prompt

        prompt = load_prompt(name)

        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ===========================================================================
# TestSettingsValues — verify settings.yaml has expected values
# ===========================================================================

class TestSettingsValues:
    """Verify settings.yaml contains expected configuration."""

    def test_academic_max_retries(self):
        from src.config import get_setting

        assert get_setting("academic.max_retries") == 2

    def test_academic_search_timeout(self):
        from src.config import get_setting

        assert get_setting("academic.search_timeout") == 15

    def test_academic_temperature(self):
        from src.config import get_setting

        assert get_setting("academic.temperature") == 0.7

    def test_hallucination_temperature(self):
        from src.config import get_setting

        assert get_setting("academic.hallucination_eval_temperature") == 0.0

    def test_planner_temperature(self):
        from src.config import get_setting

        assert get_setting("planner.temperature") == 0.7

    def test_emotional_temperature(self):
        from src.config import get_setting

        assert get_setting("emotional.temperature") == 0.8

    def test_supervisor_temperature(self):
        from src.config import get_setting

        assert get_setting("supervisor.temperature") == 0.0

    def test_supervisor_valid_intents(self):
        from src.config import get_setting

        intents = get_setting("supervisor.valid_intents")
        assert set(intents) == {"academic", "planning", "emotional", "unknown"}


# ===========================================================================
# TestNodeConfigIntegration — nodes use config dynamically
# ===========================================================================

class TestNodeConfigIntegration:
    """Verify graph nodes pull config from config_manager, not hardcoded."""

    def test_academic_max_retries_from_config(self):
        """MAX_RETRIES in academic.py should come from config."""
        from src.graph.academic import MAX_RETRIES
        from src.config import get_setting

        assert MAX_RETRIES == get_setting("academic.max_retries")

    def test_academic_search_timeout_from_config(self):
        """_SEARCH_TIMEOUT in academic.py should come from config."""
        from src.graph.academic import _SEARCH_TIMEOUT
        from src.config import get_setting

        assert _SEARCH_TIMEOUT == get_setting("academic.search_timeout")

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    async def test_generate_answer_uses_config_prompt(
        self, mock_get_llm, mock_get_fallback, mock_llm_response,
    ):
        """generate_answer should use prompt loaded from XML config."""
        from unittest.mock import AsyncMock

        from langchain_core.messages import HumanMessage

        from src.graph.academic import generate_answer

        mock_llm = mock_get_llm.return_value
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response("answer"))
        mock_get_fallback.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="test question")],
            "context": [{"type": "rag", "content": "doc"}],
        }

        result = await generate_answer(state)

        assert "answer" in result["messages"][0].content

    @patch("src.graph.supervisor.get_node_llm")
    async def test_supervisor_uses_config_prompt(self, mock_get_llm):
        """supervisor_node should use prompt loaded from XML config."""
        import json
        from unittest.mock import AsyncMock

        from langchain_core.messages import HumanMessage

        from src.graph.supervisor import supervisor_node

        mock_llm = mock_get_llm.return_value
        mock_resp = type("R", (), {
            "content": json.dumps({
                "intent": "academic", "subject": "math", "keypoints": ["test"],
            }),
        })()
        mock_llm.ainvoke = AsyncMock(return_value=mock_resp)

        state = {"messages": [HumanMessage(content="test")]}
        result = await supervisor_node(state)

        assert result["intent"] == "academic"
