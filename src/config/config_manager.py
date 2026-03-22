"""Configuration manager — loads YAML settings and XML prompt templates.

Provides a cached, thread-safe interface for accessing system parameters
and prompt strings throughout the application.
"""

from __future__ import annotations

import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
_SETTINGS_PATH = _CONFIG_DIR / "settings.yaml"
_PROMPTS_DIR = _CONFIG_DIR / "prompts"

_cache_lock = threading.Lock()
_settings_cache: dict | None = None
_prompt_cache: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_xml_prompt(path: Path) -> str:
    tree = ET.parse(path)
    root = tree.getroot()
    text = "".join(root.itertext())
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_settings(*, reload: bool = False) -> dict:
    """Load and cache settings from settings.yaml.

    Returns an empty dict if the file does not exist (graceful degradation).
    """
    global _settings_cache
    with _cache_lock:
        if _settings_cache is None or reload:
            try:
                _settings_cache = _load_yaml(_SETTINGS_PATH)
            except FileNotFoundError:
                _settings_cache = {}
        return _settings_cache


def get_setting(key: str, default: Any = None) -> Any:
    """Access a setting by dot-notation key (e.g. ``academic.max_retries``).

    Returns *default* if the key path does not exist.
    """
    settings = load_settings()
    current: Any = settings
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def load_prompt(name: str, *, reload: bool = False) -> str:
    """Load and cache an XML prompt template by name.

    Looks for ``config/prompts/{name}.xml``.  Raises ``FileNotFoundError``
    if the file does not exist.
    """
    with _cache_lock:
        if name not in _prompt_cache or reload:
            path = _PROMPTS_DIR / f"{name}.xml"
            if not path.exists():
                raise FileNotFoundError(f"Prompt file not found: {path}")
            _prompt_cache[name] = _load_xml_prompt(path)
        return _prompt_cache[name]


def clear_cache() -> None:
    """Invalidate all cached settings and prompts."""
    global _settings_cache
    with _cache_lock:
        _settings_cache = None
        _prompt_cache.clear()
