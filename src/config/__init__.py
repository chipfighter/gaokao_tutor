"""Configuration module — YAML settings + XML prompt registry.

Public API:
    load_settings()   — load and cache settings.yaml
    get_setting(key)  — dot-notation access to settings
    load_prompt(name) — load and cache an XML prompt template
    clear_cache()     — invalidate all caches
"""

from src.config.config_manager import clear_cache, get_setting, load_prompt, load_settings

__all__ = [
    "clear_cache",
    "get_setting",
    "load_prompt",
    "load_settings",
]
