"""Security audit tests — run offline, no API keys needed."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestNoHardcodedSecrets:

    SECRET_PATTERNS = [
        re.compile(r"sk-[a-zA-Z0-9]{20,}"),
        re.compile(r"(DEEPSEEK_API_KEY|SILICONFLOW_API_KEY)\s*=\s*[\"'][a-zA-Z0-9]"),
    ]

    def _scan_file(self, filepath: Path) -> list[str]:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        violations = []
        for pattern in self.SECRET_PATTERNS:
            matches = pattern.findall(content)
            if matches:
                violations.append(f"{filepath.name}: {matches}")
        return violations

    def test_no_secrets_in_src(self):
        violations = []
        for py_file in (PROJECT_ROOT / "src").rglob("*.py"):
            violations.extend(self._scan_file(py_file))
        assert not violations, f"Hardcoded secrets found: {violations}"

    def test_no_secrets_in_app(self):
        violations = self._scan_file(PROJECT_ROOT / "app.py")
        assert not violations, f"Hardcoded secrets in app.py: {violations}"


class TestGitignoreCoverage:

    def test_sensitive_paths_in_gitignore(self):
        gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        required = [".env", "secrets.toml", "chroma_store"]
        missing = [r for r in required if r not in gitignore]
        assert not missing, f"Missing from .gitignore: {missing}"

    def test_env_not_tracked(self):
        result = subprocess.run(
            ["git", "ls-files", ".env"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        assert not result.stdout.strip(), ".env is tracked by git"

    def test_chroma_store_not_tracked(self):
        result = subprocess.run(
            ["git", "ls-files", "chroma_store/"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        assert not result.stdout.strip(), "chroma_store/ is tracked by git"


class TestEnvExample:

    def test_env_example_exists(self):
        assert (PROJECT_ROOT / ".env.example").is_file()

    def test_env_example_has_required_keys(self):
        content = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        required = ["DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "SILICONFLOW_API_KEY"]
        for key in required:
            assert key in content, f".env.example missing key: {key}"

    def test_env_example_has_no_real_keys(self):
        content = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        pattern = re.compile(r"sk-[a-zA-Z0-9]{20,}")
        assert not pattern.findall(content), "Real API key found in .env.example"
