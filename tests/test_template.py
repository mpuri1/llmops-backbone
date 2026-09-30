"""Generate a project from the copier template and run its own checks.

A template that produces a failing project breaks every new repo on day one,
so the generated project's lint and tests run here.
"""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd, cwd):
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env["LLMOPS_TRACING"] = "off"
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False)


def test_generated_project_passes_its_own_lint_and_tests(tmp_path):
    (tmp_path / "llmops-backbone").symlink_to(ROOT)  # the template's local source is ../llmops-backbone
    dest = tmp_path / "demo-project"
    gen = run(["uv", "run", "copier", "copy", "--trust", "--defaults",
               "-d", "project_name=demo-project", "-d", "description=Template test",
               "-d", "llmops_kit_source=path", str(ROOT / "template"), str(dest)], ROOT)
    assert gen.returncode == 0, gen.stderr

    for relative in [".gitignore", ".env.example", ".github/workflows/ci.yml",
                     "decision/DECISION_LOG.md", "decision/LEARNING_EXPERIENCE.md", ".copier-answers.yml"]:
        assert (dest / relative).exists(), relative
    assert "decision/" in (dest / ".gitignore").read_text()
    assert ".env\n" in (dest / ".gitignore").read_text()

    for cmd in (["uv", "sync", "-q"], ["uv", "run", "ruff", "check", "."], ["uv", "run", "pytest", "-q"]):
        result = run(cmd, dest)
        assert result.returncode == 0, f"{' '.join(cmd)}\n{result.stdout}\n{result.stderr}"
