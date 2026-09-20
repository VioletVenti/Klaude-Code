"""Make the staged lesson packages importable from the repo root."""

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STAGES = [
    "s01_agent_loop", "s02_tool_use", "s03_permissions", "s04_hooks",
    "s05_todo", "s06_subagents", "s07_skills", "s08_compaction",
    "s09_memory", "s10_system_prompt", "s11_error_recovery", "s12_tasks",
    "s13_background", "s14_scheduler", "s15_teams", "s16_protocols",
    "s17_autonomous", "s18_worktrees", "s19_mcp", "s20_full_harness",
]


@pytest.fixture(autouse=True)
def jailed_workspace(tmp_path, monkeypatch):
    """Every stage's file tools write into the test sandbox, never the repo."""
    for stage in STAGES:
        try:
            module = importlib.import_module(f"{stage}.file_tools")
        except ImportError:
            continue
        monkeypatch.setattr(module, "WORKSPACE", tmp_path)
    yield
