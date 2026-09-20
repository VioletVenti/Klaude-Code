"""s10 — system prompt as assembled, cached sections."""

import pytest

from s10_system_prompt import prompting
from s10_system_prompt.agent import AgentState, run_agent

from test_s01_agent_loop import FakeClient, FakeMessage


def test_sections_assemble_from_real_state(tmp_path):
    context = {
        "enabled_tools": ["run_bash", "read_file"],
        "workspace": str(tmp_path),
        "skills_catalog": "- notes: writes notes",
        "memory_index": "- [fact](fact.md) — a fact",
        "recalled": '[{"source": "fact.md"}]',
    }
    text = prompting.assemble(context)
    assert text.startswith("You are Klaude")
    assert "Tools available: read_file, run_bash" in text
    assert f"Working directory: {tmp_path}" in text
    assert "Skills installed:" in text
    assert "Memory catalog:" in text
    assert "Recalled memories (" in text


def test_optional_sections_skip_cleanly():
    text = prompting.assemble({"enabled_tools": [], "workspace": "", "skills_catalog": "", "memory_index": "", "recalled": ""})
    assert text.startswith("You are Klaude")
    assert "Skills installed" not in text
    assert "Memory catalog" not in text
    assert "Tools available" not in text


def test_workspace_section_includes_git_state(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("x")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    line = prompting._workspace({"workspace": str(tmp_path)})
    assert "git branch" in line and "clean" in line
    (tmp_path / "b.txt").write_text("y")
    line = prompting._workspace({"workspace": str(tmp_path)})
    assert "1 uncommitted change" in line


def test_cache_hits_reuse_rendered_string():
    context = {"enabled_tools": ["run_bash"], "workspace": "/w", "skills_catalog": "", "memory_index": "", "recalled": ""}
    first = prompting.get(context)
    second = prompting.get(context)
    assert first is second  # same object — no re-render
    prompting.get({**context, "workspace": "/other"})
    assert len(prompting._cache) == 2


def test_update_reads_registries_not_transcript(tmp_path, monkeypatch):
    monkeypatch.setenv("KLAUDE_SKILLS_DIR", str(tmp_path / "none"))
    monkeypatch.setenv("KLAUDE_MEMORY_DIR", str(tmp_path / "mem"))
    state = AgentState(recalled="some memories")
    context = prompting.update(state)
    assert "run_bash" in context["enabled_tools"]
    assert "delegate" in context["enabled_tools"]
    assert context["recalled"] == "some memories"
    assert context["memory_index"] == ""


def test_loop_refreshes_system_each_turn(monkeypatch):
    import s10_system_prompt.memory as memory_mod

    monkeypatch.setattr(memory_mod, "recall", lambda messages: "")
    seen_systems = []

    class Recording(FakeClient):
        pass

    client = Recording(
        [FakeMessage(content="first"), FakeMessage(content="second")]
    )
    original_create = client.chat.completions.create

    def recording_create(**kwargs):
        seen_systems.append(kwargs["messages"][0]["content"])
        return original_create(**kwargs)

    client.chat.completions.create = recording_create
    state = AgentState()
    run_agent("task", state=state, client=client)
    run_agent("another", state=state, client=client)
    assert len(seen_systems) >= 2
    assert all(s.startswith("You are Klaude") for s in seen_systems)
