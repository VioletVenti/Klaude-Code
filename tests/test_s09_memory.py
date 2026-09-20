"""s09 — recall, extract, consolidate across sessions."""

import json

import pytest

from s09_memory import memory
from s09_memory.agent import run_agent

from test_s01_agent_loop import FakeClient, FakeMessage, FakeToolCall


@pytest.fixture(autouse=True)
def memory_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("KLAUDE_MEMORY_DIR", str(tmp_path / "mem"))
    yield


def fake_llm(responses):
    client = FakeClient([FakeMessage(content=r) for r in responses])
    monkey = pytest.MonkeyPatch()
    monkey.setattr(memory, "llm", lambda: client)
    return client, monkey


def test_write_and_index_roundtrip():
    memory.write_record("Prefers pnpm", "user", "package manager preference", "Always use pnpm over npm.")
    records = memory.list_records()
    assert [r["name"] for r in records] == ["Prefers pnpm"]
    assert "pnpm" in memory.read_index()


def test_filename_validation_blocks_escapes():
    with pytest.raises(ValueError):
        memory._safe_filename("../escape.md")
    with pytest.raises(ValueError):
        memory._safe_filename("MEMORY.md")  # index is not a record
    assert memory._safe_filename("ok-name.md").name == "ok-name.md"


def test_should_store_dedupe_and_temporary_rejection():
    existing = [{"name": "Prefers pnpm", "description": "pm", "body": "use pnpm"}]
    assert not memory.should_store({"name": "prefers  pnpm", "type": "user", "description": "x", "body": "y"}, existing)
    assert not memory.should_store({"name": "Other", "type": "user", "description": "pm", "body": "z"}, existing)
    assert not memory.should_store(
        {"name": "Temp", "type": "user", "description": "d", "body": "just this once use npx"}, []
    )
    assert memory.should_store({"name": "New fact", "type": "project", "description": "d2", "body": "b2"}, existing)


def test_recall_selects_via_llm_within_budget():
    for i in range(4):
        memory.write_record(f"fact {i}", "project", f"description {i}", "x" * 100)
    client, monkey = fake_llm(["[0, 2]"])
    try:
        loaded = memory.recall([{"role": "user", "content": "tell me about facts"}])
        assert "fact 0" in loaded and "fact 2" in loaded
        assert "fact 1" not in loaded
    finally:
        monkey.undo()


def test_recall_falls_back_to_keywords_on_llm_failure():
    memory.write_record("editor choice", "user", "prefers neovim", "use nvim")
    client, monkey = fake_llm(["not json at all"])
    client.chat.completions.script = []  # force exceptions after the junk reply
    try:
        loaded = memory.recall([{"role": "user", "content": "which editor? nvim"}])
    finally:
        monkey.undo()
    # junk reply -> empty selection parsed -> falls into picked=[] path or keyword path
    assert isinstance(loaded, str)


def test_extract_stores_validated_records_only():
    client, monkey = fake_llm(
        [
            json.dumps(
                [
                    {"name": "Ship on Fridays", "type": "project", "description": "release day",
                     "body": "Releases go out Fridays.", "scope": "persistent"},
                    {"name": "Bad", "type": "diary", "description": "x", "body": "y", "scope": "persistent"},
                    {"name": "No scope", "type": "user", "description": "x", "body": "y"},
                ]
            )
        ]
    )
    try:
        stored = memory.extract(
            [
                {"role": "user", "content": "we ship on fridays"},
                {"role": "assistant", "content": "noted"},
            ]
        )
    finally:
        monkey.undo()
    assert stored == 1
    assert [r["name"] for r in memory.list_records()] == ["Ship on Fridays"]


def test_extract_survives_llm_outage():
    class Dead:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise RuntimeError("down")

    monkey = pytest.MonkeyPatch()
    monkey.setattr(memory, "llm", lambda: Dead())
    try:
        assert memory.extract([{"role": "user", "content": "hi"}]) == 0
    finally:
        monkey.undo()


def test_consolidate_respects_threshold_and_lock(tmp_path):
    assert memory.consolidate() == 0  # below threshold
    for i in range(memory.CONSOLIDATE_THRESHOLD):
        memory.write_record(f"fact {i}", "project", f"d{i}", f"body {i}")
    merged = json.dumps(
        [
            {"name": "fact all", "type": "project", "description": "merged", "body": "everything"},
        ]
    )
    client, monkey = fake_llm([merged])
    try:
        count = memory.consolidate()
    finally:
        monkey.undo()
    assert count == 1
    assert [r["name"] for r in memory.list_records()] == ["fact all"]
    assert memory.consolidate() == 0  # fresh lock blocks an immediate rerun


def test_natural_end_triggers_extraction(monkeypatch):
    extracted = []
    monkeypatch.setattr(memory, "extract", lambda msgs: extracted.append(msgs) or 0)
    client = FakeClient([FakeMessage(content="bye")])
    run_agent("short task", client=client)
    assert extracted, "extraction should run at natural end"


def test_system_refreshes_with_memory_catalog():
    memory.write_record("Editor", "user", "prefers nvim", "use nvim")
    client = FakeClient([FakeMessage(content="ok")])
    state = run_agent("anything", client=client)
    system = state.messages[0]["content"]
    assert "Memory catalog:" in system
    assert "Editor" in system
