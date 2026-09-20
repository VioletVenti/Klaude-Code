"""s02 — tool registry, file tools, and multi-call dispatch."""

import json

from s02_tool_use import file_tools, toolbox
from s02_tool_use.agent import run_agent

from test_s01_agent_loop import FakeClient, FakeMessage, FakeToolCall


def test_every_spec_has_a_handler_and_vice_versa():
    spec_names = {spec["function"]["name"] for spec in toolbox.SPECS}
    assert spec_names == set(toolbox.HANDLERS)


def test_unknown_tool_returns_error_text():
    assert "unknown tool" in toolbox.execute("teleport", {})


def test_bad_arguments_return_error_text():
    assert "Error" in toolbox.execute("read_file", {"nope": 1})


def test_workspace_escape_is_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(file_tools, "WORKSPACE", tmp_path)
    (tmp_path / "in.txt").write_text("inside")
    assert "Error" in file_tools.read_file("../outside.txt")
    assert "Error" in file_tools.write_file("/tmp/escape.txt", "nope")


def test_write_read_roundtrip_with_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(file_tools, "WORKSPACE", tmp_path)
    assert "Wrote" in file_tools.write_file("sub/a.txt", "one\ntwo\nthree\n")
    assert file_tools.read_file("sub/a.txt") == "one\ntwo\nthree"
    out = file_tools.read_file("sub/a.txt", limit=2)
    assert out.startswith("one\ntwo")
    assert "1 more line" in out


def test_edit_requires_unique_match(tmp_path, monkeypatch):
    monkeypatch.setattr(file_tools, "WORKSPACE", tmp_path)
    file_tools.write_file("e.txt", "alpha beta gamma\n")
    assert "not found" in file_tools.edit_file("e.txt", "delta", "x")
    file_tools.write_file("e.txt", "x x x\n")
    assert "matches 3 times" in file_tools.edit_file("e.txt", "x", "y")
    assert "Edited" in file_tools.edit_file("e.txt", "x x", "y y")
    assert file_tools.read_file("e.txt") == "y y x"


def test_glob_recursive_and_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(file_tools, "WORKSPACE", tmp_path)
    for i in range(250):
        (tmp_path / f"d{i // 50}" ).mkdir(exist_ok=True)
        (tmp_path / f"d{i // 50}" / f"f{i:03}.txt").write_text("x")
    out = file_tools.glob("**/*.txt")
    assert "omitted" in out
    sub = file_tools.glob("d0/*.txt")
    assert sub.count("\n") == 49  # 50 files in d0


def test_multiple_tool_calls_in_one_turn_execute_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(file_tools, "WORKSPACE", tmp_path)
    calls = [
        FakeToolCall("c1", "write_file", json.dumps({"path": "m.txt", "content": "v1"})),
        FakeToolCall("c2", "read_file", json.dumps({"path": "m.txt"})),
    ]
    client = FakeClient([FakeMessage(tool_calls=calls), FakeMessage(content="done")])
    state = run_agent("write then read", client=client)
    tool_messages = [m for m in state.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["c1", "c2"]
    assert "v1" in tool_messages[1]["content"]


def test_bash_still_works_through_the_registry():
    assert "klaude" in toolbox.execute("run_bash", {"command": "echo klaude"})
