"""s03 — the permission pipeline gates every tool call."""

import json

from s03_permissions import file_tools, permissions
from s03_permissions.agent import run_agent
from s03_permissions.permissions import Decision

from test_s01_agent_loop import FakeClient, FakeMessage, FakeToolCall


def test_hard_deny_never_reaches_the_human():
    calls = []
    permissions.approver = lambda *a: calls.append(a) or True
    allowed, reason = permissions.check("run_bash", {"command": "sudo rm x"})
    assert not allowed
    assert "deny list" in reason
    assert calls == []  # hard denies never reach the human


def test_destructive_command_asks_and_can_be_refused():
    permissions.approver = lambda *a: False
    allowed, reason = permissions.check("run_bash", {"command": "rm notes.txt"})
    assert not allowed
    assert "destructive" in reason


def test_destructive_command_allowed_on_approval():
    permissions.approver = lambda *a: True
    allowed, _ = permissions.check("run_bash", {"command": "rm notes.txt"})
    assert allowed


def test_write_outside_workspace_asks(tmp_path, monkeypatch):
    monkeypatch.setattr(file_tools, "WORKSPACE", tmp_path)
    decision, reason = permissions.evaluate("write_file", {"path": "../out.txt", "content": "x"})
    assert decision is Decision.ASK
    assert "outside" in reason


def test_benign_calls_pass_silently(tmp_path, monkeypatch):
    monkeypatch.setattr(file_tools, "WORKSPACE", tmp_path)
    asked = []
    permissions.approver = lambda *a: asked.append(a) or True
    for tool, args in [
        ("read_file", {"path": "a.txt"}),
        ("write_file", {"path": "a.txt", "content": "x"}),
        ("run_bash", {"command": "ls -la"}),
        ("run_bash", {"command": "echo hi > local.txt"}),
        ("glob", {"pattern": "**/*.py"}),
    ]:
        allowed, _ = permissions.check(tool, args)
        assert allowed, tool
    assert asked == []


def test_denied_call_feeds_reason_back_to_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(file_tools, "WORKSPACE", tmp_path)
    (tmp_path / "keep.txt").write_text("precious")
    permissions.approver = lambda *a: False
    client = FakeClient(
        [
            FakeMessage(
                tool_calls=[
                    FakeToolCall("c1", "run_bash", json.dumps({"command": "rm keep.txt"}))
                ]
            ),
            FakeMessage(content="cannot delete"),
        ]
    )
    state = run_agent("delete keep.txt", client=client)
    tool_msg = next(m for m in state.messages if m["role"] == "tool")
    assert "Permission denied" in tool_msg["content"]
    assert (tmp_path / "keep.txt").read_text() == "precious"  # never executed
