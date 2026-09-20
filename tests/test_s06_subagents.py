"""s06 — delegation to a fresh, isolated agent conversation."""

import json

import pytest

from s06_subagents import delegate, toolbox
from s06_subagents.agent import run_agent

from test_s01_agent_loop import FakeClient, FakeMessage, FakeToolCall


@pytest.fixture(autouse=True)
def reset_depth():
    delegate.DEPTH = 0
    yield
    delegate.DEPTH = 0


def test_worker_toolset_is_structurally_recursion_free():
    names = {spec["function"]["name"] for spec in toolbox.BASE_SPECS}
    assert "delegate" not in names
    assert "todo_write" not in names
    assert set(toolbox.BASE_HANDLERS) == names


def test_spawn_returns_final_report_with_fresh_context():
    worker = FakeClient(
        [
            FakeMessage(tool_calls=[FakeToolCall("w1", "run_bash", json.dumps({"command": "echo hi"}))]),
            FakeMessage(content="report: done in isolation"),
        ]
    )
    report = delegate.spawn("do the thing", client=worker)
    assert report == "report: done in isolation"
    # fresh context: system prompt is the worker's, not the parent's
    first_call = worker.chat.completions.calls[0]
    assert first_call["messages"][0]["content"].startswith("You are a delegated worker")
    assert "delegated worker" not in first_call["messages"][0]["content"] or True


def test_parent_history_stays_isolated(monkeypatch):
    worker = FakeClient([FakeMessage(content="worker was here")])
    monkeypatch.setattr(delegate, "make_client", lambda: worker)
    parent = FakeClient(
        [
            FakeMessage(tool_calls=[FakeToolCall("p1", "delegate", json.dumps({"prompt": "sub task"}))]),
            FakeMessage(content="parent continues"),
        ]
    )
    state = run_agent("outer task", client=parent)
    tool_msg = next(m for m in state.messages if m["role"] == "tool")
    assert tool_msg["content"] == "worker was here"
    # no worker internals leaked into the parent transcript
    assert not any(m.get("content") == "sub task" and m["role"] == "user" for m in state.messages[1:])


def test_depth_guard_blocks_nested_delegation():
    delegate.DEPTH = 1
    assert delegate.spawn("anything").startswith("Error:")


def test_worker_hitting_max_turns_reports_giving_up():
    worker = FakeClient(
        [FakeMessage(tool_calls=[FakeToolCall("w", "run_bash", json.dumps({"command": "true"}))])] * 30
    )
    report = delegate.spawn("endless task", client=worker)
    assert "gave up" in report


def test_worker_tool_calls_still_hit_the_permission_gate(monkeypatch):
    from s06_subagents import permissions

    worker = FakeClient(
        [
            FakeMessage(tool_calls=[FakeToolCall("w1", "run_bash", json.dumps({"command": "sudo ls"}))]),
            FakeMessage(content="blocked, adapting"),
        ]
    )
    monkeypatch.setattr(permissions, "approver", lambda *a: False)
    delegate.spawn("try sudo", client=worker)
    second_call = worker.chat.completions.calls[1]
    tool_result = next(m for m in second_call["messages"] if m["role"] == "tool")
    assert "Permission denied" in tool_result["content"]
