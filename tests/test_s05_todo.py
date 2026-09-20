"""s05 — structured planning: todo list validation and stale-plan reminders."""

import json

import pytest

from s05_todo import toolbox
from s05_todo.agent import run_agent
from s05_todo.todos import ACTIVE, TodoList

from test_s01_agent_loop import FakeClient, FakeMessage, FakeToolCall


@pytest.fixture(autouse=True)
def fresh_list():
    ACTIVE.items = []
    yield
    ACTIVE.items = []


def test_full_lifecycle_rendering():
    out = ACTIVE.replace(
        [
            {"content": "scan", "status": "done"},
            {"content": "patch", "status": "active", "active_form": "patching config"},
            {"content": "report", "status": "pending"},
        ]
    )
    assert "[x] scan" in out
    assert "[~] patch — patching config" in out
    assert "[ ] report" in out
    assert "(1/3 done)" in out


def test_json_string_payload_is_accepted():
    out = ACTIVE.replace(json.dumps([{"content": "one", "status": "pending"}]))
    assert "[ ] one" in out


def test_validation_rules():
    with pytest.raises(ValueError):
        ACTIVE.replace([{"content": "x", "status": "wip"}])
    with pytest.raises(ValueError):
        ACTIVE.replace([{"status": "pending"}])  # no content
    with pytest.raises(ValueError):
        ACTIVE.replace(
            [
                {"content": "a", "status": "active"},
                {"content": "b", "status": "active"},
            ]
        )
    with pytest.raises(ValueError):
        ACTIVE.replace([{"content": f"t{i}", "status": "pending"} for i in range(21)])
    with pytest.raises(ValueError):
        ACTIVE.replace("not json at all {{{")


def test_all_done_gets_verification_nudge():
    out = ACTIVE.replace(
        [
            {"content": "write", "status": "done"},
            {"content": "edit", "status": "done"},
            {"content": "test", "status": "done"},
        ]
    )
    assert "Verify the completed work" in out
    assert ACTIVE.items[-1].status == "pending"


def test_verification_item_prevents_nudge():
    ACTIVE.replace(
        [
            {"content": "write", "status": "done"},
            {"content": "verify output", "status": "done"},
            {"content": "edit", "status": "done"},
        ]
    )
    assert all("Verify the completed" not in t.content for t in ACTIVE.items)


def test_registry_routes_todo_write():
    out = toolbox.execute("todo_write", {"todos": [{"content": "via registry", "status": "pending"}]})
    assert "via registry" in out
    bad = toolbox.execute("todo_write", {"todos": [{"content": "x", "status": "nope"}]})
    assert bad.startswith("Error")


def test_stale_plan_triggers_reminder_after_three_rounds():
    # five rounds of plain tool use, no todo_write anywhere
    rounds = [
        FakeMessage(tool_calls=[FakeToolCall(f"c{i}", "run_bash", json.dumps({"command": "true"}))])
        for i in range(5)
    ] + [FakeMessage(content="done")]
    client = FakeClient(rounds)
    state = run_agent("busy work", client=client)
    reminders = [m for m in state.messages if m.get("role") == "user" and "reminder" in m.get("content", "")]
    assert reminders, "expected a stale-plan reminder"
    # reminder lands right after the 3rd round's tool results
    tool_positions = [i for i, m in enumerate(state.messages) if m["role"] == "tool"]
    assert reminders[0] in [state.messages[p + 1] for p in tool_positions]


def test_todo_usage_resets_the_counter():
    script = []
    for i in range(4):
        script.append(
            FakeMessage(
                tool_calls=[
                    FakeToolCall(f"t{i}", "todo_write", json.dumps({"todos": [{"content": "plan", "status": "done"}]})),
                    FakeToolCall(f"c{i}", "run_bash", json.dumps({"command": "true"})),
                ]
            )
        )
    script.append(FakeMessage(content="done"))
    state = run_agent("planned work", client=FakeClient(script))
    reminders = [m for m in state.messages if m.get("role") == "user" and "reminder" in m.get("content", "")]
    assert reminders == []
