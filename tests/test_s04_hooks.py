"""s04 — hooks as the loop's only extension mechanism."""

import json

import pytest

from s04_hooks import file_tools, hooks, permissions
from s04_hooks.agent import run_agent

from test_s01_agent_loop import FakeClient, FakeMessage, FakeToolCall


@pytest.fixture(autouse=True)
def clean_hooks():
    hooks.install_defaults()
    permissions.approver = lambda *a: False
    yield
    hooks.install_defaults()


def test_emit_runs_in_order_and_first_block_wins():
    hooks.clear()
    hooks.register("pre_tool", lambda ctx: None)
    hooks.register("pre_tool", lambda ctx: "first block")
    hooks.register("pre_tool", lambda ctx: "second block")
    assert hooks.emit("pre_tool", tool="run_bash", args={}) == "first block"


def test_unknown_event_rejected():
    with pytest.raises(ValueError):
        hooks.register("mid_tool", lambda ctx: None)


def test_user_prompt_hook_can_rewrite_the_prompt():
    hooks.clear()
    hooks.register("user_prompt", lambda ctx: ctx["prompt"].upper() + " (annotated)")
    client = FakeClient([FakeMessage(content="ok")])
    state = run_agent("hello", client=client)
    assert state.messages[1]["content"] == "HELLO (annotated)"


def test_permission_now_runs_through_the_hook_path(tmp_path, monkeypatch):
    monkeypatch.setattr(file_tools, "WORKSPACE", tmp_path)
    client = FakeClient(
        [
            FakeMessage(tool_calls=[FakeToolCall("c1", "run_bash", json.dumps({"command": "sudo ls"}))]),
            FakeMessage(content="cannot"),
        ]
    )
    state = run_agent("list with sudo", client=client)
    tool_msg = next(m for m in state.messages if m["role"] == "tool")
    assert "Permission denied" in tool_msg["content"]


def test_later_hooks_cannot_override_a_denial():
    # register an enthusiastic "approver" AFTER the permission gate
    hooks.register("pre_tool", lambda ctx: "blocked by late hook")
    allowed_reason = permissions.check  # noqa: F841  (sanity: gate still exists)
    hooks._HOOKS["pre_tool"].insert(0, lambda ctx: "should win")  # even first can't allow
    result = hooks.emit("pre_tool", tool="run_bash", args={"command": "sudo ls"})
    assert "should win" == result  # first non-None wins; no allow-override exists


def test_post_tool_hook_sees_output():
    seen = []
    hooks.register("post_tool", lambda ctx: seen.append((ctx["tool"], ctx["output"])))
    client = FakeClient(
        [
            FakeMessage(tool_calls=[FakeToolCall("c1", "run_bash", json.dumps({"command": "echo done"}))]),
            FakeMessage(content="ok"),
        ]
    )
    run_agent("echo", client=client)
    assert seen and seen[0][0] == "run_bash" and "done" in seen[0][1]


def test_stop_hook_forces_exactly_one_more_round():
    forced = []

    def stop_hook(ctx):
        if not forced:
            forced.append(True)
            return "Before you finish: state how many files you touched."

    hooks.register("stop", stop_hook)
    client = FakeClient([FakeMessage(content="first answer"), FakeMessage(content="final answer")])
    state = run_agent("do something", client=client)
    assert len(client.chat.completions.calls) == 2
    user_turns = [m for m in state.messages if m["role"] == "user"]
    assert any("Before you finish" in m["content"] for m in user_turns)
    assert state.last_stop == "end"


def test_stop_hook_cannot_loop_forever():
    hooks.register("stop", lambda ctx: "again!")  # always forces
    state_maker = FakeClient([FakeMessage(content="x")] * 50)
    state = run_agent("loop test", client=state_maker)
    # stop_hook_used caps forced continuations at one despite the hook insisting
    assert state.last_stop == "end"
    assert len(state_maker.chat.completions.calls) == 2
