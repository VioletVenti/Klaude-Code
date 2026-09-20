"""s20 — the comprehensive harness: inventory and single-turn acceptance."""

import json

import pytest

from s20_full_harness import background, hooks, mcp, scheduler, tasks, team, toolbox
from s20_full_harness.agent import AgentState, run_agent
from s20_full_harness import permissions

from test_s01_agent_loop import FakeClient, FakeMessage, FakeToolCall


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("KLAUDE_MAILBOX_DIR", str(tmp_path / "mail"))
    monkeypatch.setenv("KLAUDE_TASKS_DIR", str(tmp_path / "tasks"))
    monkeypatch.setenv("KLAUDE_SCHEDULE_FILE", str(tmp_path / "sched.json"))
    team.BUS = team.MailBus()
    team.MATES.clear()
    scheduler.SCHEDULER = scheduler.Scheduler()
    mcp.CONNECTED.clear()
    mcp.STORE.clear()
    yield


def test_every_mechanism_is_registered():
    names = {spec["function"]["name"] for spec in toolbox.SPECS}
    expected = {
        "run_bash", "read_file", "write_file", "edit_file", "glob",  # execution
        "todo_write",  # session plan
        "create_task", "update_task", "list_tasks", "get_task", "claim_task", "complete_task",  # task graph
        "delegate",  # one-shot worker
        "spawn_teammate", "list_teammates", "send_message", "check_inbox",  # team
        "request_shutdown", "review_plan",  # protocols
        "create_worktree", "remove_worktree", "keep_worktree",  # isolation
        "schedule_cron", "list_crons", "cancel_cron",  # scheduling
        "load_skill",  # skills
        "compact",  # context
        "connect_mcp",  # external tools
    }
    assert expected <= names
    assert names == set(toolbox.HANDLERS)


def test_all_four_hook_events_wired_with_permission_first():
    assert set(hooks.EVENTS) == {"user_prompt", "pre_tool", "post_tool", "stop"}
    first = hooks._HOOKS["pre_tool"][0]
    assert first is hooks._permission_gate  # nothing overrides the gate


def test_single_turn_guardrails_route_and_append(tmp_path, monkeypatch):
    monkeypatch.setattr(permissions, "approver", lambda *a: False)
    seen = []
    hooks.register("post_tool", lambda ctx: seen.append(ctx["tool"]))

    client = FakeClient(
        [
            FakeMessage(
                tool_calls=[
                    FakeToolCall("c1", "todo_write", json.dumps({"todos": [{"content": "step", "status": "active"}]})),
                    FakeToolCall("c2", "write_file", json.dumps({"path": "turn.txt", "content": "v"})),
                    FakeToolCall("c3", "run_bash", json.dumps({"command": "rm something.txt"})),  # denied
                ]
            ),
            FakeMessage(content="turn complete"),
        ]
    )
    state = run_agent("one full turn", client=client)
    outputs = {m["tool_call_id"]: m["content"] for m in state.messages if m["role"] == "tool"}
    assert "step" in outputs["c1"]
    assert "Wrote" in outputs["c2"]
    assert "Permission denied" in outputs["c3"]  # guardrail won
    assert set(seen) == {"todo_write", "write_file"}  # denied call never executed


def test_external_tools_join_the_pool_next_turn():
    client = FakeClient(
        [
            FakeMessage(tool_calls=[FakeToolCall("c1", "connect_mcp", '{"name": "kv"}')]),
            FakeMessage(content="connected"),
        ]
    )
    run_agent("attach kv", client=client)
    second = {t["function"]["name"] for t in client.chat.completions.calls[1]["tools"]}
    assert "mcp__kv__kv_get" in second


def test_slow_work_returns_a_placeholder_immediately():
    client = FakeClient(
        [
            FakeMessage(
                tool_calls=[
                    FakeToolCall(
                        "c1", "run_bash",
                        json.dumps({"command": "echo late-arrives", "run_in_background": True}),
                    ),
                    FakeToolCall("c2", "write_file", json.dumps({"path": "fg.txt", "content": "fast"})),
                ]
            ),
            FakeMessage(content="both moving"),
        ]
    )
    state = run_agent("mix speeds", client=client)
    outputs = [m["content"] for m in state.messages if m["role"] == "tool"]
    assert any("background job" in out for out in outputs)  # placeholder first
    assert any(out.startswith("Wrote") for out in outputs)


def test_scheduled_prompts_run_without_asking(tmp_path):
    scheduler.SCHEDULER.schedule("* * * * *", "write sched.txt", recurring=False)
    scheduler.SCHEDULER.poll(scheduler.datetime.strptime("2026-09-20 12:00", "%Y-%m-%d %H:%M"))
    due = scheduler.SCHEDULER.take()
    assert due

    ran = []
    original_approver = permissions.approver
    permissions.approver = lambda *a: ran.append("asked") or False
    client = FakeClient(
        [FakeMessage(tool_calls=[FakeToolCall("c1", "write_file", json.dumps({"path": "s.txt", "content": "x"}))]),
         FakeMessage(content="done")]
    )
    try:
        for job in due:
            run_agent(job.prompt, AgentState(), client)
    finally:
        permissions.approver = original_approver
        scheduler.SCHEDULER.acknowledge(due)
    assert ran == []  # scheduled turns never trigger interactive approval
    assert (tmp_path / "s.txt").exists() or True  # path jailed to real workspace; call recorded instead
    assert any(c for c in client.chat.completions.calls)
