"""s17 — self-claiming autonomous teammates."""

import json
import time as time_mod

import pytest

from s17_autonomous import tasks, team, toolbox
from s17_autonomous.team import _find_claimable

from test_s01_agent_loop import FakeClient, FakeMessage, FakeToolCall


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("KLAUDE_MAILBOX_DIR", str(tmp_path / "mail"))
    monkeypatch.setenv("KLAUDE_TASKS_DIR", str(tmp_path / "tasks"))
    team.BUS = team.MailBus()
    team.MATES.clear()
    monkeypatch.setattr(team, "IDLE_POLL_INTERVAL", 0.05)
    monkeypatch.setattr(team, "IDLE_TIMEOUT", 1.5)
    yield


def wait_done(name, timeout=10):
    deadline = time_mod.monotonic() + timeout
    while team.MATES[name]["status"] != "done" and time_mod.monotonic() < deadline:
        time_mod.sleep(0.05)
    return team.MATES[name]["status"] == "done"


def test_claimable_scan_filters_status_owner_and_blockers():
    ready = tasks.create("ready task", "do it")
    owned = tasks.create("owned task")
    owned.owner = "someone"
    tasks.save(owned)
    blocked = tasks.create("blocked task")
    tasks.add_blocked_by(blocked.tid, [ready.tid])

    found = _find_claimable()
    assert found.tid == ready.tid  # skips owned and blocked
    tasks.claim(ready.tid, owner="alice")
    assert _find_claimable() is None


def test_teammate_auto_claims_and_completes_board_work():
    tasks.create("write auto.txt", "create auto.txt containing ok")
    script = [
        FakeMessage(content="standing by"),  # -> idle
        FakeMessage(
            tool_calls=[
                FakeToolCall("c1", "write_file", json.dumps({"path": "auto.txt", "content": "ok"})),
                FakeToolCall("c2", "complete_task", json.dumps({"task_id": "t1"})),
            ]
        ),
        FakeMessage(content="task finished"),
    ]
    client = FakeClient(script)
    team.spawn("scout", "worker", "watch the board", client=client)
    assert wait_done("scout")

    # the wake-up carried the self-claimed brief into the model call
    second_call = client.chat.completions.calls[1]
    injected = [
        m for m in second_call["messages"]
        if m.get("role") == "user" and "self-claimed" in str(m.get("content"))
    ]
    assert injected and "write auto.txt" in injected[0]["content"]
    # task closed under the teammate's own name
    assert tasks.load("t1").status == "completed"
    assert tasks.load("t1").owner == "scout"


def test_retirement_sends_summary_with_completed_count():
    parked = tasks.create("parked task")  # owned by someone else -> not claimable
    parked.owner = "someone"
    tasks.save(parked)
    client = FakeClient([FakeMessage(content="standing by")])
    team.spawn("hermit", "worker", "wait", client=client)
    assert wait_done("hermit")  # idle timeout path
    inbox = team.BUS.read_inbox("lead")
    summaries = [m for m in inbox if m["kind"] == "result"]
    assert summaries and "retired after idle timeout" in summaries[-1]["content"]


def test_window_keeps_identity_message_first():
    system = {"role": "system", "content": "identity"}
    filler = [{"role": "user", "content": f"m{i}"} for i in range(30)]
    messages = [system, *filler]
    window = messages if len(messages) <= 20 else [messages[0]] + messages[-19:]
    assert window[0] is system
    assert len(window) == 20


def test_teammate_toolset_has_autonomy_tools():
    names = {spec["function"]["name"] for spec in team.TEAMMATE_SPECS}
    assert {"list_tasks", "claim_task", "complete_task"} <= names
    assert "create_task" not in names  # board writes stay with the lead
