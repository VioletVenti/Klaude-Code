"""s15 — persistent teammates over file mailboxes."""

import json
import time as time_mod

import pytest

from s15_teams import team, toolbox
from s15_teams.agent import run_agent

from test_s01_agent_loop import FakeClient, FakeMessage, FakeToolCall


@pytest.fixture(autouse=True)
def fresh_bus(tmp_path, monkeypatch):
    monkeypatch.setenv("KLAUDE_MAILBOX_DIR", str(tmp_path / "mail"))
    team.BUS = team.MailBus()
    team.MATES.clear()
    yield


def test_bus_send_and_consuming_read():
    team.BUS.send("alice", "lead", "first", "text")
    team.BUS.send("alice", "lead", "second", "result")
    messages = team.BUS.read_inbox("lead")
    assert [m["content"] for m in messages] == ["first", "second"]
    assert team.BUS.read_inbox("lead") == []  # consumed
    rendered = team.MailBus.render(messages)
    assert '<inbox from="alice" kind="result">second</inbox>' in rendered


def test_bus_rejects_bad_names():
    with pytest.raises(ValueError):
        team.BUS.send("../evil", "lead", "x")
    with pytest.raises(ValueError):
        team.BUS.send("alice", "no/slashes", "x")


def test_teammate_reports_to_lead_when_done():
    teammate_client = FakeClient(
        [
            FakeMessage(
                tool_calls=[
                    FakeToolCall("c1", "write_file", json.dumps({"path": "t.txt", "content": "made"}))
                ]
            ),
            FakeMessage(content="wrote t.txt successfully"),
        ]
    )
    out = team.spawn("worker", "builder", "build t.txt", client=teammate_client)
    assert out.startswith("spawned")
    deadline = time_mod.monotonic() + 5
    while team.MATES["worker"]["status"] != "done" and time_mod.monotonic() < deadline:
        time_mod.sleep(0.02)
    assert team.MATES["worker"]["status"] == "done"
    inbox = team.BUS.read_inbox("lead")
    assert inbox and inbox[0]["kind"] == "result"
    assert "t.txt" in inbox[0]["content"]
    # restricted toolset: no spawning, no tasks, no scheduling
    names = {spec["function"]["name"] for spec in team.TEAMMATE_SPECS}
    assert names == {"run_bash", "read_file", "write_file", "send_message"}


def test_teammate_hits_hard_deny_and_cannot_ask():
    teammate_client = FakeClient(
        [
            FakeMessage(
                tool_calls=[
                    FakeToolCall("c1", "run_bash", json.dumps({"command": "sudo ls"})),
                    FakeToolCall("c2", "run_bash", json.dumps({"command": "rm notes.txt"})),
                ]
            ),
            FakeMessage(content="both refused"),
        ]
    )
    team.spawn("guard", "tester", "try dangerous things", client=teammate_client)
    deadline = time_mod.monotonic() + 5
    while team.MATES["guard"]["status"] != "done" and time_mod.monotonic() < deadline:
        time_mod.sleep(0.02)
    second_call = teammate_client.chat.completions.calls[1]
    tool_results = [m for m in second_call["messages"] if m["role"] == "tool"]
    assert "Permission denied" in tool_results[0]["content"]  # hard deny
    assert "cannot ask" in tool_results[1]["content"]  # ask -> auto-deny


def test_lead_inbox_injected_into_loop():
    team.BUS.send("worker", "lead", "teammate says hi", "text")
    client = FakeClient([FakeMessage(content="noted")])
    state = run_agent("anything", client=client)
    first_call = client.chat.completions.calls[0]
    injected = [
        m for m in first_call["messages"]
        if m.get("role") == "user" and "teammate says hi" in str(m.get("content"))
    ]
    assert injected


def test_registry_routes_team_tools():
    assert "spawn_teammate" in {spec["function"]["name"] for spec in toolbox.SPECS}
    listing = toolbox.execute("list_teammates", {})
    assert "no teammates" in listing
    toolbox.execute("send_message", {"to": "lead", "content": "self note"})
    assert "self note" in team.BUS.read_inbox("lead")[0]["content"]


def test_spawn_rejects_duplicate_and_bad_names():
    assert team.spawn("Bad Name!", "r", "p").startswith("Error")
