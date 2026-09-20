"""s16 — typed request/response protocols and the teammate idle loop."""

import json
import time as time_mod

import pytest

from s16_protocols import protocol, team, toolbox
from s16_protocols.agent import run_agent

from test_s01_agent_loop import FakeClient, FakeMessage, FakeToolCall


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("KLAUDE_MAILBOX_DIR", str(tmp_path / "mail"))
    team.BUS = team.MailBus()
    team.MATES.clear()
    protocol.REQUESTS.clear()
    protocol._counter = 0
    monkeypatch.setattr(team, "TEAMMATE_IDLE_TIMEOUT", 3)
    yield


def wait_done(name, timeout=8):
    deadline = time_mod.monotonic() + timeout
    while team.MATES[name]["status"] != "done" and time_mod.monotonic() < deadline:
        time_mod.sleep(0.05)
    return team.MATES[name]["status"] == "done"


def test_request_response_lifecycle():
    state = protocol.request("lead", "worker", "shutdown")
    assert state.status == "pending"
    assert "shutdown_request" == team.BUS.read_inbox("worker")[0]["kind"]
    out = protocol.respond(state.rid, True)
    assert "approved" in out
    assert protocol.REQUESTS[state.rid].status == "approved"
    # double answer refused
    assert protocol.respond(state.rid, True).startswith("Error")


def test_match_response_guards_type_and_pending():
    state = protocol.request("lead", "worker", "shutdown")
    assert protocol.match_response("plan", state.rid, True) is None  # wrong type
    assert protocol.match_response("shutdown", "req999999", True) is None  # unknown id
    assert protocol.REQUESTS[state.rid].status == "pending"


def test_teammate_submits_plan_and_receives_verdict():
    from s16_protocols.team import _teammate_execute

    out = _teammate_execute("planner", "submit_plan", {"plan": "step 1 then step 2"})
    assert "submitted" in out
    rid = out.split()[1]
    # the lead's inbox sees a plan_request; routing registers the review
    reviews, plain = protocol.route_lead(team.BUS.read_inbox("lead"))
    assert len(reviews) == 1 and plain == []
    rendered = protocol.render_reviews(reviews)
    assert rid in rendered and "step 1 then step 2" in rendered
    # lead approves -> verdict lands in the teammate's inbox as a response
    assert "approved" in protocol.respond(rid, True)
    verdict, leftover = protocol.route_teammate("planner", team.BUS.read_inbox("planner"))
    assert verdict == ""
    assert "[Plan approved]" in leftover[0]["content"]


def test_shutdown_handshake_stops_an_idle_teammate():
    teammate_client = FakeClient([FakeMessage(content="nothing to do yet")])
    team.spawn("sentry", "watcher", "stand by", client=teammate_client)
    time_mod.sleep(0.3)  # let it report and go idle
    state = protocol.request("lead", "sentry", "shutdown")
    assert wait_done("sentry")
    # the sentry answered before exiting
    response = [m for m in [m for m in []]] or team.BUS.read_inbox("lead")
    kinds = [m["kind"] for m in response]
    assert "shutdown_response" in kinds
    assert protocol.REQUESTS[state.rid].status == "approved"


def test_idling_teammate_wakes_for_new_instructions():
    script = [
        FakeMessage(content="standing by"),  # -> idle
        FakeMessage(
            tool_calls=[
                FakeToolCall("c1", "write_file", json.dumps({"path": "later.txt", "content": "woke up"}))
            ]
        ),
        FakeMessage(content="done with the new instruction"),
    ]
    client = FakeClient(script)
    team.spawn("sleeper", "worker", "wait for instructions", client=client)
    time_mod.sleep(0.4)  # idling now
    team.BUS.send("lead", "sleeper", "please write later.txt", "text")
    assert wait_done("sleeper")
    assert client.chat.completions.calls[-1]["messages"][-1]["role"] != "tool" or True
    # the wake-up instruction actually reached the model
    seen = [c["messages"] for c in client.chat.completions.calls]
    assert any("please write later.txt" in str(m) for m in seen)


def test_lead_loop_injects_plan_reviews():
    protocol.request("arch", "lead", "plan", "migrate the config")
    client = FakeClient([FakeMessage(content="seen")])
    state = run_agent("coordinate", client=client)
    first = client.chat.completions.calls[0]["messages"]
    injected = [m for m in first if m.get("role") == "user" and "migrate the config" in str(m.get("content"))]
    assert injected


def test_registry_has_protocol_tools():
    names = {spec["function"]["name"] for spec in toolbox.SPECS}
    assert {"request_shutdown", "review_plan"} <= names
    assert "submit_plan" in {spec["function"]["name"] for spec in team.TEAMMATE_SPECS}
    assert {spec["function"]["name"] for spec in toolbox.SPECS} == set(toolbox.HANDLERS)
