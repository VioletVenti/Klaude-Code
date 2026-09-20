"""s13 — background jobs and notification injection."""

import json
import time as time_mod

import pytest

from s13_background import background, toolbox
from s13_background.agent import run_agent

from test_s01_agent_loop import FakeClient, FakeMessage, FakeToolCall


@pytest.fixture(autouse=True)
def fresh_jobs():
    background.JOBS = background.BackgroundJobs()
    yield


def wait_for(job_id, timeout=5.0):
    deadline = time_mod.monotonic() + timeout
    while time_mod.monotonic() < deadline:
        drained = background.JOBS.drain()
        if drained:
            return drained
        time_mod.sleep(0.02)
    raise TimeoutError("job never finished")


def test_start_runs_command_and_drains_notification():
    job_id = background.JOBS.start("echo bg-done")
    assert job_id == "job1"
    assert background.JOBS.pending() == ["job1"]
    notes = wait_for(job_id)
    assert len(notes) == 1
    assert "<background-job id=job1 status=completed>" in notes[0]
    assert "bg-done" in notes[0]
    assert background.JOBS.drain() == []  # delivered once
    assert background.JOBS.pending() == []


def test_failed_command_reports_failed_status():
    background.JOBS.start("exit 3")
    notes = wait_for("job1")
    assert "status=failed" in notes[0]


def test_empty_command_rejected():
    with pytest.raises(ValueError):
        background.JOBS.start("   ")


def wait_idle(timeout=5.0):
    deadline = time_mod.monotonic() + timeout
    while background.JOBS.pending() and time_mod.monotonic() < deadline:
        time_mod.sleep(0.02)


def test_loop_defers_background_result_to_next_turn(tmp_path):
    client = FakeClient(
        [
            FakeMessage(
                tool_calls=[
                    FakeToolCall(
                        "c1",
                        "run_bash",
                        json.dumps({"command": "echo later-result", "run_in_background": True}),
                    ),
                    FakeToolCall("c2", "run_bash", json.dumps({"command": "echo now"})),
                ]
            ),
            FakeMessage(content="all done"),
            FakeMessage(content="finished"),
        ]
    )
    state = run_agent("mix", client=client)
    tool_outputs = [m["content"] for m in state.messages if m["role"] == "tool"]
    assert any("background job job1 started" in out for out in tool_outputs)
    assert any(out == "now" for out in tool_outputs)
    wait_idle()  # let the thread finish, but do not drain
    run_agent("check", state=state, client=client)  # drains + injects before the call
    injected = [m for m in state.messages if m["role"] == "user" and "background-job" in str(m.get("content"))]
    assert injected


def test_background_still_hits_permission_gate(monkeypatch):
    from s13_background import permissions

    monkeypatch.setattr(permissions, "approver", lambda *a: False)
    client = FakeClient(
        [
            FakeMessage(
                tool_calls=[
                    FakeToolCall(
                        "c1",
                        "run_bash",
                        json.dumps({"command": "sudo rm x", "run_in_background": True}),
                    ),
                ]
            ),
            FakeMessage(content="blocked then"),
        ]
    )
    state = run_agent("try sneaky background", client=client)
    tool_msg = next(m for m in state.messages if m["role"] == "tool")
    assert "Permission denied" in tool_msg["content"]
    assert background.JOBS.pending() == []  # never started


def test_spec_documents_the_flag():
    spec = next(s for s in toolbox.SPECS if s["function"]["name"] == "run_bash")
    assert "run_in_background" in spec["function"]["parameters"]["properties"]
