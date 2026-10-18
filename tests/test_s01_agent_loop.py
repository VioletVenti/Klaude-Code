"""s01 — the agent loop, tested against a scripted fake model."""

import json

import pytest
from openai import OpenAIError

from s01_agent_loop import bash_tool
from s01_agent_loop.agent import (
    AgentState,
    _assistant_message,
    _clean_text,
    make_client,
    run_agent,
)


class FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, cid, name="run_bash", arguments="{}"):
        self.id = cid
        self.type = "function"
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    def __init__(self, content=None, tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content  # must never reach history


class FakeCompletion:
    def __init__(self, message):
        self.choices = [type("Choice", (), {"message": message, "finish_reason": "stop"})()]


class FakeChat:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.script:
            self._last = self.script.pop(0)
        return FakeCompletion(self._last)


class FakeClient:
    def __init__(self, script):
        self.chat = type("ChatNamespace", (), {"completions": FakeChat(script)})()


def tool_round(cid="call_1", command="echo hi"):
    return FakeMessage(
        content="",
        tool_calls=[FakeToolCall(cid, arguments=json.dumps({"command": command}))],
    )


def test_loop_ends_when_model_stops_calling_tools():
    client = FakeClient([FakeMessage(content="done, nothing to run")])
    state = run_agent("say hi", client=client)
    assert state.last_stop == "end"
    assert len(client.chat.completions.calls) == 1
    assert state.messages[-1]["role"] == "assistant"


def test_tool_call_roundtrip_feeds_result_back():
    client = FakeClient([tool_round(command="echo klaude"), FakeMessage(content="it printed klaude")])
    state = run_agent("run echo klaude", client=client)
    assert state.last_stop == "end"
    assert len(client.chat.completions.calls) == 2
    tool_msg = state.messages[3]  # system, user, assistant, tool
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_1"
    assert "klaude" in tool_msg["content"]


def test_max_turns_gives_up_instead_of_looping_forever():
    state = AgentState(max_turns=3)
    client = FakeClient([tool_round(command="true")] * 10)
    state = run_agent("loop forever", state=state, client=client)
    assert state.last_stop == "max_turns"
    assert len(client.chat.completions.calls) == 3


def test_system_prompt_injected_once():
    client = FakeClient([FakeMessage(content="ok")])
    state = run_agent("anything", client=client)
    assert state.messages[0]["role"] == "system"
    assert "terminal coding agent" in state.messages[0]["content"]
    # second run on same state must not duplicate the system message
    run_agent("again", state=state, client=client)
    assert sum(m["role"] == "system" for m in state.messages) == 1


def test_blocked_commands_never_reach_the_shell():
    assert "blocked" in bash_tool.run_bash("sudo apt install something").lower()
    assert "blocked" in bash_tool.run_bash("rm -rf /").lower()


def test_timeout_returns_error_text(monkeypatch):
    monkeypatch.setattr(bash_tool, "COMMAND_TIMEOUT", 1)
    out = bash_tool.run_bash("sleep 5")
    assert out.startswith("Error:")


def test_long_output_is_truncated():
    out = bash_tool.run_bash("python3 -c \"print('x' * 50000)\"")
    assert len(out) < 20_000
    assert "truncated" in out


def test_reasoning_content_is_dropped():
    msg = FakeMessage(content="answer", reasoning_content="long chain of thought")
    record = _assistant_message(msg)
    assert record == {"role": "assistant", "content": "answer"}
    assert "reasoning" not in record


def test_think_tags_are_stripped():
    assert _clean_text("<think>internal</think>visible") == "visible"
    assert _clean_text(None) == ""


def test_base_url_is_required(monkeypatch):
    monkeypatch.delenv("KLAUDE_BASE_URL", raising=False)
    with pytest.raises(OpenAIError, match="KLAUDE_BASE_URL"):
        make_client()


def test_base_url_comes_from_env(monkeypatch):
    monkeypatch.setenv("KLAUDE_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("KLAUDE_API_KEY", "test")
    assert str(make_client().base_url).startswith("http://localhost:8000/v1")
