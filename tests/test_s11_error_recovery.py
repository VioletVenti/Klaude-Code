"""s11 — recovery: truncation escalation, backoff, model fallback."""

import random

import pytest

from s11_error_recovery import recovery
from s11_error_recovery.agent import AgentState, run_agent
from s11_error_recovery.recovery import RecoveryState

from test_s01_agent_loop import FakeClient, FakeMessage, FakeToolCall


class ScriptedClient:
    """Returns whole completion objects (message + finish_reason)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        inner = self

        class Completions:
            @staticmethod
            def create(**kwargs):
                inner.calls.append(kwargs)
                return inner.responses.pop(0)

        class Chat:
            completions = Completions()

        self.chat = Chat()


class Flaky:
    """create() fails N times with the given error, then delegates."""

    def __init__(self, failures, inner, exc=None):
        self.failures = failures
        self.inner = inner
        self.exc = exc or RuntimeError("429 rate limit")
        self.attempts = 0
        self.models_seen = []

    def create(self, **kwargs):
        self.models_seen.append(kwargs.get("model"))
        self.attempts += 1
        if self.attempts <= self.failures:
            raise self.exc
        return self.inner.create(**kwargs)


def make_completion(content="ok", finish="stop"):
    message = FakeMessage(content=content)
    return type("R", (), {"choices": [type("C", (), {"message": message, "finish_reason": finish})()]})()


def test_backoff_formula_bounds():
    random.seed(1)
    low = recovery.retry_delay(0)
    high = recovery.retry_delay(6)
    assert 0.5 <= low <= 0.625
    assert high <= 32 * 1.25 + 1e-9


def test_retry_after_header_wins():
    class WithRetryAfter(RuntimeError):
        retry_after = 7

    assert recovery.retry_delay(0, WithRetryAfter("429")) == 7.0


def test_transient_classification():
    assert recovery.is_transient(RuntimeError("Error code: 429"))
    assert recovery.is_transient(RuntimeError("Request timed out"))
    assert not recovery.is_transient(RuntimeError("invalid api key"))
    assert recovery.is_overload(RuntimeError("529 overloaded"))


def test_with_retry_absorbs_transient_failures():
    inner = FakeClient([FakeMessage(content="made it")])
    flaky = Flaky(2, inner.chat.completions)
    rec = RecoveryState(model="glm-5.3")
    sleeps = []
    response = recovery.with_retry(flaky.create, rec, sleep=sleeps.append)
    assert response.choices[0].message.content == "made it"
    assert len(sleeps) == 2
    assert flaky.attempts == 3


def test_with_retry_reraises_non_transient_immediately():
    inner = FakeClient([FakeMessage(content="x")])
    flaky = Flaky(3, inner.chat.completions, exc=RuntimeError("invalid api key"))
    with pytest.raises(RuntimeError):
        recovery.with_retry(flaky.create, RecoveryState(model="m"), sleep=lambda s: None)
    assert flaky.attempts == 1


def test_repeated_overload_switches_to_fallback(monkeypatch):
    monkeypatch.setenv("KLAUDE_FALLBACK_MODEL", "backup-model")
    rec = RecoveryState(model="glm-5.3")
    seen_models = []
    state = {"failures": 3}

    def call():
        seen_models.append(rec.model)
        if state["failures"] > 0:
            state["failures"] -= 1
            raise RuntimeError("529 overloaded")
        return make_completion("rescued")

    recovery.with_retry(call, rec, sleep=lambda s: None)
    assert rec.model == "backup-model"
    assert seen_models[-1] == "backup-model"


def test_truncation_first_escalates_then_continues_then_gives_up():
    rec = RecoveryState(model="m")
    assert recovery.note_truncation(rec, 100) == "escalate"
    assert rec.max_tokens == recovery.ESCALATED_MAX_TOKENS
    assert recovery.note_truncation(rec, 900) == "continue"  # good growth
    assert recovery.note_truncation(rec, 1000) == "give_up"  # growth < floor
    assert rec.stalled


def test_loop_escalates_max_tokens_without_appending_partial():
    client = ScriptedClient([make_completion("short", "length"), make_completion("full answer")])
    state = run_agent("count", client=client)
    assert state.last_stop == "end"
    tokens = [c["max_tokens"] for c in client.calls]
    assert tokens[0] == recovery.DEFAULT_MAX_TOKENS
    assert tokens[1] == recovery.ESCALATED_MAX_TOKENS
    # truncated first reply must NOT be in history — request was retried whole
    assert not any(m.get("role") == "assistant" and m.get("content") == "short" for m in state.messages)


def test_loop_continuation_appends_partial_and_prompt():
    client = ScriptedClient(
        [
            make_completion("part one", "length"),  # escalates
            make_completion("part two", "length"),  # continuation path
            make_completion("done"),  # final
        ]
    )
    state = run_agent("long task", client=client)
    assert state.last_stop == "end"
    user_prompts = [m["content"] for m in state.messages if m["role"] == "user" and isinstance(m.get("content"), str)]
    assert any("Resume directly" in p for p in user_prompts)
    assert any(m.get("content") == "part two" for m in state.messages if m["role"] == "assistant")


def test_giving_up_marks_truncated():
    client = ScriptedClient(
        [
            make_completion("a" * 10, "length"),
            make_completion("b" * 10, "length"),
            make_completion("c" * 10, "length"),
            make_completion("d" * 10, "length"),
        ]
    )
    state = run_agent("endless", client=client)
    assert state.last_stop == "truncated"
