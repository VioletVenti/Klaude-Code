"""s08 — escalating context compaction."""

import json

import pytest

from s08_compaction import compact, toolbox
from s08_compaction.agent import AgentState, run_agent

from test_s01_agent_loop import FakeClient, FakeMessage, FakeToolCall


@pytest.fixture(autouse=True)
def archive_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("KLAUDE_ARCHIVE_DIR", str(tmp_path / "archive"))
    compact.REQUESTED = False
    yield
    compact.REQUESTED = False


def make_compactor(summaries=None):
    fake = FakeClient([FakeMessage(content=summaries or "the summary")])
    return compact.Compactor(client=fake), fake


def tool_message(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def call_message(cid="c1", name="run_bash"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": "{}"}}],
    }


def test_budget_batch_persists_huge_results():
    compactor, _ = make_compactor()
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "run_bash", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "run_bash", "arguments": "{}"}},
            ],
        },
        tool_message("c1", "x" * 130_000),
        tool_message("c2", "small"),
    ]
    out = compactor.budget_batch(messages)
    persisted = out[-2]["content"]
    assert persisted.startswith("[output saved at ")
    assert "preview:" in persisted
    path = compact.PREVIEW_RE.match(persisted.strip()).group(1)
    assert len(open(path).read()) == 130_000
    assert out[-1]["content"] == "small"  # small results untouched


def test_snip_archives_middle_and_keeps_pairs():
    compactor, _ = make_compactor()
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "start"}]
    for i in range(30):
        messages.append(call_message(f"a{i}"))
        messages.append(tool_message(f"a{i}", f"result {i}"))
    out = compactor.snip(messages)
    assert len(out) < len(messages)  # middle archived, head+tail kept
    marker_index = next(
        i for i, m in enumerate(out)
        if m["role"] == "user" and "archived at" in m.get("content", "")
    )
    # head boundary may extend past tool results to keep pairs whole — the
    # message right after the archive marker must start a fresh pair
    seam = out[marker_index + 1]
    assert seam["role"] in ("assistant", "user")


def test_point_old_results_keeps_recent_three():
    compactor, _ = make_compactor()
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for i in range(6):
        messages.append(call_message(f"t{i}"))
        messages.append(tool_message(f"t{i}", "payload " * 30))
    target = 0  # force maximal pointing
    out = compactor.point_old_results(messages, target)
    contents = [m["content"] for m in out if m["role"] == "tool"]
    pointers = sum(c.startswith("[earlier output saved at ") for c in contents)
    assert pointers == 3  # six results, newest three kept


def test_summarizer_hardening_and_tag_extraction():
    compactor, fake = make_compactor()
    fake.chat.completions.script.insert(
        0,
        FakeMessage(content="<analysis>thinking</analysis>\n<summary>goal: ship it</summary>"),
    )
    summary = compactor._summarize([{"role": "user", "content": "hi"}])
    assert summary == "goal: ship it"
    system_seen = fake.chat.completions.calls[0]["messages"][0]["content"]
    assert "TEXT ONLY" in system_seen


def test_compact_rewrites_history_to_single_summary():
    compactor, _ = make_compactor("state: halfway done")
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "original task"},
        call_message(),
        tool_message("c1", "out"),
    ]
    out = compactor.compact(messages, "original task")
    assert len(out) == 1
    assert out[0]["role"] == "user"
    assert "state: halfway done" in out[0]["content"]
    assert "original task" in out[0]["content"]
    assert compact.REQUESTED is False


def test_compact_survives_summarizer_failure():
    class ExplodingClient:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("summarizer down")

    compactor = compact.Compactor(client=ExplodingClient())
    out = compactor.compact([{"role": "user", "content": "x"}], "req")
    assert "summarizer failed" in out[0]["content"]


def test_reactive_keeps_recent_tail():
    compactor, _ = make_compactor("earlier state")
    messages = [{"role": "user", "content": f"m{i}"} for i in range(10)]
    out = compactor.reactive(messages, "req")
    assert out[0]["content"].startswith("[Reactive compact]")
    assert len(out) == 1 + compact.KEEP_RECENT_MESSAGES


def test_prepare_escalates_until_small_enough():
    compactor, _ = make_compactor("condensed")
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for i in range(40):
        messages.append(call_message(f"t{i}"))
        messages.append(tool_message(f"t{i}", "z" * 5_000))
    out = compactor.prepare(messages, "u")
    assert compactor.size(out) <= compact.CONTEXT_CHAR_LIMIT


def test_loop_recovers_from_prompt_too_long():
    # a client whose first create raises, then behaves
    class FlakyChat:
        def __init__(self, inner, flag):
            self.inner, self.flag = inner, flag

        def create(self, **kwargs):
            if not self.flag["raised"]:
                self.flag["raised"] = True
                raise RuntimeError("prompt_too_long: reduce input")
            return self.inner.create(**kwargs)

    inner = FakeClient([FakeMessage(content="recovered")])
    flag = {"raised": False}
    client = FakeClient([FakeMessage(content="x")])
    client.chat.completions = FlakyChat(inner.chat.completions, flag)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(compact.COMPACTOR, "client", FakeClient([FakeMessage(content="sum")]))
    try:
        state = run_agent("big task", client=client)
        assert state.last_stop == "end"
        assert state.reactive_used is True
        assert any("Reactive compact" in m.get("content", "") for m in state.messages if m["role"] == "user")
    finally:
        monkey.undo()


def test_compact_tool_triggers_post_batch_compaction():
    compact.COMPACTOR.client = FakeClient([FakeMessage(content="post-batch summary")])
    assert "scheduled" in toolbox.execute("compact", {})
    assert compact.REQUESTED is True
    client = FakeClient(
        [
            FakeMessage(tool_calls=[FakeToolCall("c1", "compact", "{}")]),
            FakeMessage(content="fresh and small"),
        ]
    )
    state = run_agent("shrink me", client=client)
    assert compact.REQUESTED is False
    assert any(m["role"] == "user" and m["content"].startswith("[Compacted]") for m in state.messages)
