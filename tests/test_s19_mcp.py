"""s19 — MCP discovery, namespacing, and host-gated dispatch."""

import pytest

from s19_mcp import mcp, toolbox
from s19_mcp.agent import run_agent

from test_s01_agent_loop import FakeClient, FakeMessage, FakeToolCall


@pytest.fixture(autouse=True)
def fresh():
    mcp.CONNECTED.clear()
    mcp.STORE.clear()
    yield


def test_register_validates_names_and_handlers():
    server = mcp.ServerProxy("x")
    with pytest.raises(ValueError):
        server.register([{"name": ""}], {})
    with pytest.raises(ValueError):
        server.register([{"name": "a"}, {"name": "a"}], {"a": lambda: 1})
    with pytest.raises(ValueError):
        server.register([{"name": "a"}], {})
    server.register([{"name": "a"}], {"a": lambda: "ok"})
    assert server.call_tool("a", {}) == "ok"
    assert "unknown tool" in server.call_tool("nope", {})
    assert "MCP error" in mcp.ServerProxy("x").call_tool("a", {})  # unregistered


def test_normalize_and_prefixing():
    assert mcp.normalize("my server!") == "my_server_"
    assert mcp.prefixed("kv", "kv_get") == "mcp__kv__kv_get"
    with pytest.raises(ValueError):
        mcp.normalize("")


def test_connect_discovers_and_pool_namespaces():
    out = mcp.connect("kv")
    assert out.startswith("connected to kv")
    names = mcp.exposed_names()
    assert {"mcp__kv__kv_set", "mcp__kv__kv_get"} <= set(names)
    assert mcp.connect("ghost").startswith("Error")
    assert mcp.connect("kv").startswith("already connected")


def test_dispatch_routes_and_stores():
    mcp.connect("kv")
    out = mcp.dispatch("mcp__kv__kv_set", {"key": "greeting", "value": "hi"}, approver=lambda *a: True)
    assert out == "set greeting"
    assert mcp.dispatch("mcp__kv__kv_get", {"key": "greeting"}) == "hi"  # allow-listed, no approver needed
    assert "not connected" in mcp.dispatch("mcp__ops__uptime", {})
    assert "malformed" in mcp.dispatch("mcp__only_two", {})


def test_confirm_policy_requires_host_approval():
    mcp.connect("ops")
    denied = mcp.dispatch("mcp__ops__uptime", {}, approver=None)
    assert "needs host confirmation" in denied
    allowed = mcp.dispatch("mcp__ops__uptime", {}, approver=lambda *a: True)
    assert "up" in allowed or "unavailable" in allowed
    # disk_free is allow-listed and runs without any approver
    assert "GiB" in mcp.dispatch("mcp__ops__disk_free", {})


def test_loop_uses_connected_mcp_tools_across_turns():
    script = [
        FakeMessage(
            tool_calls=[
                FakeToolCall("c1", "connect_mcp", '{"name": "kv"}'),
                FakeToolCall("c2", "mcp__kv__kv_set", '{"key": "k", "value": "v1"}'),
            ]
        ),
        FakeMessage(content="stored"),
    ]
    client = FakeClient(script)
    state = run_agent("use the kv server", client=client)
    # turn 1 saw only builtin tools; the mcp tool arrived after connect...
    first_tools = {t["function"]["name"] for t in client.chat.completions.calls[0]["tools"]}
    second_tools = {t["function"]["name"] for t in client.chat.completions.calls[1]["tools"]}
    assert "mcp__kv__kv_set" not in first_tools
    assert {"mcp__kv__kv_set", "mcp__kv__kv_get"} <= second_tools
    tool_msg = next(m for m in state.messages if m["role"] == "tool" and m["tool_call_id"] == "c2")
    assert tool_msg["content"] == "set k"


def test_registry_carries_connect_tool():
    names = {spec["function"]["name"] for spec in toolbox.SPECS}
    assert "connect_mcp" in names
    assert names == set(toolbox.HANDLERS)
