"""MCP-style external tools: discover, namespace, and gate them.

A server proxy stands in for the protocol handshake: register its tool
definitions and handlers, then connect_mcp pulls them into the pool
under mcp__<server>__<tool> names. Authorization is a host decision —
server descriptions never grant anything.
"""

import re
import shutil

from . import file_tools

NAME_SCRUB = re.compile(r"[^a-zA-Z0-9_-]")
MAX_TOOL_NAME = 64

# What the host allows; anything unlisted needs interactive confirmation.
HOST_POLICY = {
    ("kv", "kv_get"): "allow",
    ("kv", "kv_set"): "allow",
    ("ops", "disk_free"): "allow",
}

STORE: dict[str, str] = {}  # backing store for the demo kv server


class ServerProxy:
    """In-process stand-in for tools/list and tools/call."""

    def __init__(self, name: str):
        self.name = name
        self.tools: list[dict] = []
        self._handlers: dict[str, callable] = {}

    def register(self, tool_defs: list, handlers: dict) -> None:
        names = [tool.get("name") for tool in tool_defs]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("every MCP tool needs a non-empty name")
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate tool name on server {self.name!r}")
        missing = [name for name in names if name not in handlers]
        if missing:
            raise ValueError(f"missing handlers: {', '.join(missing)}")
        self.tools = list(tool_defs)
        self._handlers = dict(handlers)

    def call_tool(self, tool_name: str, args: dict) -> str:
        handler = self._handlers.get(tool_name)
        if handler is None:
            return f"MCP error: unknown tool {tool_name!r}"
        try:
            return str(handler(**args))
        except Exception as exc:
            return f"MCP error: {type(exc).__name__}: {exc}"


CONNECTED: dict[str, ServerProxy] = {}


def normalize(name: str) -> str:
    scrubbed = NAME_SCRUB.sub("_", name)
    if not scrubbed:
        raise ValueError("name normalizes to empty")
    return scrubbed


def _kv_set(key: str, value: str) -> str:
    STORE[key] = value
    return f"set {key}"


def _kv_get(key: str) -> str:
    return STORE.get(key, "(unset)")


def _disk_free() -> str:
    usage = shutil.disk_usage(str(file_tools.WORKSPACE))
    return f"{usage.free // (1 << 30)} GiB free of {usage.total // (1 << 30)} GiB"


def _uptime() -> str:
    try:
        seconds = float(open("/proc/uptime").read().split()[0])
        return f"up {int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"
    except OSError:
        return "uptime unavailable"


def available_servers() -> dict[str, ServerProxy]:
    kv = ServerProxy("kv")
    kv.register(
        [
            {
                "name": "kv_set",
                "description": "Store a value under a key.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                    "required": ["key", "value"],
                },
            },
            {
                "name": "kv_get",
                "description": "Read the value stored under a key.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            },
        ],
        {"kv_set": _kv_set, "kv_get": _kv_get},
    )
    ops = ServerProxy("ops")
    ops.register(
        [
            {
                "name": "disk_free",
                "description": "Report free disk space for the workspace.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "uptime",
                "description": "Report how long the host has been up.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ],
        {"disk_free": _disk_free, "uptime": _uptime},
    )
    return {"kv": kv, "ops": ops}


def connect(name: str) -> str:
    if name in CONNECTED:
        return f"already connected to {name}: {exposed_names() and [n for n in exposed_names() if n.startswith(f'mcp__{name}__')]}"
    registry = available_servers()
    if name not in registry:
        available = ", ".join(registry) or "none"
        return f"Error: unknown MCP server {name!r}; available: {available}"
    CONNECTED[name] = registry[name]
    names = [prefixed(name, tool["name"]) for tool in registry[name].tools]
    print(f"\033[94m[mcp] connected {name} -> {', '.join(names)}\033[0m")
    return f"connected to {name}; tools: {', '.join(names)}"


def prefixed(server: str, tool: str) -> str:
    return f"mcp__{normalize(server)}__{normalize(tool)}"


def pool_specs() -> list:
    """Builtin-shaped specs for every connected server tool."""
    specs = []
    for server_name, server in CONNECTED.items():
        for tool_def in server.tools:
            full = prefixed(server_name, tool_def["name"])
            if len(full) > MAX_TOOL_NAME:
                raise ValueError(f"MCP tool name too long: {full!r}")
            schema = tool_def.get("inputSchema", {})
            if not isinstance(schema, dict) or schema.get("type", "object") != "object":
                raise ValueError(f"invalid input schema for {server_name}/{tool_def['name']}")
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": full,
                        "description": tool_def.get("description", ""),
                        "parameters": schema,
                    },
                }
            )
    return specs


def exposed_names() -> list:
    return [spec["function"]["name"] for spec in pool_specs()]


def dispatch(full_name: str, args: dict, approver=None) -> str:
    """Route an mcp__ prefixed call, gated by the host policy."""
    parts = full_name.split("__")
    if len(parts) != 3:
        return f"MCP error: malformed tool name {full_name!r}"
    _, server_name, tool_name = parts
    server = CONNECTED.get(server_name)
    if server is None:
        return f"MCP error: server {server_name!r} is not connected"
    raw = next((tool["name"] for tool in server.tools if prefixed(server_name, tool["name"]) == full_name), None)
    if raw is None:
        return f"MCP error: unknown tool {full_name!r}"
    if HOST_POLICY.get((server_name, raw), "confirm") == "confirm":
        if approver is None or not approver(full_name, args):
            return "Permission denied: this MCP tool needs host confirmation"
    return server.call_tool(raw, args)
