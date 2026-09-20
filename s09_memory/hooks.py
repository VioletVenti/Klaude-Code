"""In-process hooks: extension points around the agent loop.

A hook is any callable taking the event context dict. Returning None
lets things proceed; returning a string means "block" (pre_tool), or
"redirect/annotate" (user_prompt), or "force one more round" (stop).
The loop itself stays free of policy.
"""

from . import permissions

EVENTS = ("user_prompt", "pre_tool", "post_tool", "stop")

_HOOKS: dict[str, list] = {event: [] for event in EVENTS}


def register(event: str, fn) -> None:
    if event not in _HOOKS:
        raise ValueError(f"unknown hook event {event!r}")
    _HOOKS[event].append(fn)


def clear() -> None:
    for event in EVENTS:
        _HOOKS[event].clear()


def emit(event: str, **ctx):
    """Run hooks in registration order; first non-None result wins."""
    for fn in _HOOKS[event]:
        result = fn(ctx)
        if result is not None:
            return result
    return None


# -- Built-in hooks -------------------------------------------------------

def _permission_gate(ctx: dict):
    """pre_tool: the s03 pipeline, now just another hook.

    Registered first on purpose — nothing later in the chain may
    override a denial, and the protocol has no "allow" override.
    """
    allowed, reason = permissions.check(ctx["tool"], ctx["args"])
    return None if allowed else f"Permission denied: {reason}"


def _audit_log(ctx: dict):
    args = ctx.get("args", {})
    preview = str(list(args.values())[:2])[:60]
    print(f"\033[90m[hook] {ctx['tool']}({preview})\033[0m")


def _big_output_warning(ctx: dict):
    output = str(ctx.get("output", ""))
    if len(output) > 20_000:
        print(f"\033[33m[hook] large output from {ctx['tool']}: {len(output)} chars\033[0m")


def _workspace_banner(ctx: dict):
    from . import file_tools

    print(f"\033[90m[hook] workspace: {file_tools.WORKSPACE}\033[0m")


def _session_summary(ctx: dict):
    messages = ctx.get("messages", [])
    tool_calls = sum(1 for m in messages if m.get("role") == "tool")
    print(f"\033[90m[hook] stop: {tool_calls} tool calls this session\033[0m")


def install_defaults() -> None:
    clear()
    register("pre_tool", _permission_gate)  # must stay first — see docstring
    register("pre_tool", _audit_log)
    register("post_tool", _big_output_warning)
    register("user_prompt", _workspace_banner)
    register("stop", _session_summary)


install_defaults()
