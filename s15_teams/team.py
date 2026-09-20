"""Persistent teammates: independent agent threads with file mailboxes.

One lead plus N teammates. A teammate runs on a daemon thread with its
own history, a small toolset (no spawning, no scheduling, no plans),
reads its inbox each turn, and files a final report to the lead when it
stops. Messages are one JSON line per delivery under .mailboxes/.
"""

import json
import os
import re
import threading

from . import bash_tool, permissions, toolbox

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
TEAMMATE_MAX_TURNS = 10
LEAD = "lead"

_send_message_spec = {
    "type": "function",
    "function": {
        "name": "send_message",
        "description": "Deliver a message to another agent's inbox.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["to", "content"],
        },
    },
}

TEAMMATE_SPECS = [
    bash_tool.SPEC,
    toolbox.READ_SPEC,
    toolbox.WRITE_SPEC,
    _send_message_spec,
]


def _mailbox_dir():
    from pathlib import Path

    root = Path(os.environ.get("KLAUDE_MAILBOX_DIR") or toolbox.file_tools.WORKSPACE / ".mailboxes")
    root.mkdir(parents=True, exist_ok=True)
    return root


class MailBus:
    """One append-only file per recipient; reading consumes."""

    def __init__(self):
        self._lock = threading.Lock()

    def _path(self, agent: str):
        if not NAME_RE.fullmatch(agent or ""):
            raise ValueError(f"invalid mailbox name: {agent!r}")
        return _mailbox_dir() / f"{agent}.jsonl"

    def send(self, sender: str, recipient: str, content: str, kind: str = "text") -> None:
        if not NAME_RE.fullmatch(sender or ""):
            raise ValueError(f"invalid sender name: {sender!r}")
        record = {"from": sender, "to": recipient, "kind": kind, "content": str(content)}
        with self._lock:
            with self._path(recipient).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_inbox(self, agent: str) -> list:
        path = self._path(agent)
        with self._lock:
            if not path.is_file():
                return []
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            path.write_text("", encoding="utf-8")
        messages = []
        for line in lines:
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return messages

    @staticmethod
    def render(messages: list) -> str:
        return "\n".join(
            f'<inbox from="{m["from"]}" kind="{m["kind"]}">{m["content"]}</inbox>'
            for m in messages
        )


BUS = MailBus()
MATES: dict[str, dict] = {}
_mates_lock = threading.Lock()


def _teammate_system(name: str, role: str) -> str:
    return (
        f"You are {name}, a teammate agent ({role}) working for the lead. "
        "Read files and run commands to do your work; use send_message only to "
        "talk to other agents. You cannot spawn teammates or schedule work. "
        "When your task is done, stop with a concise final report — it is "
        "delivered to the lead automatically."
    )


def _teammate_execute(name: str, tool: str, args: dict) -> str:
    if tool == "send_message":
        try:
            BUS.send(name, args.get("to", ""), args.get("content", ""))
            return "delivered"
        except ValueError as exc:
            return f"Error: {exc}"
    decision, reason = permissions.evaluate(tool, args)
    if decision is permissions.Decision.DENY:
        return f"Permission denied: {reason}"
    if decision is permissions.Decision.ASK:
        return "Permission denied: teammates cannot ask for interactive approval"
    return toolbox.execute(tool, args)


def teammate_loop(name: str, role: str, prompt: str, client=None) -> None:
    from .agent import _assistant_message, make_client  # late: agent imports team

    client = client or make_client()
    model = os.environ.get("KLAUDE_MODEL", "glm-5.3")
    messages = [
        {"role": "system", "content": _teammate_system(name, role)},
        {"role": "user", "content": f"Role: {role}\n\nTask: {prompt}"},
    ]
    final = "(teammate produced no report)"

    for _ in range(TEAMMATE_MAX_TURNS):
        incoming = BUS.read_inbox(name)
        if incoming:
            messages.append({"role": "user", "content": BUS.render(incoming)})
        response = client.chat.completions.create(
            model=model,
            messages=messages[-20:],
            tools=TEAMMATE_SPECS,
            max_tokens=8192,
        )
        msg = response.choices[0].message
        messages.append(_assistant_message(msg))
        calls = msg.tool_calls or []
        if not calls:
            final = msg.content or final
            break
        import json as _json

        for call in calls:
            try:
                args = _json.loads(call.function.arguments or "{}")
            except _json.JSONDecodeError:
                args = {}
            output = _teammate_execute(name, call.function.name, args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

    BUS.send(name, LEAD, final, "result")
    with _mates_lock:
        if name in MATES:
            MATES[name]["status"] = "done"


def spawn(name: str, role: str, prompt: str, client=None) -> str:
    if not NAME_RE.fullmatch(name or ""):
        return f"Error: invalid teammate name {name!r}"
    with _mates_lock:
        if name in MATES and MATES[name]["status"] == "running":
            return f"Error: teammate {name!r} is already running"
        MATES[name] = {"role": role, "status": "running"}
    thread = threading.Thread(
        target=teammate_loop, args=(name, role, prompt, client), daemon=True, name=f"mate-{name}"
    )
    with _mates_lock:
        MATES[name]["thread"] = thread
    thread.start()
    print(f"\033[95m[team] spawned {name} ({role})\033[0m")
    return f"spawned {name} ({role}); its final report will arrive in your inbox"


def render_mates() -> str:
    with _mates_lock:
        if not MATES:
            return "(no teammates)"
        return "\n".join(
            f"{name} [{info['status']}] {info['role']}" for name, info in sorted(MATES.items())
        )


def inject_lead_inbox(state) -> None:
    """Deliver queued lead mail into the running conversation."""
    incoming = BUS.read_inbox(LEAD)
    if incoming:
        state.messages.append({"role": "user", "content": BUS.render(incoming)})
