"""Permission gates: hard-deny outright, or ask a human before risky tools.

Policy lives here, not inside the tools — a tool just executes; the
pipeline decides whether it is allowed to.
"""

import re
from enum import Enum
from pathlib import Path

from . import file_tools


class Decision(Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


# Never allowed, no questions asked.
DENY_PATTERNS = ("rm -rf /", "mkfs", "dd of=/dev/", "shutdown", "reboot", "sudo")

# Allowed only after a human says yes.
DESTRUCTIVE_COMMAND = re.compile(r"(?:^|[;&|]\s*)\b(?:rm|rmdir|del|chmod|chown|mv)\b")
RISKY_REDIRECT = re.compile(r">\s*/etc/|>\s*/boot/|chmod\s+777")

EDITING_TOOLS = ("write_file", "edit_file")


def _escapes_workspace(args: dict) -> bool:
    path = Path(args.get("path", ""))
    if not path.is_absolute():
        path = file_tools.WORKSPACE / path
    try:
        resolved = path.resolve()
    except OSError:
        return True
    return resolved != file_tools.WORKSPACE and file_tools.WORKSPACE not in resolved.parents


def evaluate(tool: str, args: dict) -> tuple[Decision, str]:
    """Pure decision: which gate does this call hit, and why."""
    if tool == "run_bash":
        command = args.get("command", "")
        lowered = command.lower()
        for pattern in DENY_PATTERNS:
            if pattern in lowered:
                return Decision.DENY, f"'{pattern}' is on the hard deny list"
        if DESTRUCTIVE_COMMAND.search(command) or RISKY_REDIRECT.search(command):
            return Decision.ASK, "potentially destructive shell command"

    if tool in EDITING_TOOLS and _escapes_workspace(args):
        return Decision.ASK, "target path outside the workspace"

    return Decision.ALLOW, ""


def _interactive_approver(tool: str, args: dict, reason: str) -> bool:
    short = {k: (v[:60] + "…" if isinstance(v, str) and len(v) > 60 else v) for k, v in args.items()}
    print(f"\n\033[33m[permission] {reason}\033[0m")
    print(f"   {tool}({short})")
    choice = input("   allow? [y/N] ").strip().lower()
    return choice in ("y", "yes")


# Swap this in tests or to auto-approve in scripted runs.
approver = _interactive_approver


def check(tool: str, args: dict) -> tuple[bool, str]:
    """Gate a tool call: returns (allowed, denial_reason)."""
    decision, reason = evaluate(tool, args)
    if decision is Decision.DENY:
        print(f"\033[31m[denied] {reason}\033[0m")
        return False, reason
    if decision is Decision.ASK and not approver(tool, args, reason):
        return False, reason
    return True, ""
