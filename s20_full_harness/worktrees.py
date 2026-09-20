"""One git worktree per task: isolated directories for parallel work.

Teammates sharing one directory overwrite each other; a worktree gives
each claimed task its own checkout on its own branch. Creation binds a
task by name; removal refuses to lose uncommitted work; keeping leaves
the branch for human review. Every transition lands in an audit log.
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path

from . import bash_tool, file_tools

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
FORBIDDEN_NAMES = {".", ".."}


def root() -> Path:
    base = Path(os.environ.get("KLAUDE_WORKTREES_DIR") or file_tools.WORKSPACE / ".worktrees")
    return base


def events_path() -> Path:
    return root() / "events.jsonl"


def validate(name: str) -> str | None:
    if name in FORBIDDEN_NAMES or not NAME_RE.fullmatch(name or ""):
        return f"invalid worktree name: {name!r}"
    return None


def run_git(args: list, cwd=None) -> tuple[bool, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd or file_tools.WORKSPACE),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def log_event(kind: str, name: str, task_id: str = "") -> None:
    root().mkdir(parents=True, exist_ok=True)
    record = {"type": kind, "worktree": name, "task_id": task_id, "ts": time.time()}
    with events_path().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def path_of(name: str) -> Path:
    return root() / name


def create(name: str, task_id: str = "") -> str:
    error = validate(name)
    if error:
        return f"Error: {error}"
    path = path_of(name)
    if path.exists():
        return f"Error: {name!r} already exists"
    ok, out = run_git(["worktree", "add", str(path), "-b", f"wt/{name}", "HEAD"])
    if not ok:
        return f"Error: git worktree add failed: {out}"
    if task_id:
        bind(task_id, name)
    log_event("create", name, task_id)
    return f"worktree {name} at {path} (branch wt/{name})"


def bind(task_id: str, name: str) -> None:
    from . import tasks

    task = tasks.load(task_id)
    task.worktree = name
    tasks.save(task)


def pending_changes(name: str) -> int:
    """Uncommitted files plus commits ahead of the base branch."""
    path = path_of(name)
    if not path.is_dir():
        return 0
    ok, out = run_git(["status", "--porcelain"], cwd=path)
    dirty = len(out.splitlines()) if ok else 0
    ok, out = run_git(["rev-list", "--count", f"HEAD..wt/{name}"])
    ahead = int(out or 0) if ok else 0
    return dirty + ahead


def remove(name: str, discard_changes: bool = False) -> str:
    error = validate(name)
    if error:
        return f"Error: {error}"
    changes = pending_changes(name)
    if changes and not discard_changes:
        return (
            f"Error: {name!r} has {changes} uncommitted change(s); "
            "pass discard_changes=true or keep it for review"
        )
    ok, out = run_git(["worktree", "remove", "--force", str(path_of(name))])
    if not ok:
        return f"Error: git worktree remove failed: {out}"
    run_git(["branch", "-D", f"wt/{name}"])
    log_event("remove", name)
    return f"removed worktree {name}"


def keep(name: str) -> str:
    error = validate(name)
    if error:
        return f"Error: {error}"
    log_event("keep", name)
    return f"kept branch wt/{name} for review"


# -- worktree-scoped file and command execution for teammates -----------------

def _safe_join(base: Path, path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError(f"path escapes worktree: {path}")
    return resolved


def run_in(command: str, base: Path) -> str:
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(base),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=bash_tool.COMMAND_TIMEOUT,
        )
        output = (proc.stdout + proc.stderr).strip()
        if proc.returncode != 0:
            output = f"[exit code {proc.returncode}] {output}".strip()
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {bash_tool.COMMAND_TIMEOUT}s"
    except OSError as exc:
        return f"Error: {exc}"
    if len(output) > bash_tool.MAX_OUTPUT_CHARS:
        output = output[: bash_tool.MAX_OUTPUT_CHARS] + "\n... (truncated)"
    return output if output else "(no output)"


def read_in(path: str, base: Path) -> str:
    try:
        lines = _safe_join(base, path).read_text(encoding="utf-8", errors="replace").splitlines()
    except (OSError, ValueError) as exc:
        return f"Error: {exc}"
    return "\n".join(lines)


def write_in(path: str, content: str, base: Path) -> str:
    try:
        target = _safe_join(base, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except (OSError, ValueError) as exc:
        return f"Error: {exc}"
    return f"Wrote {len(content)} chars to {path} (in {base.name})"
