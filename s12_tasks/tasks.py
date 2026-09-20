"""Persistent, dependency-tracked tasks: one JSON file per task.

A task list that survives the process: create nodes, wire blockedBy
edges (cycles rejected), claim what is ready, complete what you own.
IDs come from a high-water mark and are never reused, even if files
are deleted.
"""

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import file_tools

STATUSES = ("pending", "in_progress", "completed")
ID_RE = re.compile(r"^t(\d+)$")
LOCK_RETRIES = 30
LOCK_SLEEP = 0.005


def store_dir() -> Path:
    root = Path(os.environ.get("KLAUDE_TASKS_DIR") or file_tools.WORKSPACE / ".tasks")
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass
class Task:
    id: int
    subject: str
    description: str = ""
    status: str = "pending"
    owner: str | None = None
    blockedBy: list = field(default_factory=list)

    @property
    def tid(self) -> str:
        return f"t{self.id}"


def _path(tid: str) -> Path:
    if not isinstance(tid, str) or not ID_RE.fullmatch(tid):
        raise ValueError(f"invalid task id: {tid!r}")
    return store_dir() / f"{tid}.json"


def _next_id() -> int:
    marker = store_dir() / ".highwatermark"
    last = 0
    if marker.is_file():
        try:
            last = int(marker.read_text().strip() or 0)
        except ValueError:
            last = 0
    marker.write_text(str(last + 1), encoding="utf-8")
    return last + 1


class _Lock:
    """Best-effort exclusive lock around one task file, cross-process."""

    def __init__(self, tid: str):
        self.lock_path = _path(tid).with_suffix(".lock")

    def __enter__(self):
        for _ in range(LOCK_RETRIES):
            try:
                self.handle = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                time.sleep(LOCK_SLEEP)
        raise TimeoutError(f"task lock busy: {self.lock_path}")

    def __exit__(self, *exc):
        os.close(self.handle)
        try:
            self.lock_path.unlink()
        except OSError:
            pass


def create(subject: str, description: str = "") -> Task:
    subject = subject.strip()
    if not subject:
        raise ValueError("task subject cannot be empty")
    task = Task(id=_next_id(), subject=subject, description=description.strip())
    path = _path(task.tid)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(asdict(task), handle, indent=2)
    return task


def save(task: Task) -> None:
    path = _path(task.tid)
    path.write_text(json.dumps(asdict(task), indent=2), encoding="utf-8")


def load(tid: str) -> Task:
    path = _path(tid)
    if not path.is_file():
        raise ValueError(f"no such task: {tid}")
    return Task(**json.loads(path.read_text(encoding="utf-8")))


def list_all() -> list:
    tasks = []
    for path in sorted(store_dir().glob("t*.json")):
        if ID_RE.fullmatch(path.stem):
            try:
                tasks.append(Task(**json.loads(path.read_text(encoding="utf-8"))))
            except (ValueError, json.JSONDecodeError):
                continue
    return tasks


def depends_on(tid: str, target: str, _seen=None) -> bool:
    """Transitive reachability: does tid (eventually) block on target?"""
    seen = _seen or set()
    if tid == target or tid in seen:
        return tid == target
    seen.add(tid)
    try:
        task = load(tid)
    except ValueError:
        return False
    return any(depends_on(dep, target, seen) for dep in task.blockedBy)


def add_blocked_by(tid: str, deps) -> Task:
    if not isinstance(deps, list) or not deps:
        raise ValueError("add_blocked_by needs a non-empty list of task ids")
    task = load(tid)
    for dep in dict.fromkeys(deps):
        if dep == tid or depends_on(dep, tid):
            raise ValueError(f"adding {dep} would create a dependency cycle")
        if dep not in task.blockedBy:
            try:
                load(dep)
            except ValueError as exc:
                raise ValueError(f"unknown dependency {dep!r}") from exc
            task.blockedBy.append(dep)
    save(task)
    return task


def unmet_deps(task: Task) -> list:
    return [dep for dep in task.blockedBy if load(dep).status != "completed"]


def can_start(tid: str) -> bool:
    return not unmet_deps(load(tid))


def render_list() -> str:
    tasks = list_all()
    if not tasks:
        return "(no tasks)"
    lines = []
    for task in tasks:
        blocked = f" — blocked by {', '.join(task.blockedBy)}" if task.blockedBy else ""
        owner = f" (owner {task.owner})" if task.owner else ""
        lines.append(f"{task.tid} [{task.status}]{owner} {task.subject}{blocked}")
    return "\n".join(lines)


def render_task(tid: str) -> str:
    task = load(tid)
    return (
        f"{task.tid} [{task.status}] {task.subject}\n"
        f"description: {task.description or '(none)'}\n"
        f"owner: {task.owner or '(unassigned)'}\n"
        f"blockedBy: {', '.join(task.blockedBy) or '(none)'}"
    )


def claim(tid: str, owner: str = "agent") -> str:
    with _Lock(tid):
        task = load(tid)  # re-read inside the lock
        if task.status != "pending":
            return f"{tid} is {task.status}, cannot claim"
        blockers = unmet_deps(task)
        if blockers:
            return f"{tid} is blocked by: {', '.join(blockers)}"
        task.owner = owner
        task.status = "in_progress"
        save(task)
    return f"claimed {tid} ({task.subject}) by {owner}"


def complete(tid: str, owner: str = "agent") -> str:
    with _Lock(tid):
        task = load(tid)
        if task.status != "in_progress":
            return f"{tid} is {task.status}, cannot complete"
        if task.owner != owner:
            return f"{tid} is owned by {task.owner}, not {owner}"
        was_blocked = {
            candidate.tid: bool(candidate.blockedBy) and candidate.status == "pending"
            for candidate in list_all()
        }
        task.status = "completed"
        save(task)
    message = f"completed {tid} ({task.subject})"
    unblocked = [
        candidate.subject
        for candidate in list_all()
        if was_blocked.get(candidate.tid) and can_start(candidate.tid)
    ]
    if unblocked:
        message += "\nunblocked: " + ", ".join(unblocked)
    return message
