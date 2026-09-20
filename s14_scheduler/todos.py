"""A structured task list the model plans with and keeps updated."""

import json
from dataclasses import dataclass, field

STATUSES = ("pending", "active", "done")
MAX_ITEMS = 20


@dataclass
class Todo:
    content: str
    status: str = "pending"
    active_form: str = ""  # one-liner shown while this item is running


@dataclass
class TodoList:
    items: list = field(default_factory=list)

    def replace(self, todos) -> str:
        """Validate and swap in a whole new list; returns its rendering."""
        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except json.JSONDecodeError as exc:
                raise ValueError("todos must be a list or a JSON array string") from exc
        if not isinstance(todos, list):
            raise ValueError("todos must be a list")
        if len(todos) > MAX_ITEMS:
            raise ValueError(f"at most {MAX_ITEMS} items allowed")

        parsed = []
        active_count = 0
        for index, raw in enumerate(todos):
            if not isinstance(raw, dict):
                raise ValueError(f"item {index} must be an object")
            content = str(raw.get("content", "")).strip()
            if not content:
                raise ValueError(f"item {index} needs content")
            status = str(raw.get("status", "pending")).lower()
            if status not in STATUSES:
                raise ValueError(f"item {index} has invalid status {status!r}")
            if status == "active":
                active_count += 1
            parsed.append(Todo(content, status, str(raw.get("active_form", ""))))

        if active_count > 1:
            raise ValueError("only one item may be active at a time")

        # Nudge: everything marked done but nothing ever verifies the work.
        if (
            len(parsed) >= 3
            and all(t.status == "done" for t in parsed)
            and not any("verif" in t.content.lower() for t in parsed)
        ):
            parsed.append(Todo("Verify the completed work", "pending", "verifying"))

        self.items = parsed
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "(no tasks)"
        lines = []
        for index, todo in enumerate(self.items, 1):
            mark = {"pending": " ", "active": "~", "done": "x"}[todo.status]
            suffix = f" — {todo.active_form}" if todo.status == "active" and todo.active_form else ""
            lines.append(f"{index:>2}. [{mark}] {todo.content}{suffix}")
        done = sum(t.status == "done" for t in self.items)
        lines.append(f"({done}/{len(self.items)} done)")
        return "\n".join(lines)


ACTIVE = TodoList()  # one list per process; a REPL session is one session
