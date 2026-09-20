"""Tool registry: the specs the model sees and the handlers we run.

Adding a tool means adding one spec + one handler here; the loop never changes.
"""

from . import bash_tool, compact, file_tools
from . import scheduler
from . import skills
from . import tasks
from . import todos as todo_store

READ_SPEC = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a text file's contents, optionally limited to the first N lines.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer", "description": "Only return the first N lines."},
            },
            "required": ["path"],
        },
    },
}

WRITE_SPEC = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Create or overwrite a file with the given content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
}

EDIT_SPEC = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": "Replace an exact, unique text snippet inside a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
}

GLOB_SPEC = {
    "type": "function",
    "function": {
        "name": "glob",
        "description": "Find files under the workspace matching a glob pattern (** is recursive).",
        "parameters": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
}

TODO_SPEC = {
    "type": "function",
    "function": {
        "name": "todo_write",
        "description": "Replace the session task list. Plan multi-step work before executing it.",
        "parameters": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "minLength": 1},
                            "status": {"type": "string", "enum": ["pending", "active", "done"]},
                            "active_form": {"type": "string", "description": "Short 'doing X' shown while active."},
                        },
                        "required": ["content", "status"],
                    },
                }
            },
            "required": ["todos"],
        },
    },
}


def _todo_write(todos) -> str:
    try:
        rendered = todo_store.ACTIVE.replace(todos)
    except ValueError as exc:
        return f"Error: {exc}"
    print(f"\033[33m## tasks\033[0m\n{rendered}")
    return rendered


DELEGATE_SPEC = {
    "type": "function",
    "function": {
        "name": "delegate",
        "description": "Hand a self-contained task to a fresh worker agent; returns its final report.",
        "parameters": {
            "type": "object",
            "properties": {"prompt": {"type": "string", "minLength": 1}},
            "required": ["prompt"],
        },
    },
}


def _delegate(prompt: str) -> str:
    from .delegate import spawn  # imported late: delegate pulls in agent.py

    return spawn(prompt)


LOAD_SKILL_SPEC = {
    "type": "function",
    "function": {
        "name": "load_skill",
        "description": "Load the full instructions of an installed skill.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
}


def _load_skill(name: str) -> str:
    return skills.library().load(name)


COMPACT_SPEC = {
    "type": "function",
    "function": {
        "name": "compact",
        "description": "Request context compaction; takes effect after this batch of tool calls.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _compact() -> str:
    compact.REQUESTED = True
    return "Compaction scheduled after this batch."


TASK_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": "Create a persistent task node; returns its runtime id (tN).",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "minLength": 1},
                    "description": {"type": "string"},
                },
                "required": ["subject"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_task",
            "description": "Add blockedBy dependencies to a task using exact ids.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "add_blocked_by": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                },
                "required": ["task_id", "add_blocked_by"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "List all tasks with status, owner, and blockers.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_task",
            "description": "Show one task's full record.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "claim_task",
            "description": "Claim a ready (pending, unblocked) task as its owner.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "owner": {"type": "string"},
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_task",
            "description": "Complete a task you own; reports anything this unblocked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "owner": {"type": "string"},
                },
                "required": ["task_id"],
            },
        },
    },
]


def _guard(fn):
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as exc:
            return f"Error: {exc}"

    return wrapped


TASK_HANDLERS = {
    "create_task": _guard(lambda subject, description="": tasks.create(subject, description).tid),
    "update_task": _guard(lambda task_id, add_blocked_by: tasks.render_task(tasks.add_blocked_by(task_id, add_blocked_by).tid)),
    "list_tasks": lambda: tasks.render_list(),
    "get_task": _guard(lambda task_id: tasks.render_task(task_id)),
    "claim_task": _guard(lambda task_id, owner="agent": tasks.claim(task_id, owner)),
    "complete_task": _guard(lambda task_id, owner="agent": tasks.complete(task_id, owner)),
}


SCHEDULE_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "schedule_cron",
            "description": "Schedule a prompt to run at a future local time (5-field cron).",
            "parameters": {
                "type": "object",
                "properties": {
                    "cron": {"type": "string", "description": "minute hour day month weekday"},
                    "prompt": {"type": "string"},
                    "recurring": {"type": "boolean"},
                    "durable": {"type": "boolean"},
                },
                "required": ["cron", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_crons",
            "description": "List scheduled cron jobs.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_cron",
            "description": "Cancel a scheduled job by id.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        },
    },
]


def _schedule(cron, prompt, recurring=True, durable=True):
    result = scheduler.SCHEDULER.schedule(cron, prompt, recurring, durable)
    if isinstance(result, str):
        return result
    return f"scheduled {result.jid}: {result.cron}"


SCHEDULE_HANDLERS = {
    "schedule_cron": _schedule,
    "list_crons": lambda: scheduler.SCHEDULER.render(),
    "cancel_cron": lambda job_id: scheduler.SCHEDULER.cancel(job_id),
}


# What a delegated worker may use — deliberately without delegate/todo_write.
BASE_SPECS = [bash_tool.SPEC, READ_SPEC, WRITE_SPEC, EDIT_SPEC, GLOB_SPEC]
BASE_HANDLERS = {
    "run_bash": bash_tool.run_bash,
    "read_file": file_tools.read_file,
    "write_file": file_tools.write_file,
    "edit_file": file_tools.edit_file,
    "glob": file_tools.glob,
}

SPECS = [*BASE_SPECS, TODO_SPEC, DELEGATE_SPEC, LOAD_SKILL_SPEC, COMPACT_SPEC, *TASK_SPECS, *SCHEDULE_SPECS]
HANDLERS = {
    **BASE_HANDLERS,
    "todo_write": _todo_write,
    "delegate": _delegate,
    "load_skill": _load_skill,
    "compact": _compact,
    **TASK_HANDLERS,
    **SCHEDULE_HANDLERS,
}


def execute(name: str, args: dict) -> str:
    """Run one tool by name; failures come back as text, never exceptions."""
    handler = HANDLERS.get(name)
    if handler is None:
        return f"Error: unknown tool {name!r}"
    try:
        return handler(**args)
    except TypeError as exc:
        return f"Error: bad arguments for {name}: {exc}"
