"""Tool registry: the specs the model sees and the handlers we run.

Adding a tool means adding one spec + one handler here; the loop never changes.
"""

from . import bash_tool, file_tools
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


SPECS = [bash_tool.SPEC, READ_SPEC, WRITE_SPEC, EDIT_SPEC, GLOB_SPEC, TODO_SPEC]

HANDLERS = {
    "run_bash": bash_tool.run_bash,
    "read_file": file_tools.read_file,
    "write_file": file_tools.write_file,
    "edit_file": file_tools.edit_file,
    "glob": file_tools.glob,
    "todo_write": _todo_write,
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
