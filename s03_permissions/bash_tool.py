"""The bash tool: run one shell command, return captured output.

Safety policy moved out — s03 decides what runs in permissions.py.
"""

import subprocess

COMMAND_TIMEOUT = 60
MAX_OUTPUT_CHARS = 10_000

SPEC = {
    "type": "function",
    "function": {
        "name": "run_bash",
        "description": "Run a shell command and return its combined stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute."}
            },
            "required": ["command"],
        },
    },
}


def run_bash(command: str) -> str:
    """Execute `command` in a shell; never raises — errors come back as text."""
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=COMMAND_TIMEOUT,
        )
        output = (proc.stdout + proc.stderr).strip()
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {COMMAND_TIMEOUT}s"
    except OSError as exc:
        return f"Error: {exc}"

    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + f"\n... (truncated at {MAX_OUTPUT_CHARS} chars)"
    return output if output else "(no output)"
