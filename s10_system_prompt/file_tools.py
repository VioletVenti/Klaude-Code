"""File tools: read, write, edit, glob — all jailed to the workspace root."""

import os
from pathlib import Path

WORKSPACE = Path(os.environ.get("KLAUDE_WORKDIR", os.getcwd())).resolve()
MAX_GLOB_RESULTS = 200


def _resolve(path: str) -> Path:
    """Turn a user/model-supplied path into one guaranteed inside WORKSPACE."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = WORKSPACE / candidate
    resolved = candidate.resolve()
    if resolved != WORKSPACE and WORKSPACE not in resolved.parents:
        raise ValueError(f"path escapes workspace: {path}")
    return resolved


def read_file(path: str, limit: int | None = None) -> str:
    try:
        lines = _resolve(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except (OSError, ValueError) as exc:
        return f"Error: {exc}"
    if limit is not None and limit < len(lines):
        shown = lines[:limit]
        return "\n".join(shown) + f"\n... ({len(lines) - limit} more lines)"
    return "\n".join(lines)


def write_file(path: str, content: str) -> str:
    try:
        target = _resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except (OSError, ValueError) as exc:
        return f"Error: {exc}"
    return f"Wrote {len(content)} chars to {path}"


def edit_file(path: str, old_text: str, new_text: str) -> str:
    try:
        target = _resolve(path)
        text = target.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError) as exc:
        return f"Error: {exc}"

    count = text.count(old_text)
    if count == 0:
        return f"Error: old_text not found in {path}"
    if count > 1:
        return f"Error: old_text matches {count} times in {path}; add surrounding context"
    target.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
    return f"Edited {path}"


def glob(pattern: str) -> str:
    import glob as globmod

    try:
        matches = sorted(
            {
                match
                for match in globmod.glob(pattern, root_dir=WORKSPACE, recursive=True)
                if (WORKSPACE / match).resolve().is_relative_to(WORKSPACE)
            }
        )
    except (OSError, ValueError) as exc:
        return f"Error: {exc}"
    if not matches:
        return "(no matches)"
    if len(matches) > MAX_GLOB_RESULTS:
        return "\n".join(matches[:MAX_GLOB_RESULTS]) + "\n... (more matches omitted)"
    return "\n".join(matches)
