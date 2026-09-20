"""Skill library: small manifest files the model loads on demand.

Two levels of disclosure — the system prompt carries only a catalog of
names and one-liners; the full instructions arrive when the model calls
load_skill for one that applies.
"""

import os
import re
from pathlib import Path

from . import file_tools

CATALOG_MAX_CHARS = 4000
CONTENT_MAX_CHARS = 16_000
KNOWN_FIELDS = ("name", "description", "when_to_use")


def _parse_manifest(text: str) -> tuple[dict, str]:
    """Split a '--- key: value ---' header from the body. No YAML dependency."""
    if not text.startswith("---"):
        return {}, text.strip()
    lines = text.splitlines()
    fields: dict[str, str] = {}
    end = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = index
            break
        match = re.match(r"([A-Za-z_][\w-]*):\s*(.*)", line)
        if match and match.group(1).lower() in KNOWN_FIELDS:
            fields[match.group(1).lower()] = match.group(2).strip()
    if end is None:  # never closed — treat the whole thing as body
        return {}, text.strip()
    body = "\n".join(lines[end + 1:]).strip()
    return fields, body


class SkillLibrary:
    def __init__(self, root):
        self.root = Path(root)
        self.skills: dict[str, dict] = {}
        self.scan()

    def scan(self) -> None:
        self.skills.clear()
        if not self.root.is_dir():
            return
        for manifest in sorted(self.root.glob("*/SKILL.md")):
            fields, body = _parse_manifest(
                manifest.read_text(encoding="utf-8", errors="replace")
            )
            name = fields.get("name") or manifest.parent.name
            description = fields.get("description") or (
                body.split("\n", 1)[0].lstrip("# ") if body else ""
            )
            self.skills[name] = {
                "name": name,
                "description": description,
                "when_to_use": fields.get("when_to_use", ""),
                "body": body,
                "path": str(manifest),
            }

    def catalog(self) -> str:
        if not self.skills:
            return "(no skills installed)"
        lines = []
        for skill in self.skills.values():
            line = f"- {skill['name']}: {skill['description']}"
            if skill["when_to_use"]:
                line += f" (use when: {skill['when_to_use']})"
            lines.append(line)
        text = "\n".join(lines)
        if len(text) > CATALOG_MAX_CHARS:
            text = text[:CATALOG_MAX_CHARS] + f"\n... ({len(self.skills)} skills, catalog truncated)"
        return text

    def load(self, name: str) -> str:
        skill = self.skills.get(name)
        if skill is None:
            available = ", ".join(self.skills) or "none"
            return f"Error: unknown skill {name!r}; available: {available}"
        text = skill["body"] or "(empty skill)"
        if len(text) > CONTENT_MAX_CHARS:
            text = text[:CONTENT_MAX_CHARS] + "\n... (truncated)"
        return text


def library() -> SkillLibrary:
    """Fresh scan each call — skills installed mid-session show up too."""
    root = os.environ.get("KLAUDE_SKILLS_DIR") or (file_tools.WORKSPACE / "skills")
    return SkillLibrary(root)
