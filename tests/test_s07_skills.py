"""s07 — two-level skill disclosure: catalog in prompt, body on demand."""

import pytest

from s07_skills import skills, toolbox
from s07_skills.agent import system_prompt


def test_frontmatter_parsing_full():
    fields, body = skills._parse_manifest(
        "---\nname: demo\ndescription: a demo\nwhen_to_use: testing\n---\n\n# Body\nstep one"
    )
    assert fields == {"name": "demo", "description": "a demo", "when_to_use": "testing"}
    assert body == "# Body\nstep one"


def test_frontmatter_missing_falls_back(tmp_path):
    lib = skills.SkillLibrary(tmp_path)
    assert lib.skills == {}
    (tmp_path / "bare").mkdir()
    (tmp_path / "bare" / "SKILL.md").write_text("# First line is the description\ninstructions")
    lib.scan()
    assert "bare" in lib.skills
    assert lib.skills["bare"]["description"].startswith("First line")


def test_unclosed_frontmatter_is_body():
    fields, body = skills._parse_manifest("---\nname: never closes\njust text")
    assert fields == {}
    assert "never closes" in body


def test_catalog_lists_skills_with_use_when(tmp_path):
    for name in ("beta", "alpha"):
        d = tmp_path / name
        d.mkdir()
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: skill {name}\nwhen_to_use: doing {name}\n---\nbody"
        )
    lib = skills.SkillLibrary(tmp_path)
    catalog = lib.catalog()
    lines = catalog.splitlines()
    assert lines[0].startswith("- alpha: skill alpha (use when: doing alpha)")
    assert lines[1].startswith("- beta: skill beta")


def test_catalog_budget_is_capped(tmp_path):
    for i in range(30):
        d = tmp_path / f"s{i:02d}"
        d.mkdir()
        (d / "SKILL.md").write_text(
            f"---\nname: s{i:02d}\ndescription: {'x' * 300}\n---\nbody"
        )
    catalog = skills.SkillLibrary(tmp_path).catalog()
    assert len(catalog) < 4500
    assert "truncated" in catalog


def test_load_unknown_lists_available(tmp_path):
    lib = skills.SkillLibrary(tmp_path)
    (tmp_path / "only").mkdir()
    (tmp_path / "only" / "SKILL.md").write_text("---\nname: only\n---\nthe body")
    lib.scan()
    out = lib.load("nope")
    assert out.startswith("Error")
    assert "only" in out


def test_registry_routes_load_skill(tmp_path, monkeypatch):
    monkeypatch.setenv("KLAUDE_SKILLS_DIR", str(tmp_path))
    (tmp_path / "greeter").mkdir()
    (tmp_path / "greeter" / "SKILL.md").write_text("---\nname: greeter\n---\nsay hello warmly")
    out = toolbox.execute("load_skill", {"name": "greeter"})
    assert out == "say hello warmly"


def test_system_prompt_carries_the_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("KLAUDE_SKILLS_DIR", str(tmp_path))
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "SKILL.md").write_text(
        "---\nname: notes\ndescription: writes notes\n---\nbody"
    )
    prompt = system_prompt("/somewhere")
    assert "Skills installed:" in prompt
    assert "- notes: writes notes" in prompt


def test_specs_and_handlers_stay_in_sync():
    spec_names = {spec["function"]["name"] for spec in toolbox.SPECS}
    assert spec_names == set(toolbox.HANDLERS)
