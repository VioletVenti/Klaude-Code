"""s12 — persistent dependency-tracked tasks."""

import json

import pytest

from s12_tasks import tasks, toolbox


@pytest.fixture(autouse=True)
def tasks_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("KLAUDE_TASKS_DIR", str(tmp_path / "tasks"))
    yield


def chain():
    a = tasks.create("write a.txt", "first")
    b = tasks.create("write b.txt")
    c = tasks.create("write c.txt")
    tasks.add_blocked_by(b.tid, [a.tid])
    tasks.add_blocked_by(c.tid, [b.tid])
    return a, b, c


def test_create_assigns_incremental_ids_and_persists():
    a, b, c = chain()
    assert (a.id, b.id, c.id) == (1, 2, 3)
    assert tasks.load("t1").subject == "write a.txt"


def test_highwatermark_prevents_id_reuse(tmp_path):
    tasks.create("gone")
    (tasks.store_dir() / "t1.json").unlink()
    survivor = tasks.create("next")
    assert survivor.id == 2


def test_cycles_and_unknown_deps_rejected():
    a, b, c = chain()
    with pytest.raises(ValueError):
        tasks.add_blocked_by(a.tid, [c.tid])  # would close the loop
    with pytest.raises(ValueError):
        tasks.add_blocked_by(a.tid, ["t99"])


def test_claim_requires_ready_task():
    a, b, c = chain()
    assert "blocked" in tasks.claim(c.tid)
    out = tasks.claim(a.tid)
    assert out.startswith("claimed")
    assert tasks.load(a.tid).status == "in_progress"
    assert tasks.load(a.tid).owner == "agent"
    assert "in_progress" in tasks.claim(a.tid)  # double claim refused


def test_complete_checks_ownership_and_reports_unblocks():
    a, b, c = chain()
    tasks.claim(a.tid)
    assert "owned by" in tasks.complete(a.tid, owner="someone-else")
    out = tasks.complete(a.tid)
    assert out.startswith("completed")
    assert "write b.txt" in out  # newly unblocked
    assert tasks.can_start(b.tid)


def test_render_list_and_detail():
    a, b, c = chain()
    listing = tasks.render_list()
    assert "t1 [pending] write a.txt" in listing
    assert "t2 [pending] write b.txt — blocked by t1" in listing
    detail = tasks.render_task(b.tid)
    assert "blockedBy: t1" in detail


def test_registry_routes_task_tools():
    created = toolbox.execute("create_task", {"subject": "via registry"})
    assert created.startswith("t")
    listing = toolbox.execute("list_tasks", {})
    assert "via registry" in listing
    bad = toolbox.execute("update_task", {"task_id": "t99", "add_blocked_by": ["t1"]})
    assert bad.startswith("Error")


def test_specs_and_handlers_stay_in_sync():
    assert {spec["function"]["name"] for spec in toolbox.SPECS} == set(toolbox.HANDLERS)


def test_invalid_ids_rejected():
    with pytest.raises(ValueError):
        tasks.load("../escape")
    with pytest.raises(ValueError):
        tasks.load("task_12345678")
