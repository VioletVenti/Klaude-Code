"""s18 — per-task git worktrees isolate parallel teammates."""

import json
import subprocess
import time as time_mod

import pytest

from s18_worktrees import tasks, team, toolbox, worktrees


@pytest.fixture(autouse=True)
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("KLAUDE_MAILBOX_DIR", str(tmp_path / "mail"))
    monkeypatch.setenv("KLAUDE_TASKS_DIR", str(tmp_path / "tasks"))
    monkeypatch.setenv("KLAUDE_WORKTREES_DIR", str(tmp_path / "wt"))
    team.BUS = team.MailBus()
    team.MATES.clear()
    monkeypatch.setattr(team, "IDLE_POLL_INTERVAL", 0.05)
    monkeypatch.setattr(team, "IDLE_TIMEOUT", 1.0)
    # a real git repo as the workspace
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "seed.txt").write_text("base")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "seed"], cwd=repo, check=True)
    monkeypatch.setattr(worktrees.file_tools, "WORKSPACE", repo)
    yield repo


def read_events():
    path = worktrees.events_path()
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def test_create_makes_path_branch_and_event(repo):
    out = worktrees.create("auth-refactor")
    assert out.startswith("worktree auth-refactor")
    assert (worktrees.path_of("auth-refactor") / "seed.txt").read_text() == "base"
    ok, branches = worktrees.run_git(["branch", "--list", "wt/auth-refactor"])
    assert ok and "wt/auth-refactor" in branches
    assert read_events()[0]["type"] == "create"


def test_name_validation_blocks_traversal():
    assert worktrees.validate("..")
    assert worktrees.validate(".")
    assert worktrees.validate("no/slashes")
    assert worktrees.create("../escape").startswith("Error")


def test_bind_via_create_task_id(repo):
    task = tasks.create("isolated work")
    worktrees.create("iso", task_id=task.tid)
    assert tasks.load(task.tid).worktree == "iso"
    assert tasks.load(task.tid).status == "pending"  # binding never advances status


def test_remove_refuses_dirty_worktree(repo):
    worktrees.create("dirty")
    (worktrees.path_of("dirty") / "new.txt").write_text("uncommitted")
    out = worktrees.remove("dirty")
    assert out.startswith("Error") and "uncommitted" in out
    assert "discard_changes=true" in worktrees.remove("dirty", discard_changes=True) or worktrees.path_of("dirty").exists() is False


def test_keep_leaves_branch_for_review(repo):
    worktrees.create("review-me")
    out = worktrees.keep("review-me")
    assert "kept" in out
    assert worktrees.path_of("review-me").is_dir()  # untouched
    assert read_events()[-1]["type"] == "keep"


def test_teammate_works_inside_its_worktree(repo):
    from test_s01_agent_loop import FakeClient, FakeMessage, FakeToolCall

    task = tasks.create("edit in isolation", "write isolated.txt")
    worktrees.create("wt-demo", task_id=task.tid)
    script = [
        FakeMessage(content="standing by"),  # idle -> self-claim
        FakeMessage(
            tool_calls=[
                FakeToolCall("c1", "write_file", json.dumps({"path": "isolated.txt", "content": "from worktree"})),
                FakeToolCall("c2", "complete_task", json.dumps({"task_id": task.tid})),
            ]
        ),
        FakeMessage(content="done"),
    ]
    client = FakeClient(script)
    team.spawn("iso-worker", "worker", "watch the board", client=client)
    deadline = time_mod.monotonic() + 10
    while team.MATES["iso-worker"]["status"] != "done" and time_mod.monotonic() < deadline:
        time_mod.sleep(0.05)
    assert team.MATES["iso-worker"]["status"] == "done"
    # the file landed inside the worktree, never in the shared root
    assert (worktrees.path_of("wt-demo") / "isolated.txt").read_text() == "from worktree"
    assert not (repo / "isolated.txt").exists()


def test_registry_has_worktree_tools():
    names = {spec["function"]["name"] for spec in toolbox.SPECS}
    assert {"create_worktree", "remove_worktree", "keep_worktree"} <= names
    assert names == set(toolbox.HANDLERS)
