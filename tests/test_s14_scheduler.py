"""s14 — cron scheduling, persistence, and delivery."""

import json
from datetime import datetime

import pytest

from s14_scheduler import scheduler, toolbox


@pytest.fixture(autouse=True)
def fresh_store(tmp_path, monkeypatch):
    monkeypatch.setenv("KLAUDE_SCHEDULE_FILE", str(tmp_path / "sched.json"))
    scheduler.SCHEDULER = scheduler.Scheduler()
    yield


def at(text):
    return datetime.strptime(text, "%Y-%m-%d %H:%M")


def test_matcher_covers_steps_ranges_lists_and_weekday_or():
    S = scheduler.Scheduler
    assert S.matches("30 2 * * *", at("2026-09-20 02:30"))
    assert not S.matches("30 2 * * *", at("2026-09-20 02:31"))
    assert S.matches("*/15 * * * *", at("2026-09-20 02:45"))
    assert not S.matches("*/15 * * * *", at("2026-09-20 02:50"))
    assert S.matches("0 9-17 * * *", at("2026-09-20 11:00"))
    assert S.matches("0 0 1,15 * *", at("2026-09-15 00:00"))
    # Sunday = 0 in cron; 2026-09-20 is a Sunday
    assert S.matches("0 12 * * 0", at("2026-09-20 12:00"))
    # day/weekday OR rule: day restricted fails, weekday matches -> fires
    assert S.matches("0 12 1 * 0", at("2026-09-20 12:00"))


def test_validation_rejects_garbage():
    S = scheduler.Scheduler
    assert S.validate("60 * * * *")
    assert S.validate("* 24 * * *")
    assert S.validate("* * 5-2 * *")
    assert S.validate("*/0 * * * *")
    assert S.validate("abc * * * *")
    assert S.validate("* * * *")
    assert S.validate("* * * * *") is None
    assert S.validate("*/15 8-18 1,15 */2 1-5") is None


def test_schedule_persists_durable_and_reloads_pending():
    result = scheduler.SCHEDULER.schedule("* * * * *", "tick", durable=True)
    assert result.jid == "c1"
    data = json.loads(scheduler.store_path().read_text())
    assert data[0]["prompt"] == "tick"

    scheduler.SCHEDULER.poll(at("2026-09-20 12:00"))
    assert scheduler.SCHEDULER.queue[0].jid == "c1"

    revived = scheduler.Scheduler()
    assert revived.load() == 1
    assert revived.queue[0].jid == "c1"  # pending firing restored


def test_once_per_minute_dedupe_and_recurring_rearm():
    scheduler.SCHEDULER.schedule("* * * * *", "tick", recurring=True)
    moment = at("2026-09-20 12:00")
    scheduler.SCHEDULER.poll(moment)
    scheduler.SCHEDULER.poll(moment)  # same minute: no double fire
    assert len(scheduler.SCHEDULER.queue) == 1
    delivered = scheduler.SCHEDULER.take()
    scheduler.SCHEDULER.acknowledge(delivered)
    scheduler.SCHEDULER.poll(at("2026-09-20 12:01"))
    assert len(scheduler.SCHEDULER.queue) == 1  # re-armed for the next minute


def test_one_shot_disappears_after_delivery():
    scheduler.SCHEDULER.schedule("5 * * * *", "once only", recurring=False)
    scheduler.SCHEDULER.poll(at("2026-09-20 12:05"))
    delivered = scheduler.SCHEDULER.take()
    scheduler.SCHEDULER.acknowledge(delivered)
    assert scheduler.SCHEDULER.jobs == {}
    scheduler.SCHEDULER.poll(at("2026-09-20 13:05"))
    assert scheduler.SCHEDULER.queue == []


def test_cancel_removes_job_and_queue_entry():
    job = scheduler.SCHEDULER.schedule("0 0 * * *", "midnight")
    scheduler.SCHEDULER.poll(at("2026-09-20 00:00"))
    assert "cancelled" in scheduler.SCHEDULER.cancel(job.jid)
    assert scheduler.SCHEDULER.queue == []
    assert "Error" in scheduler.SCHEDULER.cancel(job.jid)


def test_registry_routes_schedule_tools():
    out = toolbox.execute("schedule_cron", {"cron": "bad expr", "prompt": "x"})
    assert out.startswith("Error")
    toolbox.execute("schedule_cron", {"cron": "0 0 * * *", "prompt": "midnight job"})
    listing = toolbox.execute("list_crons", {})
    assert "midnight job" in listing
    jid = scheduler.SCHEDULER.jobs["c1"].jid
    assert "cancelled" in toolbox.execute("cancel_cron", {"job_id": jid})


def test_specs_and_handlers_stay_in_sync():
    assert {spec["function"]["name"] for spec in toolbox.SPECS} == set(toolbox.HANDLERS)
