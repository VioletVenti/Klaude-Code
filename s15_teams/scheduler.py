"""A cron scheduler: local-time jobs that fire while the agent lives.

Five-field cron expressions (minute hour day month weekday) with steps,
ranges, lists, and the usual day/weekday OR rule. Durable jobs persist
atomically and reload with their undelivered firings intact; a job fires
at most once per minute; recurring jobs re-arm, one-shots disappear
after delivery.
"""

import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from . import file_tools

TICK_SECONDS = 10
FIELD_RANGES = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 6))


def store_path() -> Path:
    return Path(os.environ.get("KLAUDE_SCHEDULE_FILE") or file_tools.WORKSPACE / ".schedule.json")


@dataclass
class Job:
    jid: str
    cron: str
    prompt: str
    recurring: bool = True
    durable: bool = True
    pending: bool = False
    last_fired: str | None = None


class Scheduler:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.queue: list[Job] = []
        self._counter = 0
        self._lock = threading.RLock()
        self._tick_thread: threading.Thread | None = None

    # -- cron parsing -------------------------------------------------------

    @staticmethod
    def _field_matches(field: str, value: int) -> bool:
        for part in field.split(","):
            part = part.strip()
            if part == "*":
                return True
            if part.startswith("*/"):
                if value % int(part[2:]) == 0:
                    return True
            elif "-" in part:
                low, high = part.split("-", 1)
                if int(low) <= value <= int(high):
                    return True
            elif value == int(part):
                return True
        return False

    @classmethod
    def matches(cls, cron: str, moment: datetime) -> bool:
        fields = cron.split()
        if len(fields) != 5:
            return False
        minute, hour, day, month, weekday = fields
        if not (
            cls._field_matches(minute, moment.minute)
            and cls._field_matches(hour, moment.hour)
            and cls._field_matches(month, moment.month)
        ):
            return False
        cron_weekday = (moment.weekday() + 1) % 7  # Sunday = 0
        if day == "*" and weekday == "*":
            return True
        if day == "*":
            return cls._field_matches(weekday, cron_weekday)
        if weekday == "*":
            return cls._field_matches(day, moment.day)
        return cls._field_matches(day, moment.day) or cls._field_matches(weekday, cron_weekday)

    @staticmethod
    def validate(cron: str) -> str | None:
        fields = cron.split()
        if len(fields) != 5:
            return f"expected 5 fields, got {len(fields)}"
        names = ("minute", "hour", "day-of-month", "month", "day-of-week")
        for field, (name, (low, high)) in zip(fields, zip(names, FIELD_RANGES)):
            for part in field.split(","):
                part = part.strip()
                if part == "*":
                    continue
                if part.startswith("*/"):
                    step = part[2:]
                    if not step.isdigit() or int(step) <= 0:
                        return f"{name}: bad step {part!r}"
                    continue
                if "-" in part:
                    a, b = part.split("-", 1)
                    if not (a.isdigit() and b.isdigit()):
                        return f"{name}: bad range {part!r}"
                    values = (int(a), int(b))
                elif part.isdigit():
                    values = (int(part), int(part))
                else:
                    return f"{name}: bad field {part!r}"
                if values[0] > values[1] or not (low <= values[0] and values[1] <= high):
                    return f"{name}: {part!r} outside [{low}-{high}]"
        return None

    # -- persistence ---------------------------------------------------------

    def _save(self) -> None:
        payload = [asdict(job) for job in self.jobs.values() if job.durable]
        target = store_path()
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, target)

    def load(self) -> int:
        path = store_path()
        if not path.is_file():
            return 0
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError("expected a list")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[scheduler] could not load {path.name}: {exc}")
            return 0
        loaded = 0
        with self._lock:
            for item in payload:
                try:
                    job = Job(**item)
                    if self.validate(job.cron) or not job.prompt.strip() or not job.jid.startswith("c"):
                        raise ValueError("invalid record")
                except (TypeError, ValueError):
                    print("[scheduler] skipped an invalid saved job")
                    continue
                self.jobs[job.jid] = job
                self._counter = max(self._counter, int(job.jid[1:]))
                if job.pending:
                    self.queue.append(job)
                loaded += 1
        if loaded:
            print(f"[scheduler] restored {loaded} job(s)")
        return loaded

    # -- job management ------------------------------------------------------

    def schedule(self, cron: str, prompt: str, recurring: bool = True, durable: bool = True):
        error = self.validate(cron)
        if error:
            return f"Error: {error}"
        if not prompt.strip():
            return "Error: prompt cannot be empty"
        with self._lock:
            self._counter += 1
            job = Job(jid=f"c{self._counter}", cron=cron, prompt=prompt.strip(),
                      recurring=recurring, durable=durable)
            self.jobs[job.jid] = job
            try:
                if durable:
                    self._save()
            except OSError:
                self.jobs.pop(job.jid)
                return "Error: could not persist the schedule"
        print(f"[scheduler] {job.jid}: {cron} -> {job.prompt[:60]}")
        return job

    def cancel(self, jid: str) -> str:
        with self._lock:
            if jid not in self.jobs:
                return f"Error: no such job {jid}"
            self.jobs.pop(jid)
            self.queue[:] = [job for job in self.queue if job.jid != jid]
            try:
                self._save()
            except OSError:
                pass
        return f"cancelled {jid}"

    def render(self) -> str:
        with self._lock:
            if not self.jobs:
                return "(no scheduled jobs)"
            lines = []
            for job in self.jobs.values():
                kind = "recurring" if job.recurring else "once"
                state = " firing" if job.pending else ""
                lines.append(f"{job.jid} [{kind}] {job.cron}{state} -> {job.prompt[:80]}")
            return "\n".join(lines)

    # -- firing ----------------------------------------------------------------

    def poll(self, moment: datetime) -> None:
        marker = moment.strftime("%Y-%m-%d %H:%M")
        with self._lock:
            for job in list(self.jobs.values()):
                if job.pending or job.last_fired == marker:
                    continue
                if self.matches(job.cron, moment):
                    job.pending = True
                    job.last_fired = marker
                    self.queue.append(job)
                    try:
                        if job.durable:
                            self._save()
                    except OSError:
                        pass
                    print(f"[scheduler] due {job.jid}: {job.prompt[:60]}")

    def take(self) -> list:
        with self._lock:
            delivered, self.queue = list(self.queue), []
        return delivered

    def acknowledge(self, delivered: list) -> None:
        with self._lock:
            for job in delivered:
                current = self.jobs.get(job.jid)
                if current is None:
                    continue
                if current.recurring:
                    current.pending = False
                else:
                    self.jobs.pop(current.jid)
            try:
                self._save()
            except OSError:
                pass

    # -- background ticking ------------------------------------------------------

    def start_ticking(self) -> None:
        if self._tick_thread:
            return

        def loop():
            import time

            while True:
                self.poll(datetime.now())
                time.sleep(TICK_SECONDS)

        self._tick_thread = threading.Thread(target=loop, daemon=True, name="scheduler-tick")
        self._tick_thread.start()


SCHEDULER = Scheduler()
