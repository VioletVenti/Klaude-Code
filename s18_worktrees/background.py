"""Background jobs: run slow commands off the critical path.

The model opts a bash call into the background with run_in_background.
The job runs on a daemon thread; finished results wait in a ready queue
and are injected into the conversation as notifications on a later turn.
"""

import threading

from . import bash_tool

NOTIFICATION_LIMIT = 500


class BackgroundJobs:
    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self._results: dict[str, str] = {}
        self._ready: list[str] = []
        self._counter = 0
        self._lock = threading.Lock()

    def start(self, command: str) -> str:
        command = (command or "").strip()
        if not command:
            raise ValueError("background command cannot be empty")
        with self._lock:
            self._counter += 1
            job_id = f"job{self._counter}"
            self.jobs[job_id] = {"command": command, "status": "running"}

        def run():
            try:
                output = bash_tool.run_bash(command)
                status = (
                    "failed"
                    if output.startswith("Error") or output.startswith("[exit code")
                    else "completed"
                )
            except Exception as exc:  # the job must never kill the loop
                output, status = f"Error: {exc}", "failed"
            with self._lock:
                job = self.jobs.get(job_id)
                if job is None:
                    return
                job["status"] = status
                self._results[job_id] = output
                self._ready.append(job_id)

        threading.Thread(target=run, daemon=True, name=f"bg-{job_id}").start()
        print(f"\033[34m[background] started {job_id}: {command[:60]}\033[0m")
        return job_id

    def pending(self) -> list[str]:
        with self._lock:
            return [job_id for job_id, job in self.jobs.items() if job["status"] == "running"]

    def drain(self) -> list[str]:
        """Collect finished jobs and render their notifications."""
        with self._lock:
            finished = []
            for job_id in self._ready:
                job = self.jobs.pop(job_id, None)
                result = self._results.pop(job_id, "")
                if job is not None:
                    finished.append((job_id, job, result))
            self._ready.clear()
        return [
            (
                f"<background-job id={job_id} status={job['status']}>\n"
                f"command: {job['command']}\n"
                f"output:\n{result[:NOTIFICATION_LIMIT]}\n"
                f"</background-job>"
            )
            for job_id, job, result in finished
        ]


JOBS = BackgroundJobs()
