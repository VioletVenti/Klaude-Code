"""Interactive entry: `python -m s16_protocols`. Reads prompts until q/EOF."""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from . import permissions, scheduler
from .agent import AgentState, make_client, run_agent


def deliver_due(client, state):
    """Run queued scheduled prompts; scheduled turns never ask interactively."""
    due = scheduler.SCHEDULER.take()
    if not due:
        return
    original = permissions.approver
    permissions.approver = lambda *a: False
    try:
        for job in due:
            print(f"\033[36m[scheduled {job.jid}] {job.prompt}\033[0m")
            run_agent(job.prompt, state, client)
    finally:
        permissions.approver = original
        scheduler.SCHEDULER.acknowledge(due)


def main() -> None:
    print("s16 protocols — type a task, q to quit")
    scheduler.SCHEDULER.load()
    scheduler.SCHEDULER.start_ticking()
    client = make_client()
    state = AgentState()

    while True:
        try:
            prompt = input("s16 >> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() in ("q", "exit"):
            break

        run_agent(prompt, state, client)
        last = state.messages[-1]
        if last["role"] == "assistant":
            print(last["content"], "\n")
        deliver_due(client, state)


if __name__ == "__main__":
    main()
