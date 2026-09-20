"""The comprehensive harness entry: every mechanism, one loop.

A single turn runs Intake (user_prompt hooks) -> Guardrails (pre_tool
hooks + permission) -> Route (registry + MCP pool) -> Execute (tools,
slow work to the background) -> External (teammates, cron, MCP servers)
-> Recover (backoff, escalation, reactive compaction) -> Append (post
hooks, notifications, memory at stop). REPL: `python -m s20_full_harness`.
"""

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
    print("klaude — comprehensive harness (s20). type a task, q to quit")
    scheduler.SCHEDULER.load()
    scheduler.SCHEDULER.start_ticking()
    client = make_client()
    state = AgentState()

    while True:
        try:
            prompt = input("klaude >> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() in ("q", "exit"):
            break

        run_agent(prompt, state, client)
        last = state.messages[-1]
        if last["role"] == "assistant":
            print(last["content"], "\n")
        deliver_due(client, state)

    # collect any stragglers on the way out
    deliver_due(client, state)


if __name__ == "__main__":
    main()
