"""Delegate work to a fresh, isolated agent conversation.

The worker gets an empty history, its own system prompt, and the base
toolset — no delegate (workers don't spawn workers) and no todo_write
(the plan belongs to the parent session). Only the final report comes
back. Tool calls inside the worker still pass through the same hooks,
so permissions gate it exactly like the parent.
"""

from .agent import AgentState, make_client, run_agent
from . import toolbox

SUB_MAX_TURNS = 15
SUB_SYSTEM = (
    "You are a delegated worker for a parent coding agent. "
    "Complete the given task on your own, verify your work, and finish "
    "with a concise report — the parent only ever sees that report."
)

DEPTH = 0  # recursion guard


def spawn(prompt: str, client=None) -> str:
    global DEPTH
    if DEPTH >= 1:
        return "Error: delegated workers cannot delegate further"

    client = client or make_client()
    print("\n\033[35m┌─ worker spawned\033[0m")
    DEPTH += 1
    try:
        state = AgentState(
            messages=[{"role": "system", "content": SUB_SYSTEM}],
            max_turns=SUB_MAX_TURNS,
        )
        run_agent(prompt, state, client, tools=toolbox.BASE_SPECS)
    finally:
        DEPTH -= 1
        print("\033[35m└─ worker finished\033[0m")

    if state.last_stop == "max_turns":
        return "Worker gave up after too many turns without a final report."
    for message in reversed(state.messages):
        if message["role"] == "assistant" and message.get("content"):
            return message["content"]
    return "(worker produced no report)"
