"""The agent loop: keep calling the model and running tools until it stops.

One model, one tool, one loop. The model decides when the work is done by
simply not asking for another tool call; our job is only to feed results
back and know when to give up.
"""

import json
import os
import re
from dataclasses import dataclass, field

from openai import OpenAI

from . import toolbox

DEFAULT_BASE_URL = "https://hub.oaifree.com/v1"
DEFAULT_MODEL = "glm-5.3"
MAX_TURNS = 24
MAX_OUTPUT_TOKENS = 8192

THINK_TAGS = re.compile(r"<think>.*?</think>", re.DOTALL)


def system_prompt(cwd: str) -> str:
    return (
        f"You are Klaude, a terminal coding agent working in {cwd}. "
        "Solve the user's task by running shell commands via the run_bash tool. "
        "Act, verify, then report the result briefly."
    )


@dataclass
class AgentState:
    """Everything the loop knows about the conversation it is running."""

    messages: list = field(default_factory=list)
    turn: int = 0
    max_turns: int = MAX_TURNS
    last_stop: str = "start"  # why the previous turn ended: "tool_calls" / "end" / "max_turns"

    @property
    def finished(self) -> bool:
        return self.last_stop in ("end", "max_turns")


def make_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("KLAUDE_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("KLAUDE_BASE_URL", DEFAULT_BASE_URL),
    )


def _clean_text(text: str | None) -> str:
    """Strip reasoning blocks some thinking models leave inside plain content."""
    if not text:
        return ""
    return THINK_TAGS.sub("", text).strip()


def _assistant_message(msg) -> dict:
    """Rebuild the assistant turn as a plain dict, dropping anything non-standard."""
    message = {"role": "assistant", "content": _clean_text(msg.content)}
    if msg.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in msg.tool_calls
        ]
    return message


def run_agent(prompt: str, state: AgentState | None = None, client: OpenAI | None = None) -> AgentState:
    """Run the loop for one user prompt until the model stops asking for tools."""
    client = client or make_client()
    model = os.environ.get("KLAUDE_MODEL", DEFAULT_MODEL)

    if state is None:
        state = AgentState()
    if not state.messages:
        state.messages.append({"role": "system", "content": system_prompt(os.getcwd())})
    state.messages.append({"role": "user", "content": prompt})

    while state.turn < state.max_turns:
        state.turn += 1

        response = client.chat.completions.create(
            model=model,
            messages=state.messages,
            tools=toolbox.SPECS,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        msg = response.choices[0].message
        state.messages.append(_assistant_message(msg))

        # Decide whether to loop again from what the model actually said,
        # not from finish_reason — streaming endpoints update it late.
        tool_calls = msg.tool_calls or []
        if not tool_calls:
            state.last_stop = "end"
            return state
        state.last_stop = "tool_calls"

        for call in tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            output = toolbox.execute(call.function.name, args)
            print(f"\033[33m> {call.function.name}\033[0m")
            state.messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": output}
            )

    state.last_stop = "max_turns"
    return state
