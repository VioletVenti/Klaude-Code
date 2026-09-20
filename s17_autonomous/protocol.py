"""Typed request/response protocols over the mailbox bus.

One mechanism, two protocols: graceful shutdown (lead asks a teammate to
stop) and plan approval (a teammate asks the lead to review its plan).
Every request carries a request id; a response is matched by id and
type, and only while still pending — late or mismatched replies are
dropped.
"""

import time
from dataclasses import dataclass, field

LEAD = "lead"


def _bus():
    from .team import BUS  # late: team imports toolbox, toolbox imports us

    return BUS

KINDS = ("shutdown", "plan")


@dataclass
class ProtocolState:
    rid: str
    kind: str
    sender: str
    target: str
    status: str = "pending"  # pending -> approved | rejected
    payload: str = ""
    created_at: float = field(default_factory=time.time)


REQUESTS: dict[str, ProtocolState] = {}
_counter = 0


def new_rid() -> str:
    global _counter
    _counter += 1
    return f"req{_counter:06d}"


def request(sender: str, target: str, kind: str, payload: str = "") -> ProtocolState:
    if kind not in KINDS:
        raise ValueError(f"unknown protocol kind: {kind!r}")
    state = ProtocolState(rid=new_rid(), kind=kind, sender=sender, target=target, payload=payload)
    REQUESTS[state.rid] = state
    _bus().send(sender, target, payload, kind=f"{kind}_request", rid=state.rid)
    return state


def match_response(response_kind: str, rid: str, approve: bool) -> ProtocolState | None:
    """Apply a response if it truly answers a still-pending request."""
    state = REQUESTS.get(rid)
    if state is None or state.kind != response_kind or state.status != "pending":
        return None
    state.status = "approved" if approve else "rejected"
    return state


def respond(rid: str, approve: bool, feedback: str = "") -> str:
    state = REQUESTS.get(rid)
    if state is None:
        return f"Error: unknown request id {rid!r}"
    updated = match_response(state.kind, rid, approve)
    if updated is None:
        return f"Error: request {rid} is {state.status}, already answered"
    _bus().send(
        state.target, state.sender,
        feedback or ("approved" if approve else "rejected"),
        kind=f"{state.kind}_response", rid=rid,
    )
    return f"{state.kind} {rid}: {updated.status}"


def route_teammate(name: str, incoming: list) -> tuple[str, list]:
    """Handle protocol mail addressed to a teammate.

    Returns ("shutdown", leftover) when the teammate should exit now,
    ("", leftover) otherwise; protocol responses become rendered lines
    for the leftover so the model sees the verdict.
    """
    leftover = []
    verdict = ""
    for message in incoming:
        kind = message.get("kind", "")
        if kind == "shutdown_request":
            match_response("shutdown", _rid_of(message), True)
            _bus().send(name, message["from"], "approved", kind="shutdown_response")
            verdict = "shutdown"
        elif kind == "plan_response":
            state = REQUESTS.get(_rid_of(message))
            outcome = state.status if state else "answered"
            leftover.append(
                {"from": message["from"], "kind": "text", "content": f"[Plan {outcome}] {message['content']}"}
            )
        else:
            leftover.append(message)
    return verdict, leftover


def route_lead(incoming: list) -> tuple[list, list]:
    """Split lead mail into reviewable plan requests and plain messages."""
    reviews, plain = [], []
    for message in incoming:
        if message.get("kind") == "plan_request":
            state = ProtocolState(
                rid=_rid_of(message), kind="plan",
                sender=message["from"], target=LEAD, payload=message["content"],
            )
            REQUESTS.setdefault(state.rid, state)
            reviews.append(state)
        else:
            plain.append(message)
    return reviews, plain


def _rid_of(message: dict) -> str:
    return str(message.get("rid", ""))


def render_reviews(reviews: list) -> str:
    return "\n".join(
        f'[Plan review {state.rid} from {state.sender}]\n{state.payload}' for state in reviews
    )


def submit_plan(name: str, plan: str) -> str:
    """Teammate-side: ask the lead to review a plan."""
    if not plan.strip():
        return "Error: plan cannot be empty"
    state = request(name, LEAD, "plan", plan)
    return f"plan {state.rid} submitted for review by the lead"
