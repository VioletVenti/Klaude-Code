"""Recovery for the three failure modes of a live agent run.

Truncation (finish_reason=length): escalate max_tokens once, then keep
the partial output and ask the model to resume directly. Context overflow
rides s08's reactive compact. Transient server trouble (429/529/timeout)
gets exponential backoff with jitter, honoring Retry-After, and a model
fallback after repeated overload.
"""

import os
import random
import time
from dataclasses import dataclass

ESCALATED_MAX_TOKENS = 64_000
DEFAULT_MAX_TOKENS = 8192
MAX_TRANSIENT_RETRIES = 10
BASE_DELAY_MS = 500
MAX_DELAY_MS = 32_000
MAX_CONTINUATIONS = 3
GROWTH_FLOOR = 500  # minimal chars of fresh content per continuation
FALLBACK_AFTER_OVERLOADS = 3

CONTINUATION_PROMPT = (
    "Your reply was cut off by the token limit. Resume directly — no apology, "
    "no recap of what you were doing. Continue from where you stopped and "
    "keep the remaining work compact."
)

TRANSIENT_MARKERS = ("429", "529", "rate limit", "overloaded", "timeout", "timed out", "temporarily", "connection")
OVERLOAD_MARKERS = ("529", "overloaded")


@dataclass
class RecoveryState:
    model: str = ""
    max_tokens: int = DEFAULT_MAX_TOKENS
    escalated: bool = False
    continuations: int = 0
    stalled: bool = False
    last_content_len: int = 0
    consecutive_overloads: int = 0


def is_transient(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in TRANSIENT_MARKERS)


def is_overload(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in OVERLOAD_MARKERS)


def is_truncated(response) -> bool:
    try:
        return response.choices[0].finish_reason == "length"
    except (AttributeError, IndexError):
        return False


def retry_delay(attempt: int, exc: Exception | None = None) -> float:
    """Exponential backoff with jitter; server-provided Retry-After wins."""
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is None:
        response = getattr(exc, "response", None)
        retry_after = getattr(response, "headers", {}) and response.headers.get("retry-after")
    try:
        if retry_after not in (None, ""):
            return min(float(retry_after), MAX_DELAY_MS / 1000 + 30)
    except (TypeError, ValueError):
        pass
    base = min(BASE_DELAY_MS * (2 ** attempt), MAX_DELAY_MS) / 1000
    return base * (1 + random.uniform(0, 0.25))


def with_retry(call, rec: RecoveryState, sleep=time.sleep):
    """Run a model call, absorbing transient failures with backoff."""
    if not rec.model:
        rec.model = os.environ.get("KLAUDE_MODEL", "glm-5.3")
    for attempt in range(MAX_TRANSIENT_RETRIES):
        try:
            return call()
        except Exception as exc:
            if not is_transient(exc):
                raise
            if is_overload(exc):
                rec.consecutive_overloads += 1
                if rec.consecutive_overloads >= FALLBACK_AFTER_OVERLOADS:
                    fallback = os.environ.get("KLAUDE_FALLBACK_MODEL")
                    if fallback and rec.model != fallback:
                        rec.model = fallback
                        print(f"\033[33m[recovery] switching to fallback model {fallback}\033[0m")
                    rec.consecutive_overloads = 0
            else:
                rec.consecutive_overloads = 0
            if attempt == MAX_TRANSIENT_RETRIES - 1:
                raise
            sleep(retry_delay(attempt, exc))
    raise RuntimeError("unreachable")


def note_truncation(rec: RecoveryState, partial_len: int) -> str:
    """Decide what to do about a truncated reply; returns the action."""
    if not rec.escalated:
        rec.escalated = True
        rec.max_tokens = ESCALATED_MAX_TOKENS
        return "escalate"
    if rec.continuations >= MAX_CONTINUATIONS:
        return "give_up"
    growth = partial_len - rec.last_content_len
    rec.last_content_len = partial_len
    if rec.continuations >= 1 and growth < GROWTH_FLOOR:
        rec.stalled = True
        return "give_up"
    rec.continuations += 1
    return "continue"
