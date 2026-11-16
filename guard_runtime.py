"""Per-request deadlines and content-free operational telemetry (Python 3.10+)."""
import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar


logger = logging.getLogger(__name__)
_request = ContextVar("guard_request", default=None)


@contextmanager
def request_scope(deadline, request_id=None):
    state = {"id": request_id or uuid.uuid4().hex, "started": time.monotonic(),
             "deadline": deadline}
    token = _request.set(state)
    try:
        yield state["id"]
    finally:
        _request.reset(token)


def remaining_time(reserve=0.0):
    state = _request.get()
    return max(0.0, state["deadline"] - time.monotonic() - reserve) if state else float("inf")


@contextmanager
def deadline_scope(deadline):
    state = _request.get()
    if state is None:
        yield
        return
    token = _request.set({**state, "deadline": deadline})
    try:
        yield
    finally:
        _request.reset(token)


def record(event, stage, **fields):
    """Allowlist only metadata; never log exception strings, prompts or verdicts."""
    state = _request.get()
    if state is None:
        return
    allowed = {"seconds", "attempt", "error_type", "reason", "finish_reason",
               "prompt_tokens", "completion_tokens", "reasoning_tokens", "remaining",
               "completed", "status"}
    row = {"request_id": state["id"], "event": event, "stage": stage,
           "elapsed": round(time.monotonic() - state["started"], 4)}
    row.update({k: v for k, v in fields.items() if k in allowed})
    try:
        logger.info("guard_event %s", json.dumps(row, ensure_ascii=True))
    except Exception:
        pass  # Logging must never invalidate a completed verdict.


@asynccontextmanager
async def timed_stage(stage):
    started = time.monotonic()
    record("start", stage)
    try:
        yield
    except asyncio.CancelledError:
        record("cancelled", stage, seconds=round(time.monotonic() - started, 4))
        raise
    except Exception as error:
        record("failure", stage, error_type=type(error).__name__,
               seconds=round(time.monotonic() - started, 4))
        raise
    else:
        record("success", stage, seconds=round(time.monotonic() - started, 4))


async def model_call(client, stage, **kwargs):
    async with timed_stage(stage):
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        record("model_response", stage, finish_reason=choice.finish_reason,
               prompt_tokens=getattr(usage, "prompt_tokens", None),
               completion_tokens=getattr(usage, "completion_tokens", None),
               reasoning_tokens=getattr(details, "reasoning_tokens", None))
        return response


class ModelResponseError(ValueError):
    def __init__(self, reason_code):
        self.reason_code = reason_code
        super().__init__(reason_code)
