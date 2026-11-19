"""Spend more reasoning tokens on short input checks, without repeated votes.

The provided proxy permits >2000 completion tokens only in streaming mode.
Streaming is internal; the public HTTP response remains the required JSON.
"""
import asyncio

from pipelines.v1.pipeline import _json_object
from .calibrated import BLOCK_SYSTEM, Pipeline as Calibrated


async def stream_text(llm, system, user, *, effort, max_tokens, timeout, json_mode=True):
    async def consume():
        stream = await llm._chat.chat.completions.create(
            model=llm.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            extra_body={"reasoning_effort": effort}, max_completion_tokens=max_tokens,
            stream=True, **({"response_format": {"type": "json_object"}} if json_mode else {}))
        parts = []
        async with stream:
            async for chunk in stream:
                if chunk.choices:
                    content = chunk.choices[0].delta.content
                    if content:
                        parts.append(content)
        return "".join(parts).strip()
    # Covers connection establishment, first token and the entire stream.
    return await asyncio.wait_for(consume(), timeout)


class Pipeline(Calibrated):
    short_effort = "medium"
    short_max_tokens = 4000
    short_timeout = 12

    async def _ask(self, state, name, system, user, effort, timeout):
        if name not in {"block", "other"}:
            return await super()._ask(state, name, system, user, effort, timeout)
        if name == "block":
            system = BLOCK_SYSTEM
        try:
            text = await stream_text(self.llm, system, user, effort=self.short_effort,
                                     max_tokens=self.short_max_tokens, timeout=self.short_timeout)
            obj = _json_object(text)
            if obj is None:
                state.errors.append(f"{name}:parse")
            return obj or {}
        except Exception as exc:
            state.errors.append(f"{name}:{type(exc).__name__}")
            return {}
