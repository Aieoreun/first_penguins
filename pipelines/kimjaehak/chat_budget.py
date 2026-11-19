"""Limit bursts of chat calls while retaining the request and per-call deadlines."""
import asyncio

from .guard_fitted import ContextFast


class LimitedChat:
    def __init__(self, base, slots):
        self.base = base
        self.slots = asyncio.Semaphore(slots)

    def __getattr__(self, name):
        return getattr(self.base, name)

    async def chat(self, system, user, *, timeout, **kwargs):
        async def run():
            async with self.slots:
                return await self.base.chat(system, user, timeout=timeout, **kwargs)
        # Queue time counts toward the original stage timeout. Cancellation
        # releases either the queued acquisition or the active slot.
        return await asyncio.wait_for(run(), timeout)


class Pipeline(ContextFast):
    chat_slots = 128

    def __init__(self, searcher, llm):
        # This control uses only non-streaming chat calls. Embeddings use their
        # separate endpoint and remain untouched by this chat semaphore.
        super().__init__(searcher, LimitedChat(llm, self.chat_slots))
