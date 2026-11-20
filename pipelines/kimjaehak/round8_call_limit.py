"""Guard-only chat backpressure, including queue time in each call's budget."""
import asyncio
from copy import copy
import time
from .round7_binary_guard import BinaryGuard


class QueuedChat:
    def __init__(self, llm, slots):
        self._wrapped = llm
        self._slots = slots

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    async def chat(self, *args, timeout, **kwargs):
        started = time.monotonic()
        if timeout <= 0:
            raise TimeoutError('No call budget')
        async def work():
            async with self._slots:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError('Chat queue exhausted budget')
                return await self._wrapped.chat(*args, timeout=remaining, **kwargs)
        # The same task owns acquisition and release, including cancellation
        # racing with a newly available permit.
        return await asyncio.wait_for(work(), timeout)


class GuardCalls96(BinaryGuard):
    guard_chat_slots = 96

    def __init__(self, searcher, llm):
        super().__init__(searcher, llm)
        self._guard_chat_slots = asyncio.Semaphore(self.guard_chat_slots)

    async def guard(self, body):
        bounded = copy(self)
        bounded.llm = QueuedChat(self.llm, self._guard_chat_slots)
        return await super(GuardCalls96, bounded).guard(body)


class GuardCalls128(GuardCalls96):
    guard_chat_slots = 128
