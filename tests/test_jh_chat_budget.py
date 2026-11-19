import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pipelines.kimjaehak.chat_budget import LimitedChat, Pipeline


class ChatBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_concurrency_is_bounded(self):
        active = peak = 0
        async def chat(*args, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(active, peak)
            try:
                await asyncio.sleep(.005)
                return "ok"
            finally:
                active -= 1
        proxy = LimitedChat(SimpleNamespace(chat=chat), 2)
        values = await asyncio.gather(*(proxy.chat("system", "user", timeout=2) for _ in range(8)))
        self.assertEqual(values, ["ok"] * 8)
        self.assertEqual(peak, 2)
        self.assertEqual(active, 0)

    async def test_queue_timeout_does_not_call_backend_or_leak_capacity(self):
        backend = SimpleNamespace(chat=AsyncMock(return_value="ok"))
        proxy = LimitedChat(backend, 1)
        await proxy.slots.acquire()
        with self.assertRaises(asyncio.TimeoutError):
            await proxy.chat("system", "user", timeout=.01)
        backend.chat.assert_not_awaited()
        proxy.slots.release()
        self.assertEqual(await proxy.chat("system", "user", timeout=1), "ok")

    async def test_cancelled_active_call_releases_slot(self):
        entered = asyncio.Event()
        async def chat(*args, **kwargs):
            entered.set()
            await asyncio.Future()
        backend = SimpleNamespace(chat=chat)
        proxy = LimitedChat(backend, 1)
        task = asyncio.create_task(proxy.chat("system", "user", timeout=10))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        backend.chat = AsyncMock(return_value="next")
        self.assertEqual(await proxy.chat("system", "user", timeout=1), "next")

    async def test_embedding_settings_and_calls_are_forwarded(self):
        backend = SimpleNamespace(can_embed=True, embedding_model="provided-model",
                                  embed=AsyncMock(return_value=[[1.0]]))
        proxy = LimitedChat(backend, 1)
        self.assertTrue(proxy.can_embed)
        self.assertEqual(proxy.embedding_model, "provided-model")
        self.assertEqual(await proxy.embed(["question"], timeout=3), [[1.0]])
        backend.embed.assert_awaited_once_with(["question"], timeout=3)

    def test_variant_loader(self):
        from pipelines import load
        self.assertIs(load("jh_r2_chat128"), Pipeline)
