import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from pipelines import load
load('jh_r7_binary_guard')
from pipelines.kimjaehak.round8_call_limit import QueuedChat, GuardCalls96, GuardCalls128
from pipelines.kimjaehak.round7_binary_guard import BinaryGuard


class CallLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_queue_time_subtracted_and_shared_limit(self):
        active = peak = 0
        async def chat(*args, timeout, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            self.assertLess(timeout, .5)
            await asyncio.sleep(.01)
            active -= 1
            return 'ok'
        llm = SimpleNamespace(chat=AsyncMock(side_effect=chat), can_embed=True)
        wrapped = QueuedChat(llm, asyncio.Semaphore(1))
        self.assertTrue(wrapped.can_embed)
        self.assertEqual(await asyncio.gather(*(wrapped.chat('s','u',timeout=.5) for _ in range(3))),['ok']*3)
        self.assertEqual(peak,1)

    async def test_expired_queue_does_not_call_api_or_leak_permit(self):
        llm = SimpleNamespace(chat=AsyncMock(return_value='ok'))
        sem = asyncio.Semaphore(0)
        wrapped = QueuedChat(llm,sem)
        with self.assertRaises(TimeoutError):await wrapped.chat('s','u',timeout=.01)
        llm.chat.assert_not_awaited()
        sem.release()
        self.assertEqual(await wrapped.chat('s','u',timeout=.1),'ok')
        self.assertEqual(sem._value,1)

    async def test_active_cancellation_releases_permit(self):
        entered=asyncio.Event()
        async def chat(*args,**kwargs):
            entered.set();await asyncio.Future()
        sem=asyncio.Semaphore(1)
        wrapped=QueuedChat(SimpleNamespace(chat=chat),sem)
        task=asyncio.create_task(wrapped.chat('s','u',timeout=1))
        await entered.wait();task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(sem._value,1)

    def test_only_guard_scheduling_changes(self):
        for name,cls in [('jh_r8_calls96',GuardCalls96),('jh_r8_calls128',GuardCalls128)]:
            self.assertIs(load(name),cls)
            for method in ['tutor','_classify','_fusion_ground','_run_safety','_search']:
                self.assertIs(getattr(cls,method),getattr(BinaryGuard,method))
            self.assertEqual(cls.guard_deadline,28)


if __name__=='__main__':unittest.main()
