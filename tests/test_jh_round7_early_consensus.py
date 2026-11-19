import asyncio
import unittest
from unittest.mock import patch
from pipelines import load
load('jh_cm_char_mlp50')
from pipelines.kimjaehak.round7_early_consensus import EarlyConsensus
from pipelines.kimjaehak.context_output import Pipeline as ContextOutput
from tests import test_jh_round7_adaptive_ground as adaptive


class EarlyTests(unittest.IsolatedAsyncioTestCase):
    def ready(self):return adaptive.AdaptiveTests().ready(EarlyConsensus)

    async def test_five_start_together_three_agree_cancel_slow_two(self):
        p,s=self.ready();calls=[];cancelled=[];started=asyncio.Event()
        async def check(instance,trial):
            i=len(calls);calls.append(trial)
            if len(calls)==5:started.set()
            await started.wait()
            if i<3:trial.unsupported=['bad'];return
            try:await asyncio.Event().wait()
            finally:cancelled.append(i)
        with patch.object(ContextOutput,'_check_output',new=check):
            await asyncio.wait_for(p._fusion_ground(s),1)
        self.assertEqual(len(calls),5);self.assertEqual(sorted(cancelled),[3,4])
        self.assertEqual(s.unsupported,['bad']);self.assertTrue(s.debug['r7_ground_early'])
        self.assertEqual(s.debug['fusion_completed_ground_votes'],3)

    async def test_two_agree_cannot_end_and_third_dissent_waits_for_all(self):
        p,s=self.ready();calls=[];first=asyncio.Event();release=asyncio.Event()
        async def check(instance,trial):
            i=len(calls);calls.append(trial)
            if i==1:first.set()
            if i>=2:await release.wait()
            trial.unsupported=[] if i==2 else ['bad']
        with patch.object(ContextOutput,'_check_output',new=check):
            task=asyncio.create_task(p._fusion_ground(s));await first.wait();await asyncio.sleep(0)
            self.assertFalse(task.done());release.set();await task
        self.assertEqual(s.debug['fusion_completed_ground_votes'],5)
        self.assertEqual(s.unsupported,['bad']);self.assertFalse(s.debug.get('r7_ground_early',False))

    async def test_parent_cancellation_keeps_completed_votes_and_cleans_up(self):
        p,s=self.ready();calls=[];ready=asyncio.Event();cancelled=[]
        async def check(instance,trial):
            i=len(calls);calls.append(trial)
            if i==0:trial.unsupported=['bad'];return
            if i==4:ready.set()
            try:await asyncio.Event().wait()
            finally:cancelled.append(i)
        with patch.object(ContextOutput,'_check_output',new=check):
            task=asyncio.create_task(p._fusion_ground(s));await ready.wait();await asyncio.sleep(0);task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(sorted(cancelled),[1,2,3,4]);self.assertEqual(s.unsupported,['bad'])
        self.assertEqual(s.debug['fusion_completed_ground_votes'],1)

    async def test_failed_vote_never_counts_toward_early_agreement(self):
        p,s=self.ready();calls=[]
        async def check(instance,trial):
            i=len(calls);calls.append(trial);trial.unsupported=['bad']
            if i==0:trial.errors.append('failed')
        with patch.object(ContextOutput,'_check_output',new=check):await p._fusion_ground(s)
        self.assertEqual(s.debug['fusion_completed_ground_votes'],4)
        self.assertFalse(s.debug.get('r7_ground_early',False))
        self.assertIs(load('jh_r7_early_consensus'),EarlyConsensus)


if __name__=='__main__':unittest.main()
