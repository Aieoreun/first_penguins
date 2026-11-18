import asyncio
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.search import Searcher
from core.state import Chunk
from pipelines.kimjaehak.material_budget import Pipeline as Base
from pipelines.kimjaehak.round6_source import EvidenceFirst, VoteThree


class SourceTests(unittest.IsolatedAsyncioTestCase):
    def test_reordering_preserves_content_and_state(self):
        p = EvidenceFirst.__new__(EvidenceFirst)
        p.searcher = Searcher([Chunk('A','title\nalpha details'),
                              Chunk('A','title\nbeta details'),
                              Chunk('B','other\n```python\nx=1\n```')])
        state = SimpleNamespace(section='B', messages=[{'role':'user','content':'beta'}], last_user='beta')
        payload = p._source_payload(state,['A','B'])
        self.assertEqual(payload['materials'][0]['text'],'title\nbeta details\n\nalpha details')
        self.assertEqual(payload['materials'][1]['text'],'other\n```python\nx=1\n```')
        self.assertEqual(state.section,'B')
        self.assertEqual(payload['conversation'],state.messages)

    async def votes(self, outcomes, timeout=.1):
        p=VoteThree.__new__(VoteThree)
        state=SimpleNamespace(errors=[],debug={})
        counter=iter(range(3)); finished=[]
        async def response(*args,**kwargs):
            i=next(counter); delay,result=outcomes[i]
            try:
                await asyncio.sleep(delay)
                if isinstance(result,Exception): raise result
                return json.dumps({'section':result})
            finally: finished.append(i)
        start=time.monotonic()
        with patch.object(Base,'_source_response',side_effect=response):
            result=await p._source_response(state,['A','B','C'],timeout)
        self.assertEqual(len(finished),3)
        return json.loads(result)['section'],state,time.monotonic()-start

    async def test_majority_finishes_early_and_cancels(self):
        section,state,elapsed=await self.votes([(.002,'B'),(.004,'B'),(1,'A')])
        self.assertEqual(section,'B');self.assertEqual(state.errors,[])
        self.assertLess(elapsed,.08)

    async def test_tie_breaks_by_call_order(self):
        section,_,_=await self.votes([(.02,'A'),(.01,'B'),(.001,'C')])
        self.assertEqual(section,'A')

    async def test_partial_timeout_preserves_valid_vote(self):
        section,state,_=await self.votes([(.001,'B'),(.001,'invalid'),(1,'A')],.03)
        self.assertEqual(section,'B')
        self.assertIn('source_vote:TimeoutError',state.errors)
        self.assertIn('source_vote:ValueError',state.errors)

    async def test_all_invalid_uses_parent_fallback_contract(self):
        with self.assertRaises(ValueError):
            await self.votes([(.001,'invalid')]*3)

    async def test_caller_cancellation_drains_children(self):
        p=VoteThree.__new__(VoteThree); state=SimpleNamespace(errors=[],debug={})
        started=[];finished=[]
        async def slow(*args):
            started.append(1)
            try: await asyncio.sleep(10)
            finally: finished.append(1)
        with patch.object(Base,'_source_response',side_effect=slow):
            task=asyncio.create_task(p._source_response(state,['A','B'],8))
            await asyncio.sleep(.01);task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
        self.assertEqual(len(started),3);self.assertEqual(len(finished),3)

    def test_guard_and_limits_unchanged(self):
        for cls in (EvidenceFirst,VoteThree):
            self.assertIs(cls.guard,Base.guard)
            self.assertIs(cls._check_output,Base._check_output)
            self.assertEqual((cls.deadline,cls.source_extract_timeout),(26,8))


if __name__=='__main__': unittest.main()
