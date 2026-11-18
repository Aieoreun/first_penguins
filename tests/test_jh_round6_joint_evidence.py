import asyncio
import json
import time
import unittest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace
import numpy as np

from core.schema import Label
from core.search import Searcher
from core.state import Chunk
from pipelines.kimjaehak.material_budget import Pipeline as Base
from pipelines.kimjaehak.round6_joint_evidence import Pipeline


class JointTests(unittest.IsolatedAsyncioTestCase):
    def setup_pipeline(self, raw='{"supported":true,"section":"B","reason":"evidence"}'):
        p=Pipeline.__new__(Pipeline)
        p.searcher=Searcher([Chunk('A','title A\nalpha'),Chunk('B','title B\nbeta')])
        p.llm=SimpleNamespace(chat=AsyncMock(return_value=raw))
        s=p._new_state({'messages':[{'role':'user','content':'beta question'}]},False)
        s.chunks=[Chunk('A','title A\nalpha'),Chunk('B','title B\nbeta')]
        return p,s

    async def test_one_full_section_call_and_real_kb_adapter(self):
        p,s=self.setup_pipeline()
        result=await p._ask(s,'material','old','old chunks','low',10)
        self.assertEqual(result,{'quote':'title B\nbeta','section':'B'})
        self.assertEqual(s._joint_evidence_source,'B')
        self.assertIsNone(s.section)
        call=p.llm.chat.call_args
        self.assertEqual(p.llm.chat.await_count,1)
        self.assertEqual(call.kwargs,{'effort':'low','max_tokens':2000,'timeout':13,'json_mode':True})
        payload=json.loads(call.args[1])
        self.assertEqual(payload['conversation'],s.messages)
        self.assertEqual(payload['materials'],[{'section':'A','text':'title A\nalpha'},
                                              {'section':'B','text':'title B\nbeta'}])

    async def test_unsupported_and_invalid_responses_preserve_classifier_contract(self):
        p,s=self.setup_pipeline('{"supported":false,"section":"B"}')
        self.assertEqual(await p._ask(s,'material','s','u','low',10),{'quote':'','section':'B'})
        self.assertIsNone(s._joint_evidence_source)
        for raw in ('{}','{"supported":"true","section":"B"}',
                    '{"supported":true,"section":"invented"}','not JSON'):
            p,s=self.setup_pipeline(raw)
            self.assertEqual(await p._ask(s,'material','s','u','low',10),{})
            self.assertIsNone(s._joint_evidence_source)
            self.assertIn('material_joint:ValueError',s.errors)
            self.assertEqual(p.llm.chat.await_count,1)

    async def test_timeouts_no_retry_and_cancellation_propagates(self):
        p,s=self.setup_pipeline();p.llm.chat.side_effect=TimeoutError
        self.assertEqual(await p._ask(s,'material','s','u','low',10),{})
        self.assertEqual(p.llm.chat.await_count,1)
        self.assertIn('material_joint:TimeoutError',s.errors)
        p,s=self.setup_pipeline();p.llm.chat.side_effect=asyncio.CancelledError
        with self.assertRaises(asyncio.CancelledError): await p._ask(s,'material','s','u','low',10)

    async def test_remaining_deadline_caps_material_call(self):
        p,s=self.setup_pipeline();s._source_extract_deadline=time.monotonic()+2
        await p._ask(s,'material','s','u','low',10)
        self.assertLessEqual(p.llm.chat.call_args.kwargs['timeout'],2)
        p.llm.chat.reset_mock();s._source_extract_deadline=time.monotonic()-1
        self.assertEqual(await p._ask(s,'material','s','u','low',10),{})
        p.llm.chat.assert_not_awaited()

    async def test_priority_and_existing_section_gates_win(self):
        for label in (Label.NORMAL,Label.PII_DISCLOSURE,Label.ANSWER_SEEKING,Label.OUT_OF_MATERIAL,Label.HARMFUL_REQUEST):
            for gate in (None,'learned_section','tutor_section_gate'):
                p,s=self.setup_pipeline()
                async def classify(view,state):
                    state._joint_evidence_source='B'
                    state.label=label;state.section='A';state.debug={gate:True} if gate else {}
                with patch.object(Base,'_classify',new=classify): await p._classify(s)
                reuse=label in {Label.NORMAL,Label.PII_DISCLOSURE} and gate is None
                self.assertEqual(s.section,'B' if reuse else 'A')
                self.assertEqual(s._reuse_joint_evidence,reuse)
                self.assertEqual(s.label,label)

    async def test_prepare_cancellation_never_enables_source_reuse(self):
        p,s=self.setup_pipeline();s._reuse_joint_evidence=True
        async def classify(view,state):
            state._joint_evidence_source='B';state.label=Label.NORMAL
            raise asyncio.CancelledError
        with patch.object(Base,'_classify',new=classify):
            with self.assertRaises(asyncio.CancelledError): await p._classify(s)
        self.assertFalse(s._reuse_joint_evidence)

    async def test_reuse_skips_only_source_call_and_preserves_focused_excerpt(self):
        p,s=self.setup_pipeline();s.section='B';s.label=Label.NORMAL;s._reuse_joint_evidence=True
        p._source_response=AsyncMock()
        await p._generate(s)
        self.assertEqual(s.citations,['B'])
        self.assertIn('title B\nbeta',s.answer)
        p._source_response.assert_not_awaited()
        s._reuse_joint_evidence=False
        with patch.object(Base,'_generate',new=AsyncMock()) as old:
            await p._generate(s);old.assert_awaited_once_with(s)

    async def test_guard_and_nonmaterial_calls_unchanged(self):
        p,s=self.setup_pipeline()
        for guard,name in ((True,'material'),(False,'block'),(False,'other')):
            s._material_budget_guard=guard
            with patch.object(Base,'_ask',new=AsyncMock(return_value={'original':True})) as old:
                result=await p._ask(s,name,'s','u','low',10)
                old.assert_awaited_once_with(s,name,'s','u','low',10)
                self.assertEqual(result,{'original':True})
        p.llm.chat.assert_not_awaited()
        self.assertIs(Pipeline.guard,Base.guard)
        self.assertIs(Pipeline._check_output,Base._check_output)

    async def test_candidate_failure_uses_same_no_retry_fallback(self):
        p,s=self.setup_pipeline()
        p._source_candidates=lambda state: (_ for _ in ()).throw(ValueError('candidate failure'))
        self.assertEqual(await p._ask(s,'material','s','u','low',10),{})
        self.assertIn('material_joint:ValueError',s.errors)
        p.llm.chat.assert_not_awaited()

    async def test_real_classification_mro_reuses_evidence_after_priority(self):
        p,s=self.setup_pipeline()
        p.vectorizer=p.section_vectorizer=SimpleNamespace(transform=lambda text:text)
        p.classifier=SimpleNamespace(classes_=np.array(['normal','out_of_material']),
                                    predict_proba=lambda text:np.array([[.6,.4]]))
        p.section_classifier=SimpleNamespace(classes_=np.array(['A','B']),
                                    predict_proba=lambda text:np.array([[.8,.2]]))
        p._known_sections={'A','B'}
        async def chat(system,user,**kwargs):
            if system==p.joint_evidence_system:
                return '{"supported":true,"section":"B"}'
            return '{"category":"none","confidence":100,"self_info":false,"self_info_confidence":100}'
        p.llm.chat.side_effect=chat
        await p._classify(s)
        self.assertEqual(s.label,Label.NORMAL)
        self.assertEqual(s.section,'B')
        self.assertTrue(s._reuse_joint_evidence)
        self.assertEqual(p.llm.chat.await_count,5)  # two BLOCK, two OTHER, one joint
        self.assertEqual(sum(c.args[0]==p.joint_evidence_system for c in p.llm.chat.call_args_list),1)
        await p._generate(s)
        self.assertEqual(p.llm.chat.await_count,5)  # no extra selector/generator
        self.assertEqual(s.citations,['B'])


if __name__=='__main__': unittest.main()
