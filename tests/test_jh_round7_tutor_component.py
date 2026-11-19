import unittest
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock
from experiments.jh_round7_tutor_component import explain
from tests import test_jh_round7_answer as answer_tests
from pipelines.kimjaehak.round7_tax_scope import TaxBudget12
from experiments.jh_round7_tutor_component import remaining_budget

class ComponentBudgetTests(unittest.TestCase):
    def test_reconstructs_only_available_original_budget(self):
        self.assertEqual(remaining_budget(8),18)
        self.assertEqual(remaining_budget(26.05),0)
        for bad in [-1,100,'8',float('nan')]:
            with self.assertRaises(ValueError):remaining_budget(bad)

class FrozenComponentTests(unittest.IsolatedAsyncioTestCase):
    async def test_freezes_nonanswer_and_uses_recorded_source_budget(self):
        p=answer_tests.AnswerTests().pipeline(TaxBudget12)
        items=[{'id':k,'messages':[{'role':'user','content':'100원의 이자를 계산해줘'}]} for k in ['a','b']]
        baseline={'a':{'id':'a','endpoint':'tutor','response':{'action':'answer','answer':'관련 교재 내용을 인용할게요.\n\n본문','citations':['A']},'seconds':8,'error':None},'b':{'id':'b','endpoint':'tutor','response':{'action':'abstain','answer':'기존 답','citations':[]},'seconds':4,'error':None}}
        original=deepcopy(baseline)
        async def update(state):
            self.assertEqual(state.section,'A')
            self.assertGreater(state._source_extract_deadline-time.monotonic(),17)
            self.assertLessEqual(state._source_extract_deadline-time.monotonic(),18)
            state.answer='새 설명';state.debug.update(r7_apply_attempted=True,r7_apply_completed=True)
        p._explain=AsyncMock(side_effect=update)
        with tempfile.TemporaryDirectory() as d:rows,report=await explain(p,items,baseline,Path(d))
        self.assertEqual(baseline,original)
        self.assertEqual(rows[1]['response'],baseline['b']['response'])
        self.assertEqual(rows[0]['response']['answer'],'새 설명')
        self.assertEqual(report['changed'],1);self.assertEqual(report['completed'],1)
        p._explain.assert_awaited_once()

if __name__=='__main__':unittest.main()
