import asyncio
import json
import time
import unittest
from unittest.mock import patch
from pipelines import load
load('jh_cm_char_mlp50')
from pipelines.kimjaehak.classifier_matrix import CharMLP50
from pipelines.kimjaehak.round7_calculator_review import ReviewedCalculator
from tests import test_jh_round7_answer as answer_tests


DRAFT={'supported':True,'reason':'','calculations':[{'label':'이자','expression':'100*0.04*3','unit':'원'}]}
FIXED={'verified':True,'supported':True,'reason':'','calculations':[{'label':'이자','expression':'100*0.04*3/12','unit':'원'}]}


class ReviewTests(unittest.IsolatedAsyncioTestCase):
    def ready(self):
        p,s=answer_tests.AnswerTests().ready(ReviewedCalculator)
        s.messages=[{'role':'user','content':'100원 이자 계산해줘'}]
        p.llm.chat.side_effect=[json.dumps(DRAFT),json.dumps(FIXED)]
        return p,s

    async def test_reviewed_expression_only_is_rendered(self):
        p,s=self.ready();old=s.answer
        await p._explain(s)
        self.assertEqual(s.answer,'이자: 1원.\n\n'+old)
        self.assertTrue(s.debug['r7_calc_review_changed'])
        self.assertEqual(p.llm.chat.await_count,2)
        self.assertEqual(json.loads(p.llm.chat.await_args.args[1])['draft'],DRAFT)
        for call in p.llm.chat.await_args_list:
            self.assertEqual(call.kwargs['max_tokens'],2000)
            self.assertEqual(call.kwargs['effort'],'low')
            self.assertLessEqual(call.kwargs['timeout'],8)

    async def test_unverified_timeout_and_invalid_expression_preserve_original(self):
        for review in [{'verified':False},TimeoutError(),{**FIXED,'calculations':[{'label':'값','expression':'__import__("os")','unit':'원'}]}]:
            p,s=self.ready();old=s.answer;cit=list(s.citations)
            p.llm.chat.side_effect=[json.dumps(DRAFT),review if isinstance(review,Exception) else json.dumps(review)]
            await p._explain(s)
            self.assertEqual((s.answer,s.citations),(old,cit))
            self.assertNotIn('r7_apply_completed',s.debug)

    async def test_review_rechecks_budget_after_draft(self):
        p,s=self.ready();old=s.answer
        async def draft(*args,**kwargs):
            s._source_extract_deadline=time.monotonic()+3
            return json.dumps(DRAFT)
        p.llm.chat.side_effect=draft
        await p._explain(s)
        self.assertEqual(s.answer,old);self.assertEqual(p.llm.chat.await_count,1)
        self.assertIn('r7_calc_review:Budget',s.errors)

    async def test_only_verified_missing_material_can_abstain(self):
        p,s=self.ready()
        p.llm.chat.side_effect=[json.dumps(DRAFT),json.dumps({'verified':True,'supported':False,'reason':'missing_material','calculations':[]})]
        await p._explain(s)
        self.assertEqual(s.citations,[])
        self.assertTrue(s.debug['r7_apply_abstained'])

    async def test_guard_skips_and_external_cancel_propagates(self):
        p,s=self.ready();s._material_budget_guard=True
        await p._explain(s);p.llm.chat.assert_not_awaited()
        p,s=self.ready();p.llm.chat.side_effect=[json.dumps(DRAFT),asyncio.CancelledError()]
        with self.assertRaises(asyncio.CancelledError):await p._explain(s)
        self.assertNotIn('r7_apply_completed',s.debug)

    def test_base_guard_and_input_methods_unchanged(self):
        self.assertIs(load('jh_r7_calc_review'),ReviewedCalculator)
        for method in ['guard','_classify','_fusion_ground','_fusion_safety','_search']:
            self.assertIs(getattr(ReviewedCalculator,method),getattr(CharMLP50,method))


if __name__=='__main__':unittest.main()
