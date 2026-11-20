import json
import unittest
from unittest.mock import Mock
from core.schema import Label
from pipelines import load
load('jh_r7_binary_guard')
from pipelines.kimjaehak.round8_numeric import ValuePrompt,ValueAnchored,BinaryValuePrompt,BinaryValueAnchored
from pipelines.kimjaehak.round7_binary_guard import BinaryGuard
from pipelines.kimjaehak.round7_tax_scope import TaxScopedConcrete
from tests import test_jh_round7_answer as fixtures

class NumericTests(unittest.IsolatedAsyncioTestCase):
    def ready(self):
        p,s=fixtures.AnswerTests().ready(ValueAnchored)
        s.messages=[{'role':'user','content':'원금 200원으로 3개월의 이자를 계산해줘'}]
        p.searcher.section_text=Mock(return_value='단리 이자는 원금 × 연이율 × 기간(년)입니다.')
        obj={'supported':True,'reason':'','rule_quote':'단리 이자는 원금 × 연이율 × 기간(년)입니다.',
             'question_values':['200원','3개월'],'answer':'질문의 조건으로 계산하면 결과입니다.'}
        return p,s,obj

    async def test_valid_provenance_keeps_original_excerpt_and_budget(self):
        p,s,obj=self.ready();old=s.answer;p.llm.chat.return_value=json.dumps(obj)
        await p._explain(s)
        self.assertTrue(s.debug['r8_value_anchored']);self.assertTrue(s.answer.endswith(old))
        self.assertEqual(s.citations,['A']);self.assertIn('질문 조건: 200원, 3개월',s.answer)
        self.assertLessEqual(p.llm.chat.await_args.kwargs['timeout'],8)
        self.assertEqual(p.llm.chat.await_args.kwargs['max_tokens'],2000)

    async def test_fabricated_rule_or_values_and_oversize_keep_original(self):
        for patch in [{'rule_quote':'교재에 없는 공식'}, {'question_values':['999원']},
                      {'question_values':[]}, {'question_values':['원금']}, {'answer':'x'*241}]:
            p,s,obj=self.ready();old=s.answer;obj.update(patch);p.llm.chat.return_value=json.dumps(obj)
            await p._explain(s)
            self.assertEqual(s.answer,old);self.assertIn('r7_apply:ValueError',s.errors)

    async def test_missing_material_abstains_and_pii_never_uses_value_echo(self):
        p,s,obj=self.ready();p.llm.chat.return_value=json.dumps({'supported':False,'reason':'missing_material','answer':''})
        await p._explain(s);self.assertEqual(s.label,Label.OUT_OF_MATERIAL);self.assertEqual(s.citations,[])
        p,s,obj=self.ready();s.label=Label.PII_DISCLOSURE
        await p._explain(s);p.llm.chat.assert_not_awaited()

    def test_guard_and_tutor_composition(self):
        for name,cls,tutor in [('jh_r8_binary_value_prompt',BinaryValuePrompt,ValuePrompt),('jh_r8_binary_value_anchored',BinaryValueAnchored,ValueAnchored)]:
            self.assertIs(load(name),cls);self.assertIs(cls.__init__,BinaryGuard.__init__)
            for method in ['guard','_classify','_fusion_ground','_fusion_safety','_search']:
                self.assertIs(getattr(cls,method),getattr(BinaryGuard,method))
            self.assertIs(cls._attach_explanation,tutor._attach_explanation)
            self.assertIs(cls._explain,TaxScopedConcrete._explain)
        self.assertIs(load('jh_r8_value_prompt'),ValuePrompt)
        self.assertIs(load('jh_r8_value_anchored'),ValueAnchored)

if __name__=='__main__':unittest.main()
