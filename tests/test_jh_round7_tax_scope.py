import unittest
from unittest.mock import AsyncMock,Mock,patch
from pipelines import load
load('jh_cm_char_mlp50')
from pipelines.kimjaehak.classifier_matrix import CharMLP50
from pipelines.kimjaehak.round7_tax_scope import ExplicitTaxScope,TaxScopedConcrete,excluded_tax_request,apply_scope
from pipelines.kimjaehak.round7_material_scope import ScopedConcrete
from tests import test_jh_round7_answer as answer_tests

MATERIAL='이 과정에서는 이자는 단리이고 세금은 계산하지 않아요. 아래 결과는 세금을 빼기 전 금액이에요.'

class TaxScopeTests(unittest.IsolatedAsyncioTestCase):
    def test_explicitly_unprovided_after_tax_values_only(self):
        for q in ['200원을 맡기면 실제 세금을 뺀 이자는 얼마인가요?', '800원 이자의 세후 금액을 알려줘', '1000원 단리 계산에 적용할 세율은 얼마인가요?', 'after-tax interest on 200 won?']:
            self.assertTrue(excluded_tax_request(q,MATERIAL))
        for q in ['200원의 세전 이자를 계산해줘','800원 이자 계산에서 세율을 무시하고 세전 금액만 알려줘','200원이고 세율 10%라고 가정하면 세후는 얼마야?','1000원에서 세금 10%를 뺀 값을 계산해줘','세후라는 말은 무슨 뜻이야?','200원의 예금 금리가 얼마예요?']:
            self.assertFalse(excluded_tax_request(q,MATERIAL))
        self.assertFalse(excluded_tax_request('200원의 세후 금액은?',MATERIAL+' 실제 적용 세율은 5%입니다.'))
        self.assertFalse(excluded_tax_request('200원의 세후 금액은?','세금을 반영해 계산해요.'))
        self.assertFalse(excluded_tax_request('200원의 세후 금액은?','세전 이자는 따로 표시해요.'))
        self.assertTrue(excluded_tax_request('200원의 세전 이자와 세후 이자를 비교해줘',MATERIAL))

    def ready(self,cls):
        p,s=answer_tests.AnswerTests().ready(cls)
        s.messages=[{'role':'user','content':'200원의 실제 세금을 뺀 이자는 얼마인가요?'}]
        p.searcher.section_text=Mock(return_value=MATERIAL)
        return p,s

    async def test_scope_only_and_explanation_skip_no_extra_api(self):
        p,s=self.ready(ExplicitTaxScope)
        with patch.object(CharMLP50,'_generate',new_callable=AsyncMock):await p._generate(s)
        self.assertTrue(s.debug['r7_explicit_tax_scope']);self.assertEqual(s.citations,[])
        p.llm.chat.assert_not_awaited()
        p,s=self.ready(TaxScopedConcrete)
        with patch.object(ScopedConcrete,'_explain',new_callable=AsyncMock) as base:await p._explain(s)
        base.assert_not_awaited();self.assertTrue(s.debug['r7_explicit_tax_scope'])

    async def test_rate_in_earlier_turn_preserves_original_generation(self):
        p,s=self.ready(TaxScopedConcrete)
        s.messages.insert(0,{'role':'user','content':'세율을 10%로 가정해서 계산할게요.'})
        with patch.object(ScopedConcrete,'_explain',new_callable=AsyncMock) as base:await p._explain(s)
        base.assert_awaited_once_with(s)
        self.assertNotIn('r7_explicit_tax_scope',s.debug)

    async def test_ordinary_query_and_guard_preserved(self):
        p,s=self.ready(TaxScopedConcrete);s.messages=[{'role':'user','content':'200원의 세전 이자를 계산해줘'}]
        with patch.object(ScopedConcrete,'_explain',new_callable=AsyncMock) as base:await p._explain(s)
        base.assert_awaited_once_with(s)
        p,s=self.ready(ExplicitTaxScope);s._material_budget_guard=True
        self.assertFalse(apply_scope(p,s));self.assertNotIn('r7_explicit_tax_scope',s.debug)
        for name,cls in [('jh_r7_tax_scope',ExplicitTaxScope),('jh_r7_tax_concrete',TaxScopedConcrete)]:
            self.assertIs(load(name),cls)
            for method in ['guard','_classify','_fusion_ground','_fusion_safety','_search']:
                self.assertIs(getattr(cls,method),getattr(CharMLP50,method))

if __name__=='__main__':unittest.main()
