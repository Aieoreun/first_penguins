import unittest
from unittest.mock import AsyncMock,patch
from core.schema import Label
from pipelines import load
load('jh_cm_char_mlp50')
from pipelines.kimjaehak.classifier_matrix import CharMLP50
from pipelines.kimjaehak.round7_compact_calculator import NoQuoteCalculator,CompactCalculator
from pipelines.kimjaehak.round7_material_verdict import MaterialVerdict,VERDICT_RULE
from pipelines.kimjaehak.round7_structured_calculator import CalculationExpressionError
from tests import test_jh_round7_structured_calculator as structured
from tests import test_jh_reasoning_ablation as reasoning


class CompactVerdictTests(unittest.IsolatedAsyncioTestCase):
    def test_noquote_keeps_arithmetic_checks_and_same_prompt(self):
        base,s,obj=structured.StructuredTests().ready()
        p=NoQuoteCalculator.__new__(NoQuoteCalculator);p.__dict__.update(base.__dict__)
        obj['calculations'][0]['evidence']='다르게 표현한 근거'
        p._attach_explanation(s,obj);self.assertTrue(s.debug['r7_calculator_completed'])
        self.assertEqual(p.tutor_apply_system,base.tutor_apply_system)
        obj['calculations'][0]['expression']='__import__("os")'
        with self.assertRaises(CalculationExpressionError):p._attach_explanation(s,obj)

    def test_compact_accepts_named_numeric_rules_without_evidence_field(self):
        base,s,obj=structured.StructuredTests().ready()
        p=CompactCalculator.__new__(CompactCalculator);p.__dict__.update(base.__dict__)
        del obj['calculations'][0]['evidence'];obj['calculations'][0]['label']='3·3·4 규칙 목표 기간'
        p._attach_explanation(s,obj)
        self.assertTrue(s.answer.startswith('3·3·4 규칙 목표 기간: 5개월.'))
        self.assertEqual(s.citations,['A'])

    async def test_only_explicit_missing_material_rewrites_quote_and_only_for_tutor(self):
        for guard in [True,False]:
            for status in ['sufficient','missing_material','missing_question_values',None]:
                p=reasoning.ReasoningTests().pipeline(MaterialVerdict);s=reasoning.ReasoningTests().state(p,guard)
                obj={'quote':'교재 관련 사실','section':'A','material_status':status}
                with patch.object(CharMLP50,'_ask',new_callable=AsyncMock,return_value=obj) as parent:
                    result=await p._ask(s,'material','original','payload','low',10)
                self.assertEqual(result['quote'],'' if not guard and status=='missing_material' else obj['quote'])
                self.assertEqual(VERDICT_RULE in parent.await_args.args[2],not guard)
                self.assertEqual(parent.await_args.args[3:],('payload','low',10))

    def test_registry_and_guard_preserved(self):
        for name,cls in [('jh_r7_calc_noquote',NoQuoteCalculator),('jh_r7_calc_compact',CompactCalculator),('jh_r7_material_verdict',MaterialVerdict)]:
            self.assertIs(load(name),cls)
            self.assertIs(cls.guard,CharMLP50.guard)
            self.assertIs(cls._fusion_ground,CharMLP50._fusion_ground)


if __name__=='__main__':unittest.main()
