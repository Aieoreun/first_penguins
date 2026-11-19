import json
import unittest
from unittest.mock import patch
from pipelines import load
load('jh_cm_char_mlp50')
from pipelines.kimjaehak.classifier_matrix import CharMLP50
from pipelines.kimjaehak.round7_structured_calculator import (StructuredCalculator,ScopedStructuredCalculator,
    CalculationSchemaError,CalculationEvidenceError,CalculationExpressionError)
from pipelines.kimjaehak.round7_material_scope import MaterialScope
from tests import test_jh_round7_calculator_span as calculator_tests


class StructuredTests(unittest.IsolatedAsyncioTestCase):
    def ready(self):
        base,s,_=calculator_tests.CalculatorTests().ready()
        p=StructuredCalculator.__new__(StructuredCalculator);p.__dict__.update(base.__dict__)
        p.searcher.section_text=lambda sid:'남은 금액을 매달 저축할 수 있는 돈으로 나누면 기간을 계산해요.'
        obj={'supported':True,'reason':'','calculations':[{'label':'필요한 저축 기간',
            'expression':'ceil((100000-17000)/18000)','unit':'개월','evidence':'남은 금액을 매달 저축할 수 있는 돈으로 나누면'}]}
        return p,s,obj

    async def test_unit_is_rendered_from_structured_result_without_fixed_currency(self):
        p,s,obj=self.ready();old=s.answer;p.llm.chat.return_value=json.dumps(obj)
        with patch.object(CharMLP50,'_generate',return_value=None):await p._generate(s)
        self.assertEqual(s.answer,'필요한 저축 기간: 5개월.\n\n'+old)
        self.assertTrue(s.debug['r7_calculator_completed'])
        self.assertEqual(s.citations,['A'])

    def test_whitespace_only_quote_changes_are_allowed(self):
        p,s,obj=self.ready();obj['calculations'][0]['evidence']='남은 금액을\n매달 저축할 수 있는 돈으로 나누면'
        p._attach_explanation(s,obj);self.assertTrue(s.debug['r7_calculator_completed'])

    def test_reason_specific_validation_does_not_emit_partial_answer(self):
        for kind,exception in [('label',CalculationSchemaError),('evidence',CalculationEvidenceError),
                               ('expression',CalculationExpressionError)]:
            p,s,obj=self.ready();old=s.answer
            if kind=='label':obj['calculations'][0]['label']='금액 99999'
            if kind=='evidence':obj['calculations'][0]['evidence']='없는 교재 규칙'
            if kind=='expression':obj['calculations'][0]['expression']='open("x")'
            with self.assertRaises(exception):p._attach_explanation(s,obj)
            self.assertEqual(s.answer,old)

    def test_composition_and_guard_preserved(self):
        self.assertIs(load('jh_r7_structured_calc'),StructuredCalculator)
        self.assertIs(load('jh_r7_scope_structured'),ScopedStructuredCalculator)
        self.assertIs(ScopedStructuredCalculator._ask,MaterialScope._ask)
        self.assertIs(ScopedStructuredCalculator._attach_explanation,StructuredCalculator._attach_explanation)
        for cls in [StructuredCalculator,ScopedStructuredCalculator]:
            self.assertIs(cls.guard,CharMLP50.guard)
            self.assertIs(cls._fusion_ground,CharMLP50._fusion_ground)


if __name__=='__main__':unittest.main()
