import json
import unittest
from unittest.mock import patch
from core.schema import Label
from pipelines import load
load('jh_cm_char_mlp50')
from pipelines.kimjaehak.classifier_matrix import CharMLP50
from pipelines.kimjaehak.round7_calculator import calculate,CalculatorApply,MaterialCalculator
from pipelines.kimjaehak.round7_span_prompt import MinimalClause,AdaptiveMinimal,MINIMAL_CLAUSE_RULE
from pipelines.kimjaehak.round7_adaptive_ground import AdaptiveGround
from tests import test_jh_round7_answer as answer


class CalculatorTests(unittest.IsolatedAsyncioTestCase):
    def test_decimal_arithmetic_units_and_ceiling(self):
        for expr,expected in [('0.1+0.2','0.3'),('25000*(4.2/100)*(12*13/2)/12','6825'),
            ('ceil((100000-17000)/18000)','5'),('-(2-5)*3','9'),('0.00','0')]:
            self.assertEqual(calculate(expr),expected)

    def test_rejects_code_complexity_unbounded_and_implicit_rounding(self):
        for expr in ['__import__("os")','[1][0]','2**999','True','a+1','1e100','0x12',
                'ceil(1,x=2)','1/0','1/3','10000000000000000','1+'*130+'1', '1//2']:
            with self.subTest(expr=expr),self.assertRaises(Exception):calculate(expr)

    def ready(self):
        p,s=answer.AnswerTests().ready(CalculatorApply)
        s.messages[-1]['content']='180000원의 이자를 계산해줘'
        p.searcher.section_text=lambda sid:'이율은 연 4%이며 월 단위로 계산합니다.'
        obj={'supported':True,'reason':'','answer':'이자는 {{0}}원입니다.',
            'calculations':[{'expression':'180000*(4/100)*3/12','evidence':'이율은 연 4%'}]}
        p.llm.chat.return_value=json.dumps(obj,ensure_ascii=False)
        return p,s,obj

    async def test_real_generation_only_inserts_verified_calculation(self):
        p,s,obj=self.ready();old=s.answer
        with patch.object(CharMLP50,'_generate',return_value=None):await p._generate(s)
        self.assertEqual(s.answer,'이자는 1800원입니다.\n\n'+old)
        self.assertEqual(s.citations,['A']);self.assertTrue(s.debug['r7_calculator_completed'])
        self.assertEqual(p.llm.chat.await_count,1)
        self.assertEqual(p.llm.chat.await_args.kwargs['max_tokens'],2000)
        self.assertEqual(p.llm.chat.await_args.kwargs['timeout'],8)

    async def test_invalid_template_or_evidence_keeps_original(self):
        for mode in ['literal','missing','extra','evidence','expression']:
            p,s,obj=self.ready();old=s.answer
            if mode=='literal':obj['answer']='이자는 1800원입니다.'
            if mode=='missing':obj['answer']='이자는 {{1}}원입니다.'
            if mode=='extra':obj['answer']='이자는 {{0}}원, 99개월입니다.'
            if mode=='evidence':obj['calculations'][0]['evidence']='교재에 없는 이율'
            if mode=='expression':obj['calculations'][0]['expression']='__import__("os")'
            p.llm.chat.return_value=json.dumps(obj)
            with patch.object(CharMLP50,'_generate',return_value=None):await p._generate(s)
            self.assertEqual(s.answer,old);self.assertEqual(s.citations,['A'])
            self.assertNotIn('r7_calculator_completed',s.debug)
            self.assertTrue(s.errors)

    async def test_missing_material_abstains_but_guard_is_untouched(self):
        p,s,obj=self.ready()
        obj={'supported':False,'reason':'missing_material','answer':'','calculations':[]}
        p.llm.chat.return_value=json.dumps(obj)
        with patch.object(CharMLP50,'_generate',return_value=None):await p._generate(s)
        self.assertIs(s.label,Label.OUT_OF_MATERIAL);self.assertEqual(s.citations,[])
        p,s,obj=self.ready();s._material_budget_guard=True
        with patch.object(CharMLP50,'_generate',return_value=None):await p._generate(s)
        p.llm.chat.assert_not_awaited()
        for method in ('guard','_classify','_search','_fusion_ground','_fusion_safety'):
            self.assertIs(getattr(CalculatorApply,method),getattr(CharMLP50,method))
        self.assertIs(load('jh_r7_calculator'),CalculatorApply)

    def test_material_calculator_composition(self):
        from pipelines.kimjaehak.round7_material_scope import MaterialScope
        self.assertIs(load('jh_r7_material_calculator'),MaterialCalculator)
        self.assertIs(MaterialCalculator._ask,MaterialScope._ask)
        self.assertIs(MaterialCalculator._attach_explanation,CalculatorApply._attach_explanation)
        self.assertIs(MaterialCalculator._generate,CalculatorApply._generate)
        self.assertIs(MaterialCalculator.guard,CharMLP50.guard)

    def test_minimal_span_prompt_changes_no_tutor_or_input_methods(self):
        for name,cls in [('jh_r7_minimal_span',MinimalClause),('jh_r7_adaptive_span',AdaptiveMinimal)]:
            self.assertIs(load(name),cls)
            self.assertEqual(cls.unsupported_system,CharMLP50.unsupported_system+MINIMAL_CLAUSE_RULE)
            for method in ('tutor','_generate','_classify','_search','_fusion_safety'):
                self.assertIs(getattr(cls,method),getattr(CharMLP50,method))
        self.assertIs(MinimalClause._fusion_ground,CharMLP50._fusion_ground)
        self.assertIs(AdaptiveMinimal._fusion_ground,AdaptiveGround._fusion_ground)


if __name__=='__main__':unittest.main()
