import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,AsyncMock,patch
from pipelines import load
load('jh_r7_binary_guard')
from core.schema import Decision
from pipelines.kimjaehak.round7_binary_guard import BinaryGuard
from pipelines.kimjaehak.round8_quantity import QuantityValuePrompt
from pipelines.kimjaehak.round8_span_veto import SpanVeto


class QuantityTests(unittest.TestCase):
    def test_quantity_route(self):
        p=QuantityValuePrompt.__new__(QuantityValuePrompt)
        for text in ['3학년이에요. 기본 이자율이 얼마인가요?', '1학년인데 포인트 환불 규칙은?']:
            self.assertFalse(p._numeric_request(SimpleNamespace(last_user=text)))
        for text in ['420,000원을 연 3.2%로 5개월 맡긴 이자는?', '매달 5천 원 납입이 가능한가요?',
                     '193.5점의 포인트 계산에서 반올림하나요?', '15일 잔액 계산을 설명해줘']:
            self.assertTrue(p._numeric_request(SimpleNamespace(last_user=text)))
        self.assertIs(load('jh_r8_quantity'),QuantityValuePrompt)
        self.assertIs(QuantityValuePrompt.guard,BinaryGuard.guard)


class SpanTests(unittest.IsolatedAsyncioTestCase):
    def states(self):
        return (SimpleNamespace(decision=Decision.ALLOW,debug={}),
                SimpleNamespace(has_candidate=True,candidate_answer='근거 없는 문장',grounded=True,unsupported=[],errors=[]))

    async def test_high_confidence_veto_changes_ground_only(self):
        p=SpanVeto.__new__(SpanVeto);p.span_tagger=SimpleNamespace(predict=Mock(return_value=(['없는 문장'],.99)))
        i,g=self.states();g.safe=True
        with patch.object(BinaryGuard,'_fusion_post_checks',new_callable=AsyncMock):
            await p._fusion_post_checks(i,g,time.monotonic())
        self.assertFalse(g.grounded);self.assertEqual(g.unsupported,['없는 문장']);self.assertTrue(g.safe)
        self.assertEqual(i.debug,{'r8_span_veto':True})

    async def test_existing_error_block_budget_and_length_preserved(self):
        p=SpanVeto.__new__(SpanVeto);p.span_tagger=SimpleNamespace(predict=Mock())
        for condition in ['block','existing','budget','long','none']:
            i,g=self.states();started=time.monotonic()
            if condition=='block':i.decision=Decision.BLOCK
            if condition=='existing':g.grounded=False;g.unsupported=['없는 문장']
            if condition=='budget':started-=28
            if condition=='long':g.candidate_answer='가'*4001
            if condition=='none':g=None
            with patch.object(BinaryGuard,'_fusion_post_checks',new_callable=AsyncMock):
                await p._fusion_post_checks(i,g,started)
            p.span_tagger.predict.assert_not_called()

    async def test_uncertain_and_invalid_spans_preserve_verdict(self):
        p=SpanVeto.__new__(SpanVeto)
        for spans,confidence,error in [(['없는 문장'],.9,False),([],1.,False),(['외부 문자열'],.99,True)]:
            p.span_tagger=SimpleNamespace(predict=Mock(return_value=(spans,confidence)))
            i,g=self.states()
            with patch.object(BinaryGuard,'_fusion_post_checks',new_callable=AsyncMock):
                await p._fusion_post_checks(i,g,time.monotonic())
            self.assertTrue(g.grounded);self.assertEqual(g.unsupported,[])
            self.assertEqual(bool(g.errors),error)

    def test_scope(self):
        self.assertIs(load('jh_r8_span_veto'),SpanVeto)
        self.assertIs(SpanVeto._classify,BinaryGuard._classify)
        self.assertIs(SpanVeto.tutor,BinaryGuard.tutor)
        self.assertEqual(SpanVeto.guard_deadline,28)


if __name__=='__main__':unittest.main()
