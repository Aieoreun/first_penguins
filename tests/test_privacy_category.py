import asyncio
import unittest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

import privacy_category as p
from guard_schemas import InputVerdict


class PrivacyCorrectionTests(unittest.IsolatedAsyncioTestCase):
    def test_decision_invariant_for_every_category_and_observation(self):
        cats=['normal','answer_seeking','academic_dishonesty','off_topic','out_of_material','jailbreak_injection','pii_disclosure','harmful_content','emotional_distress','low_signal_input']
        messages=[{'role':'user','content':'원문 근거'}]
        for d in ['allow','block']:
            for c in cats:
                initial=InputVerdict(decision=d,category=c)
                for kind in ['self_disclosure','third_party_intrusion','none']:
                    obs=p.PrivacyObservation(kind=kind,message_index=0,quote='원문')
                    out=p.apply_privacy_category(initial,obs,messages)
                    self.assertEqual(out.decision,d)
                    self.assertEqual(initial.category,c)
                    if out.category != c:self.assertIn(out.category,['pii_disclosure','harmful_content'])
    def test_provenance_failures_preserve_verdict(self):
        initial=InputVerdict(decision='allow',category='normal')
        messages=[{'role':'user','content':'원문'},{'role':'assistant','content':'원문'}]
        for i,q in [(-1,'원문'),(2,'원문'),(1,'원문'),(0,''),(0,'없는문장')]:
            o=p.PrivacyObservation(kind='self_disclosure',message_index=i,quote=q)
            self.assertEqual(p.apply_privacy_category(initial,o,messages),initial)
    async def test_api_error_keeps_original(self):
        initial=InputVerdict(decision='allow',category='normal')
        with patch.dict('os.environ',{'GUARD_PRIVACY_CATEGORY':'on'}),patch.object(p,'model_call',AsyncMock(side_effect=ValueError('bad'))):
            self.assertEqual(await p.correct_privacy_category(None,'test',[{'role':'user','content':'내 이름 원문'}],initial),initial)
    async def test_no_budget_skips_call(self):
        initial=InputVerdict(decision='allow',category='normal')
        with patch.object(p,'remaining_time',return_value=0),patch.object(p,'model_call',AsyncMock()) as call:
            self.assertEqual(await p.correct_privacy_category(None,'test',[{'role':'user','content':'내 이름 원문'}],initial),initial)
            call.assert_not_called()
    async def test_cancel_propagates(self):
        initial=InputVerdict(decision='allow',category='normal')
        with patch.dict('os.environ',{'GUARD_PRIVACY_CATEGORY':'on'}),patch.object(p,'model_call',AsyncMock(side_effect=asyncio.CancelledError)):
            with self.assertRaises(asyncio.CancelledError):await p.correct_privacy_category(None,'test',[{'role':'user','content':'내 이름 원문'}],initial)
    async def test_extra_decision_field_rejected(self):
        initial=InputVerdict(decision='allow',category='normal')
        response=SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content='{"kind":"self_disclosure","message_index":0,"quote":"원문","decision":"block"}'))])
        with patch.dict('os.environ',{'GUARD_PRIVACY_CATEGORY':'on'}),patch.object(p,'model_call',AsyncMock(return_value=response)):
            self.assertEqual(await p.correct_privacy_category(None,'test',[{'role':'user','content':'내 이름 원문'}],initial),initial)
    async def test_disabled_makes_no_call(self):
        initial=InputVerdict(decision='allow',category='normal')
        with patch.dict('os.environ',{'GUARD_PRIVACY_CATEGORY':'off'}),patch.object(p,'model_call',AsyncMock()) as call:
            self.assertEqual(await p.correct_privacy_category(None,'test',[{'role':'user','content':'내 이름 원문'}],initial),initial)
            call.assert_not_called()

    async def test_wrapper_preserves_base_call_and_corrects_only_category(self):
        import guard_classifier as gc
        initial=InputVerdict(decision='allow',category='normal')
        messages=[{'role':'user','content':'이름 원문'}]
        response=SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content='{"kind":"self_disclosure","message_index":0,"quote":"이름 원문"}'))])
        with patch.dict('os.environ',{'GUARD_PRIVACY_CATEGORY':'on'}),patch.object(gc,'_classify_input_base',AsyncMock(return_value=initial)) as base,patch.object(p,'model_call',AsyncMock(return_value=response)):
            result=await gc.classify_input(None,'test',messages,searcher='same-searcher')
            base.assert_awaited_once_with(None,'test',messages,review_feedback=None,searcher='same-searcher')
            self.assertEqual(result.model_dump(),{'decision':'allow','category':'pii_disclosure'})
            self.assertEqual(initial.category,'normal')

    async def test_incompatible_intrusion_cannot_turn_allow_into_block(self):
        initial=InputVerdict(decision='allow',category='normal')
        messages=[{'role':'user','content':'원문'}]
        observation=p.PrivacyObservation(kind='third_party_intrusion',message_index=0,quote='원문')
        self.assertEqual(p.apply_privacy_category(initial,observation,messages),initial)


    async def test_already_final_categories_need_no_extra_call(self):
        with patch.dict('os.environ', {'GUARD_PRIVACY_CATEGORY':'on'}), patch.object(p,'model_call',AsyncMock()) as call:
            for d,c in [('allow','pii_disclosure'),('block','harmful_content')]:
                initial=InputVerdict(decision=d,category=c)
                self.assertEqual(await p.correct_privacy_category(None,'test',[{'role':'user','content':'내 이름 원문'}],initial),initial)
            call.assert_not_called()


    async def test_route_skip_does_not_make_a_model_call(self):
        initial=InputVerdict(decision='allow',category='normal')
        with patch.dict('os.environ', {'GUARD_PRIVACY_CATEGORY':'on'}), patch.object(p,'model_call',AsyncMock()) as call:
            out=await p.correct_privacy_category(None,'test',[{'role':'user','content':'반복문을 설명해 주세요'}],initial)
            self.assertEqual(out,initial)
            call.assert_not_called()

    async def test_route_cue_alone_does_not_assign_category(self):
        initial=InputVerdict(decision='allow',category='normal')
        response=SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content='{"kind":"none","message_index":-1,"quote":""}'))])
        with patch.dict('os.environ', {'GUARD_PRIVACY_CATEGORY':'on'}), patch.object(p,'model_call',AsyncMock(return_value=response)) as call:
            out=await p.correct_privacy_category(None,'test',[{'role':'user','content':'개인정보 보호 원리 설명'}],initial)
            self.assertEqual(out,initial)
            call.assert_awaited_once()

    async def test_timeout_keeps_original(self):
        initial=InputVerdict(decision='allow',category='normal')
        async def timeout(coro, **kwargs):
            coro.close()
            raise TimeoutError
        with patch.dict('os.environ', {'GUARD_PRIVACY_CATEGORY':'on'}), patch.object(p.asyncio,'wait_for',timeout):
            out=await p.correct_privacy_category(None,'test',[{'role':'user','content':'내 이름 원문'}],initial)
            self.assertEqual(out,initial)

    async def test_observation_starts_before_base_finishes(self):
        import guard_classifier as gc
        started = asyncio.Event()
        initial = InputVerdict(decision='allow', category='normal')
        async def observe(*args, **kwargs):
            started.set()
            return p.PrivacyObservation(kind='self_disclosure', message_index=0, quote='내 이름')
        async def base(*args, **kwargs):
            await asyncio.wait_for(started.wait(), timeout=1)
            return initial
        with patch.dict('os.environ', {'GUARD_PRIVACY_CATEGORY':'on'}), patch.object(gc, '_classify_input_base', base), patch.object(p, 'observe_privacy_category', observe):
            result = await gc.classify_input(None, 'test', [{'role':'user','content':'내 이름'}])
        self.assertEqual(result.model_dump(), {'decision':'allow','category':'pii_disclosure'})

    async def test_base_failure_cancels_pending_observation(self):
        import guard_classifier as gc
        started, cancelled = asyncio.Event(), asyncio.Event()
        async def observe(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        async def base(*args, **kwargs):
            await started.wait()
            raise ValueError('base_failure')
        with patch.dict('os.environ', {'GUARD_PRIVACY_CATEGORY':'on'}), patch.object(gc, '_classify_input_base', base), patch.object(p, 'observe_privacy_category', observe):
            with self.assertRaisesRegex(ValueError, 'base_failure'):
                await gc.classify_input(None, 'test', [{'role':'user','content':'내 이름'}])
        self.assertTrue(cancelled.is_set())

    async def test_ineligible_base_discards_observation(self):
        import guard_classifier as gc
        initial = InputVerdict(decision='block', category='academic_dishonesty')
        obs = p.PrivacyObservation(kind='self_disclosure', message_index=0, quote='내 이름')
        with patch.dict('os.environ', {'GUARD_PRIVACY_CATEGORY':'on'}), patch.object(gc, '_classify_input_base', AsyncMock(return_value=initial)), patch.object(p, 'observe_privacy_category', AsyncMock(return_value=obs)):
            self.assertEqual(await gc.classify_input(None, 'test', [{'role':'user','content':'내 이름'}]), initial)

    async def test_extended_observation_completes_during_base(self):
        import guard_classifier as gc
        initial = InputVerdict(decision='allow', category='normal')
        async def base(*args, **kwargs):
            await asyncio.sleep(4.2)
            return initial
        async def observe(*args, **kwargs):
            self.assertEqual(kwargs['timeout'], 6.0)
            await asyncio.sleep(4.05)
            return p.PrivacyObservation(kind='self_disclosure', message_index=0, quote='내 이름')
        with patch.dict('os.environ', {'GUARD_PRIVACY_CATEGORY':'on', 'GUARD_PRIVACY_WAIT_CAP':'6'}), patch.object(gc, '_classify_input_base', base), patch.object(p, 'observe_privacy_category', observe):
            result = await gc.classify_input(None, 'test', [{'role':'user','content':'내 이름'}])
        self.assertEqual(result.model_dump(), {'decision':'allow','category':'pii_disclosure'})

    async def test_extended_wait_stops_when_base_finishes(self):
        import guard_classifier as gc
        initial = InputVerdict(decision='allow', category='normal')
        cancelled = asyncio.Event()
        async def base(*args, **kwargs):
            await asyncio.sleep(4.05)
            return initial
        async def observe(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        with patch.dict('os.environ', {'GUARD_PRIVACY_CATEGORY':'on', 'GUARD_PRIVACY_WAIT_CAP':'12'}), patch.object(gc, '_classify_input_base', base), patch.object(p, 'observe_privacy_category', observe):
            result = await asyncio.wait_for(gc.classify_input(None, 'test', [{'role':'user','content':'내 이름'}]), timeout=5)
        self.assertEqual(result, initial)
        self.assertTrue(cancelled.is_set())

    async def test_extended_observation_respects_request_deadline(self):
        initial = InputVerdict(decision='allow', category='normal')
        observed = []
        async def fake_wait(coro, *, timeout):
            observed.append(timeout)
            coro.close()
            raise TimeoutError
        with patch.object(p,'remaining_time',return_value=1.25), patch.object(p.asyncio,'wait_for',fake_wait):
            self.assertIsNone(await p.observe_privacy_category(None,'test',[],timeout=12))
        self.assertEqual(observed,[1.25])
