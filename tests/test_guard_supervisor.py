import asyncio
import io
import json
import logging
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from openai import BadRequestError, RateLimitError
from pydantic import ValidationError

import guard_supervisor as gs
import guard_review as gr
from guard_output import GroundingVerdict, SafetyVerdict
from guard_review import OutputReviewReport, ReviewReport, review_verdict, review_json_schema
from guard_schemas import InputVerdict

MESSAGES = [{'role': 'user', 'content': '현재 수업의 조건을 알려줘'}]
CANDIDATE = '이 과정의 조건은 85점이에요.'
EVIDENCE = [{'section_id': 'PY-01-01', 'text': '이 과정의 조건은 90점이에요.'}]
INITIAL = {'input': {'decision': 'allow', 'category': 'normal'}, 'output': {
    'grounded': True, 'unsupported': [], 'safe': True, 'category': 'normal'}}


def review(target='input', action='keep', **kwargs):
    return dict(target=target, action=action,
                rule={'input':'input_intent','grounding':'factual_support','safety':'answer_safety'}[target],
                reason='원문 기준', citations=[], **kwargs)


def report(*items):
    return ReviewReport.model_validate({'reviews': list(items)})


def citation(source, source_id, quote):
    return {'source':source, 'source_id':source_id, 'quote':quote}


def grounding_change(span='85점', quote='90점'):
    item = review('grounding', 'revise', grounding={'grounded':False,'unsupported':[span]})
    item['citations'] = [citation('candidate','candidate',CANDIDATE), citation('kb','PY-01-01',quote)]
    return item


def observation(relation="nonfactual", safety="benign"):
    return OutputReviewReport.model_validate({
        "grounding":[{"candidate_span":CANDIDATE,"relation":relation,"reason":"観察",
                      "citations":[]}],
        "safety":{"candidate_span":CANDIDATE,"relation":safety,"reason":"行動"}})


class SchemaAndProvenanceTests(unittest.TestCase):
    def test_fake_quote_is_not_applied(self):
        result = gs.apply_review(INITIAL,report(grounding_change(quote='없는 근거')),
                                 MESSAGES,CANDIDATE,EVIDENCE)
        self.assertEqual(result, INITIAL)

    def test_exact_candidate_span_required(self):
        result = gs.apply_review(INITIAL,report(grounding_change(span='고쳐 쓴 정답')),
                                 MESSAGES,CANDIDATE,EVIDENCE)
        self.assertEqual(result, INITIAL)

    def test_valid_grounding_change_preserves_other_fields(self):
        result = gs.apply_review(INITIAL,report(grounding_change()),MESSAGES,CANDIDATE,EVIDENCE)
        self.assertFalse(result['output']['grounded'])
        self.assertEqual(result['output']['unsupported'],['85점'])
        self.assertEqual(result['input'], INITIAL['input'])
        self.assertTrue(result['output']['safe'])
        self.assertTrue(INITIAL['output']['grounded'])  # original is not mutated

    def test_without_candidate_output_change_is_ignored(self):
        initial={'input':INITIAL['input']}
        result=gs.apply_review(initial,report(grounding_change()),MESSAGES,None,EVIDENCE)
        self.assertEqual(result,initial)

    def test_keep_echoed_verdict_is_ignored(self):
        item=review('input','keep',input={'decision':'block','category':'harmful_content'})
        item['citations']=[citation('conversation','0',MESSAGES[0]['content'])]
        parsed=report(item)
        self.assertIsNone(parsed.reviews[0].input)
        self.assertEqual(gs.apply_review(INITIAL,parsed,MESSAGES,CANDIDATE,EVIDENCE),INITIAL)

    def test_no_candidate_schema_only_allows_input_target(self):
        schema=review_json_schema(False)
        self.assertEqual(schema['$defs']['ReviewItem']['properties']['target']['enum'],['input'])
        self.assertEqual(schema['properties']['reviews']['maxItems'],1)

    def test_bool_is_strict(self):
        item=grounding_change()
        item['grounding']['grounded']='false'
        with self.assertRaises(ValidationError): report(item)

    def test_duplicate_targets_rejected(self):
        with self.assertRaises(ValidationError): report(review(),review())

    def test_wrong_field_scope_rejected(self):
        with self.assertRaises(ValidationError):
            report(review('input','revise',safety={'safe':False}))

    def test_unrelated_source_cannot_change_input(self):
        item=review('input','revise',input={'decision':'block','category':'harmful_content'})
        item['citations']=[citation('kb','PY-01-01','90점')]
        self.assertEqual(gs.apply_review(INITIAL,report(item),MESSAGES,CANDIDATE,EVIDENCE),INITIAL)

    def test_block_and_safe_are_independent(self):
        initial={'input':{'decision':'block','category':'harmful_content'},'output':INITIAL['output']}
        self.assertTrue(gs.validated(initial,CANDIDATE)['output']['safe'])


class SupervisorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.stack=ExitStack()
        self.addCleanup(self.stack.close)
        self.input=self.stack.enter_context(patch.object(gs,'classify_input',new_callable=AsyncMock))
        self.input.return_value=InputVerdict(decision='allow',category='normal')
        self.grounding=self.stack.enter_context(patch.object(gs,'check_grounding',new_callable=AsyncMock))
        self.grounding.return_value=GroundingVerdict(grounded=True,unsupported=[])
        self.safety=self.stack.enter_context(patch.object(gs,'check_safety',new_callable=AsyncMock))
        self.safety.return_value=SafetyVerdict(safe=True)
        self.reviewer=self.stack.enter_context(patch.object(gs,'review_grounding',new_callable=AsyncMock))
        self.reviewer.return_value=gr.GroundingStatusReview(status='supported',rule='factual_support',reason='근거 일치',observations=observation().grounding)
        self.input_reviewer=self.stack.enter_context(patch.object(gs,'review_input_status',new_callable=AsyncMock))
        self.input_reviewer.return_value=gr.InputStatusReview(status='supported',rule='input_intent',reason='일치',citations=[])
        self.safety_reviewer=self.stack.enter_context(patch.object(gs,'review_safety',new_callable=AsyncMock))
        self.safety_reviewer.return_value=gr.SafetyStatusReview(status='supported',rule='answer_safety',reason='일치',observation=observation().safety)
        self.searcher=SimpleNamespace(search=AsyncMock(return_value=EVIDENCE))

    def supervisor(self,mode='all',**kwargs):
        return gs.GuardSupervisor(None,'test-model',self.searcher,review_mode=mode,**kwargs)

    async def test_candidate_absent_runs_input_only(self):
        self.reviewer.return_value=report(review())
        result=await self.supervisor().evaluate(MESSAGES)
        self.assertEqual(result,{'input':INITIAL['input']})
        self.searcher.search.assert_not_awaited()
        self.grounding.assert_not_awaited()
        self.safety.assert_not_awaited()

    async def test_review_timeout_preserves_initial(self):
        async def slow(*args, **kwargs): await asyncio.sleep(1)
        self.reviewer.side_effect=slow
        result=await self.supervisor(total_timeout=3,base_timeout=2,review_timeout=.01).evaluate(MESSAGES,CANDIDATE)
        self.assertEqual(result,INITIAL)
        self.assertEqual(self.reviewer.await_count,1)

    async def test_bad_request_is_not_retried(self):
        response=SimpleNamespace(status_code=400,request=SimpleNamespace(),headers={})
        error=BadRequestError('bad',response=response,body={})
        self.input.side_effect=error
        with self.assertRaises(BadRequestError): await self.supervisor().evaluate(MESSAGES)
        self.assertEqual(self.input.await_count,1)
        self.reviewer.assert_not_awaited()

    async def test_quota_exhaustion_is_not_retried(self):
        response=SimpleNamespace(status_code=429,request=SimpleNamespace(),headers={})
        self.input.side_effect=RateLimitError('quota exhausted',response=response,
                                             body={'code':'insufficient_quota'})
        with self.assertRaises(RateLimitError): await self.supervisor().evaluate(MESSAGES)
        self.assertEqual(self.input.await_count,1)
        self.reviewer.assert_not_awaited()

    async def test_output_observation_failure_preserves_initial(self):
        self.reviewer.side_effect=ValueError('review failed')
        result=await self.supervisor().evaluate(MESSAGES,CANDIDATE)
        self.assertEqual(result,INITIAL)
        self.reviewer.assert_awaited_once()

    async def test_validation_failure_retried_once(self):
        self.input.side_effect=[ValueError('invalid'),InputVerdict(decision='allow',category='normal')]
        result=await self.supervisor(mode='off').evaluate(MESSAGES)
        self.assertEqual(result,{'input':INITIAL['input']})
        self.assertEqual(self.input.await_count,2)

    async def test_repeated_validation_failure_stops(self):
        self.input.side_effect=ValueError('invalid')
        with self.assertRaises(ValueError): await self.supervisor(mode='off').evaluate(MESSAGES)
        self.assertEqual(self.input.await_count,2)

    async def test_base_timeout_cancels_all_stages(self):
        started=set(); cancelled=set()
        def slow(name):
            async def operation(*args,**kwargs):
                started.add(name)
                try: await asyncio.sleep(10)
                finally: cancelled.add(name)
            return operation
        self.input.side_effect=slow('input')
        self.searcher.search.side_effect=slow('search')
        self.safety.side_effect=slow('safety')
        with self.assertRaises(TimeoutError):
            await self.supervisor(total_timeout=1,base_timeout=.02).evaluate(MESSAGES,CANDIDATE)
        self.assertEqual(started,{'input','search','safety'})
        self.assertEqual(cancelled,started)
        self.reviewer.assert_not_awaited()

    async def test_missing_evidence_retrieves_only_once(self):
        self.reviewer.return_value=gr.GroundingStatusReview(status='insufficient',rule='factual_support',reason='조건 자료 부족',observations=observation('insufficient').grounding)
        result=await self.supervisor().evaluate(MESSAGES,CANDIDATE)
        self.assertEqual(result,INITIAL)
        self.assertEqual(self.searcher.search.await_count,2)  # initial plus one recovery
        self.assertEqual(self.reviewer.await_count,1)
        self.assertEqual(self.searcher.search.await_args_list[1].kwargs['top_k'],8)

    async def test_off_mode_never_reviews(self):
        await self.supervisor(mode='off').evaluate(MESSAGES,CANDIDATE)
        self.reviewer.assert_not_awaited()

    async def test_selective_skips_normal_input_and_reviews_candidate(self):
        await self.supervisor(mode='selective').evaluate(MESSAGES)
        self.reviewer.assert_not_awaited()
        await self.supervisor(mode='selective').evaluate(MESSAGES,CANDIDATE)
        self.reviewer.assert_awaited_once()

    async def test_selective_reviews_blocked_input(self):
        self.input.return_value=InputVerdict(decision='block',category='jailbreak_injection')
        self.reviewer.return_value=report(review())
        await self.supervisor(mode='selective').evaluate(MESSAGES)
        self.input_reviewer.assert_awaited_once()
        self.reviewer.assert_not_awaited()

    async def test_failure_logs_do_not_include_private_body(self):
        secret='PRIVATE_REQUEST_AND_API_KEY_BODY'
        self.reviewer.side_effect=ValueError(secret)
        stream=io.StringIO()
        handler=logging.StreamHandler(stream)
        gs.logger.addHandler(handler)
        try: await self.supervisor().evaluate(MESSAGES,CANDIDATE)
        finally: gs.logger.removeHandler(handler)
        self.assertNotIn(secret,stream.getvalue())
        self.assertIn('ValueError',stream.getvalue())


class ReviewerContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_reviewer_rejects_output_target_without_candidate(self):
        create=AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(
            finish_reason='stop',message=SimpleNamespace(content=json.dumps({
                'reviews':[review(),review('grounding')]})))]))
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with self.assertRaisesRegex(ValueError,'missing_review_target'):
            await review_verdict(client,'test-model',MESSAGES,None,[],{'input':INITIAL['input']})
        self.assertLessEqual(create.call_args.kwargs['max_completion_tokens'],2000)


if __name__=='__main__': unittest.main()
