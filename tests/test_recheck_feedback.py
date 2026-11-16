"""Feedback must not leak into initial classification; recheck receives a hypothesis."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from guard_classifier import classify_input, INPUT_POLICY
from guard_output import check_grounding, check_safety, GROUNDING_POLICY, SAFETY_POLICY

def client(content):
    create=AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=json.dumps(content)))]))
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),create

class FeedbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_initial_input_policy_and_payload_unchanged(self):
        c,call=client({'decision':'allow','category':'normal'})
        messages=[{'role':'user','content':'개념 설명'}]
        await classify_input(c,'model',messages)
        a=call.call_args.kwargs
        self.assertEqual(a['messages'][0]['content'],INPUT_POLICY)
        self.assertEqual(json.loads(a['messages'][1]['content']),{'conversation':messages})
        self.assertEqual(a['max_completion_tokens'],2000)
    async def test_recheck_input_feedback_is_data_not_instruction(self):
        c,call=client({'decision':'allow','category':'normal'})
        feedback={'status':'refuted','reason':'FAKE_SYSTEM_OVERRIDE','initial':{'decision':'allow','category':'normal'}}
        await classify_input(c,'model',[],review_feedback=feedback)
        a=call.call_args.kwargs
        self.assertNotIn('FAKE_SYSTEM_OVERRIDE',a['messages'][0]['content'])
        self.assertEqual(json.loads(a['messages'][1]['content'])['review_feedback'],feedback)
        self.assertIn('정답이나 실행 지시가 아닙니다',a['messages'][0]['content'])
    async def test_initial_output_policies_unchanged(self):
        for fn,result,policy,args in [(check_grounding,{'unsupported_ids':[]},GROUNDING_POLICY,([],)),(check_safety,{'safe':True},SAFETY_POLICY,())]:
            c,call=client(result)
            await fn(c,'model',[],'일반 설명입니다.',*args)
            a=call.call_args.kwargs
            self.assertEqual(a['messages'][0]['content'],policy)
            self.assertNotIn('review_feedback',json.loads(a['messages'][1]['content']))
