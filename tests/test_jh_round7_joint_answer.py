import asyncio
import json
import unittest
from unittest.mock import patch
from pipelines import load
load('jh_cm_char_mlp50')
from pipelines.kimjaehak.classifier_matrix import CharMLP50
from pipelines.kimjaehak.round7_answer import ConcreteApply, JointConcreteApply
from tests import test_jh_round7_answer as answer_tests


class JointTests(unittest.IsolatedAsyncioTestCase):
    pipeline = answer_tests.AnswerTests.pipeline
    state = answer_tests.AnswerTests.state
    ready = answer_tests.AnswerTests.ready

    def numeric(self):
        p,s=self.ready(JointConcreteApply)
        s.messages[-1]['content']='180,000원을 연 4%로 3개월 맡기면 이자는 얼마인가요?'
        return p,s

    async def test_joint_selection_and_explanation_use_one_call_and_original_excerpt(self):
        p,s=self.numeric();old=s.answer
        p.llm.chat.return_value=json.dumps({'section':'A','supported':True,'answer':'이자는 1,800원입니다.'})
        async def generate(instance,state):
            raw=await instance._source_response(state,['A','B'],6)
            state.section=instance._validate_source(json.loads(raw),['A','B'])
        with patch.object(CharMLP50,'_generate',new=generate):await p._generate(s)
        self.assertEqual(s.answer,'이자는 1,800원입니다.\n\n'+old)
        self.assertEqual(s.citations,['A'])
        p.llm.chat.assert_awaited_once()
        self.assertEqual(p.llm.chat.await_args.kwargs,
            dict(effort='low',max_tokens=2000,timeout=6,json_mode=True))
        self.assertTrue(s.debug['r7_joint_apply_completed'])
        self.assertFalse(hasattr(s,'_r7_joint_explanation'))

    async def test_bypassed_selector_uses_one_bounded_direct_call(self):
        p,s=self.numeric()
        p.llm.chat.return_value='{"supported":true,"answer":"이자는 1,800원입니다."}'
        with patch.object(CharMLP50,'_generate',return_value=None):await p._generate(s)
        p.llm.chat.assert_awaited_once()
        self.assertTrue(s.debug['r7_apply_completed'])
        self.assertNotIn('r7_joint_apply_started',s.debug)

    async def test_wrong_source_invalid_answer_and_failed_joint_call_never_retry(self):
        for mode in ('wrong_source','invalid','unsupported','failed'):
            p,s=self.numeric();old=s.answer
            async def generate(instance,state):
                state._r7_joint_attempted=True
                if mode!='failed':
                    state._r7_joint_explanation={'section':'B' if mode=='wrong_source' else 'A',
                        'supported':mode!='unsupported','answer':'x'*241 if mode=='invalid' else ''}
            with patch.object(CharMLP50,'_generate',new=generate):await p._generate(s)
            p.llm.chat.assert_not_awaited()
            self.assertEqual(s.answer,old)
            self.assertEqual(s.citations,['A'])
            self.assertFalse(hasattr(s,'_r7_joint_explanation'))

    async def test_nonnumeric_selector_and_guard_retain_original_call(self):
        for guard,text in ((True,'18,000원의 이자'),(False,'이자란 무엇인가요?')):
            p,s=self.ready(JointConcreteApply)
            s._material_budget_guard=guard;s.messages[-1]['content']=text
            await p._source_response(s,['A','B'],6)
            joint=p.llm.chat.await_args;p.llm.chat.reset_mock()
            await CharMLP50._source_response(p,s,['A','B'],6)
            self.assertEqual(joint,p.llm.chat.await_args)
            self.assertFalse(getattr(s,'_r7_joint_attempted',False))

    async def test_cancellation_propagates_and_concrete_requires_given_numbers(self):
        p,s=self.numeric();p.llm.chat.side_effect=asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):await p._source_response(s,['A','B'],6)
        p,s=self.ready(ConcreteApply)
        self.assertFalse(p._apply_eligible(s))
        s.messages[-1]['content']='25,000원씩 12개월 납입한 이자는 얼마예요?'
        self.assertTrue(p._apply_eligible(s))
        for name,cls in [('jh_r7_apply_concrete',ConcreteApply),('jh_r7_apply_joint',JointConcreteApply)]:
            self.assertIs(load(name),cls)
            self.assertIs(cls.guard,CharMLP50.guard)
            self.assertIs(cls._classify,CharMLP50._classify)


if __name__=='__main__':unittest.main()
