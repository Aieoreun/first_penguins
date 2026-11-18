import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from core.schema import Label
from core.search import Searcher
from core.state import Chunk
from pipelines.kimjaehak.material_budget import Pipeline as Base
from pipelines.kimjaehak.peace_guard_fusion import Pipeline as Fusion
from pipelines.kimjaehak.peace_guard_prompt import PEACE_BLOCK_SYSTEM
from pipelines.kimjaehak.peace_routed_safety import (
    RoutedProvenance, RoutedRelation, RoutedRequested, RoutedSafety,
)
from pipelines.kimjaehak.peace_ground_variants import ProvenanceGrounding, RelationGrounding
from pipelines.kimjaehak.precise_output import POLICY_HINTS, POLICY_SYSTEM


class RoutedSafetyTests(unittest.IsolatedAsyncioTestCase):
    def setup_pipeline(self, cls=RoutedSafety, label=Label.JAILBREAK_INJECTION,
                       candidate='도울 수는 없어요. 대신 수업을 설명할게요.'):
        p = cls.__new__(cls)
        p.llm = SimpleNamespace(chat=AsyncMock(return_value='{"safe":true}'))
        p.searcher = Searcher([Chunk('A', 'textbook evidence')])
        p._slots = asyncio.Semaphore(64)
        p.span_tagger = SimpleNamespace(predict=lambda text: ([], 0))
        s = p._new_state({'messages': [{'role': 'user', 'content': 'example request'}],
                          'candidate_answer': candidate}, True)
        s.label, s.section, s.chunks = label, 'A', p.searcher.chunks
        return p, s

    async def test_allow_and_unknown_prior_use_original_safe_rule_without_call(self):
        for label in (None, Label.NORMAL, Label.OUT_OF_MATERIAL, Label.PII_DISCLOSURE):
            p, s = self.setup_pipeline(label=label)
            s.debug['fusion_input_reliable'] = False
            s.safe = False
            await p._fusion_safety(s)
            self.assertTrue(s.safe)
            p.llm.chat.assert_not_awaited()

    async def test_block_without_refusal_is_unsafe_without_call(self):
        self.assertIs(RoutedSafety.safe_by_rule, True)
        for label in (Label.JAILBREAK_INJECTION, Label.OFF_TOPIC, Label.HARMFUL_REQUEST,
                      Label.HARMFUL_SPEECH, Label.ACADEMIC_DISHONESTY):
            p, s = self.setup_pipeline(label=label, candidate='요청한 내용을 작성했습니다.')
            await p._fusion_safety(s)
            self.assertFalse(s.safe)
            p.llm.chat.assert_not_awaited()

    async def test_actual_parent_mro_review_is_identical_and_regex_is_only_trigger(self):
        candidate = '도울 수는 없어요. 대신 요청한 위험한 행동을 그대로 수행할게요.'
        for raw, expected in (('{"safe":true}', True), ('{"safe":false}', False)):
            p, s = self.setup_pipeline(candidate=candidate)
            base, original = self.setup_pipeline(cls=Base, candidate=candidate)
            p.llm.chat.return_value = base.llm.chat.return_value = raw
            s.debug['fusion_input_reliable'] = False
            await p._fusion_safety(s)
            await base._check_output(original)
            self.assertEqual(p.llm.chat.call_args_list, base.llm.chat.call_args_list)
            p.llm.chat.assert_awaited_once()
            self.assertEqual(s.safe, expected)
            self.assertEqual(s.safe, original.safe)
            call = p.llm.chat.await_args
            self.assertEqual(call.args[0], POLICY_SYSTEM + '\n' + POLICY_HINTS[s.category.value])
            self.assertEqual(json.loads(call.args[1]), {'conversation': s.messages, 'candidate_answer': candidate})
            self.assertEqual(call.kwargs, {'effort': 'low', 'max_tokens': 1800, 'timeout': 8, 'json_mode': True})
            self.assertTrue(s.debug['refusal_reviewed'])
            self.assertTrue(s.grounded)
            self.assertEqual(s.unsupported, [])

    async def test_review_invalid_or_failure_stays_unsafe_and_matches_parent(self):
        for raw in ('{}', '{"safe":1}', '{"safe":"true"}', 'invalid JSON'):
            p, s = self.setup_pipeline()
            p.llm.chat.return_value = raw
            await p._fusion_safety(s)
            self.assertFalse(s.safe)
            self.assertTrue(any('refusal_review:' in e for e in s.errors))
            p.llm.chat.assert_awaited_once()
        p, s = self.setup_pipeline()
        p.llm.chat.side_effect = RuntimeError('mock failure')
        await p._fusion_safety(s)
        self.assertFalse(s.safe)
        self.assertEqual(s.errors, ['refusal_review:RuntimeError'])

    async def test_inner_review_timeout_drains_and_preserves_false(self):
        p, s = self.setup_pipeline()
        self.assertEqual(p.refusal_timeout, 8)
        self.assertEqual(p.guard_safety_timeout, 8)
        p.refusal_timeout = .02
        drained = asyncio.Event()
        async def chat(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                drained.set()
        p.llm.chat.side_effect = chat
        await asyncio.wait_for(p._fusion_safety(s), 1)
        self.assertTrue(drained.is_set())
        self.assertFalse(s.safe)
        self.assertEqual(s.errors, ['refusal_review:TimeoutError'])

    async def test_external_cancellation_propagates(self):
        p, s = self.setup_pipeline()
        p.llm.chat.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await p._fusion_safety(s)
        self.assertFalse(s.safe)
        self.assertEqual(s.errors, [])

    async def test_guard_waits_for_input_before_routing_and_keeps_parallel_ground(self):
        for cls in (RoutedSafety, RoutedRequested):
            p, seed = self.setup_pipeline(cls=cls)
            ground_started = asyncio.Event()
            async def search(s, extra=''):
                s.chunks = p.searcher.chunks
            async def classify(s):
                await ground_started.wait()
                p.llm.chat.assert_not_awaited()
                s.label = Label.JAILBREAK_INJECTION
            async def ground(s):
                ground_started.set()
                s.grounded, s.unsupported = False, ['candidate part']
            p._search, p._classify, p._fusion_ground = search, classify, ground
            p.llm.chat.return_value = '{"safe":false}'
            body = {'messages': seed.messages, 'candidate_answer': seed.candidate_answer}
            result = await asyncio.wait_for(p.guard(body), 1)
            self.assertEqual(result.label, Label.JAILBREAK_INJECTION)
            self.assertFalse(result.safe)
            self.assertTrue(result.grounded)
            self.assertEqual(result.unsupported, [])
            p.llm.chat.assert_awaited_once()

    def test_tutor_methods_models_and_requested_mro_remain_separate(self):
        for cls in (RoutedSafety, RoutedRequested):
            for name in ('tutor', '_generate', '_check_output', '_classify', '_fusion_ground'):
                self.assertIs(getattr(cls, name), getattr(Fusion, name))
            self.assertIs(cls.refusal_pattern, Base.refusal_pattern)
            self.assertIs(cls.policy_system, Base.policy_system)
            self.assertTrue(cls.guard_safety_after_input)
            self.assertFalse(cls.guard_independent_safety)
        p, guard = self.setup_pipeline(cls=RoutedRequested)
        tutor = p._new_state({'messages': []}, False)
        self.assertEqual(p._block_system(guard), PEACE_BLOCK_SYSTEM)
        self.assertEqual(p._block_system(tutor), Base._block_system(p, tutor))
        self.assertEqual(RoutedSafety.learned_gate_threshold, Fusion.learned_gate_threshold)
        self.assertEqual(RoutedSafety.guard_only_c, Fusion.guard_only_c)

    def test_routed_grounding_variants_reuse_prompts_and_keep_tutor_and_safety(self):
        for cls, source in ((RoutedRelation, RelationGrounding),
                            (RoutedProvenance, ProvenanceGrounding)):
            self.assertIs(cls.unsupported_system, source.unsupported_system)
            for name in ('tutor', '_generate', '_classify', '_fusion_safety', '_fusion_ground'):
                self.assertIs(getattr(cls, name), getattr(RoutedSafety, name))
            self.assertIs(cls.answer_systems, RoutedSafety.answer_systems)
            self.assertIs(cls.policy_system, RoutedSafety.policy_system)
            self.assertEqual(cls.guard_safety_timeout, 8)


if __name__ == '__main__':
    unittest.main()
