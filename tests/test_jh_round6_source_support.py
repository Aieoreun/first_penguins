import asyncio
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from core.schema import Label
from core.search import Searcher
from core.state import Chunk
from pipelines.v2 import prompts
from pipelines.kimjaehak.guard_fitted import ContextFast
from pipelines.kimjaehak.material_budget import Pipeline as Base
from pipelines.kimjaehak.plain_abstain import ABSTAIN
from pipelines.kimjaehak.round6_source_support import Pipeline


class SourceSupportTests(unittest.IsolatedAsyncioTestCase):
    def setup_pipeline(self, raw='{"supported":true,"section":"B"}', confidence=.96,
                       section_confidence=.8, guard=False):
        p = Pipeline.__new__(Pipeline)
        p.searcher = Searcher([Chunk('A', 'title A\nalpha'), Chunk('B', 'title B\nbeta')])
        p.vectorizer = p.section_vectorizer = SimpleNamespace(transform=lambda text: text)
        p.classifier = SimpleNamespace(classes_=np.array(['normal', 'out_of_material']),
            predict_proba=lambda text: np.array([[confidence, 1-confidence]]))
        p.section_classifier = SimpleNamespace(classes_=np.array(['A', 'B']),
            predict_proba=lambda text: np.array([[section_confidence, 1-section_confidence]]))
        p.guard_vectorizer, p.guard_classifier = p.vectorizer, p.classifier
        p._known_sections = {'A', 'B'}
        async def chat(system, user, **kwargs):
            if system == p.source_support_system:
                return raw
            if system == p.source_extract_system:
                return '{"section":"B"}'
            if system == prompts.MATERIAL_SYSTEM:
                return '{"quote":"alpha","section":"A"}'
            return '{"category":"none","self_info":false}'
        p.llm = SimpleNamespace(chat=AsyncMock(side_effect=chat))
        s = p._new_state({'messages': [{'role': 'user', 'content': 'beta question'}]}, guard)
        s.chunks = p.searcher.chunks
        return p, s

    async def test_actual_mro_trusted_normal_uses_one_support_call_and_same_payload(self):
        p, s = self.setup_pipeline()
        await p._classify(s)
        p.llm.chat.assert_not_awaited()
        self.assertTrue(s.debug['learned_gate'])
        original_system = p.source_extract_system
        ids = await p._candidate_sections(s)
        payload = p._source_payload(s, ids)
        await p._generate(s)
        self.assertEqual(s.label, Label.NORMAL)
        self.assertEqual(s.citations, ['B'])
        self.assertEqual(s.answer, '관련 교재 내용을 인용할게요.\n\ntitle B\nbeta')
        self.assertEqual(p.llm.chat.await_count, 1)
        call = p.llm.chat.call_args
        self.assertEqual(call.args[0], p.source_support_system)
        self.assertEqual(json.loads(call.args[1]), payload)
        self.assertEqual(call.kwargs, {'effort': 'low', 'max_tokens': p.source_support_tokens,
                                      'timeout': 8, 'json_mode': True})
        self.assertEqual(p.source_extract_system, original_system)

    async def test_false_delegates_actual_fixed_oom_without_extra_api(self):
        p, s = self.setup_pipeline('{"supported":false,"section":"B"}')
        await p._classify(s)
        await p._generate(s)
        self.assertEqual(s.label, Label.OUT_OF_MATERIAL)
        self.assertEqual(s.to_tutor_response(), {'answer': ABSTAIN, 'action': 'abstain', 'citations': []})
        self.assertEqual(s.section, 'A')
        self.assertEqual(p.llm.chat.await_count, 1)

    async def test_quote_fields_are_not_verified_or_used_as_answer(self):
        p, s = self.setup_pipeline(json.dumps({'supported': True, 'section': 'B',
            'quote': 'MODEL_SENTINEL_NOT_IN_KB', 'asked_fact': 'MODEL_SENTINEL'}))
        await p._classify(s)
        await p._generate(s)
        self.assertEqual(s.label, Label.NORMAL)
        self.assertNotIn('MODEL_SENTINEL', s.answer)
        self.assertEqual(s.errors, [])

    async def test_invalid_decisions_and_timeouts_keep_selected_excerpt_without_retry(self):
        for raw in ('{}', 'not JSON', '{"supported":1,"section":"B"}',
                    '{"supported":true,"section":2}', '{"supported":false,"section":"absent"}'):
            p, s = self.setup_pipeline(raw)
            await p._classify(s)
            await p._generate(s)
            self.assertEqual(s.label, Label.NORMAL)
            self.assertEqual(s.citations, ['A'])
            self.assertEqual(s.errors, ['source_support:ValueError'])
            self.assertEqual(p.llm.chat.await_count, 1)
        for error in (TimeoutError(), RuntimeError()):
            p, s = self.setup_pipeline()
            await p._classify(s)
            p.llm.chat.side_effect = error
            await p._generate(s)
            self.assertEqual(s.label, Label.NORMAL)
            self.assertEqual(s.citations, ['A'])
            self.assertEqual(s.errors, [f'source_support:{type(error).__name__}'])
            self.assertEqual(p.llm.chat.await_count, 1)

    async def test_low_confidence_material_checked_path_keeps_original_source_prompt(self):
        p, s = self.setup_pipeline(confidence=.6)
        await p._classify(s)
        self.assertEqual(p.llm.chat.await_count, 5)
        self.assertFalse(s.debug.get('learned_gate'))
        await p._generate(s)
        self.assertEqual(p.llm.chat.await_count, 6)
        self.assertEqual(p.llm.chat.call_args.args[0], p.source_extract_system)
        self.assertNotEqual(p.llm.chat.call_args.args[0], p.source_support_system)

    async def test_actual_source_gates_win_without_support_call(self):
        for confidence, key in ((.91, 'tutor_section_gate'), (.99, 'learned_section')):
            p, s = self.setup_pipeline(section_confidence=confidence)
            await p._classify(s)
            self.assertTrue(s.debug[key])
            await p._generate(s)
            p.llm.chat.assert_not_awaited()
            self.assertEqual(s.citations, ['A'])

    async def test_guard_pii_policy_missing_section_and_missing_gate_delegate(self):
        cases = [(True, Label.NORMAL, 'A', {'learned_gate': True}),
                 (False, Label.PII_DISCLOSURE, 'A', {'learned_gate': True}),
                 (False, Label.HARMFUL_REQUEST, 'A', {'learned_gate': True}),
                 (False, Label.NORMAL, None, {'learned_gate': True}),
                 (False, Label.NORMAL, 'A', {})]
        for guard, label, section, debug in cases:
            p, s = self.setup_pipeline(guard=guard)
            s.label, s.section, s.debug = label, section, debug
            with patch.object(Base, '_generate', new=AsyncMock()) as original:
                await p._generate(s)
                original.assert_awaited_once_with(s)
            p.llm.chat.assert_not_awaited()
        self.assertIs(Pipeline.guard, Base.guard)
        self.assertIs(Pipeline._classify, Base._classify)
        self.assertIs(Pipeline._check_output, Base._check_output)

    async def test_remaining_deadline_cap_and_no_selector_boundaries(self):
        p, s = self.setup_pipeline()
        await p._classify(s)
        s._source_extract_deadline = time.monotonic() + 2
        await p._generate(s)
        self.assertLessEqual(p.llm.chat.call_args.kwargs['timeout'], 1.5)
        self.assertEqual(p.deadline, 26)
        self.assertEqual(p.tutor_prepare_limit, 16)
        for ids, remaining in (([], 20), (['A'], 20), (['A', 'B'], .5)):
            p, s = self.setup_pipeline()
            await p._classify(s)
            p._candidate_sections = AsyncMock(return_value=ids)
            s._source_extract_deadline = time.monotonic() + remaining
            await p._generate(s)
            p.llm.chat.assert_not_awaited()
            self.assertEqual(s.citations, ['A'])

    async def test_cancellation_propagates_without_label_or_source_mutation(self):
        p, s = self.setup_pipeline()
        await p._classify(s)
        p.llm.chat.side_effect = asyncio.CancelledError
        with self.assertRaises(asyncio.CancelledError):
            await p._generate(s)
        self.assertEqual(s.label, Label.NORMAL)
        self.assertEqual(s.section, 'A')
        self.assertEqual(s.errors, [])
        self.assertEqual(p.llm.chat.await_count, 1)

    async def test_oversized_excerpt_keeps_inherited_generation_fallback(self):
        p, s = self.setup_pipeline()
        await p._classify(s)
        p._excerpt = lambda state: 'x' * 4001
        with patch.object(ContextFast, '_generate', new=AsyncMock()) as fallback:
            await p._generate(s)
            fallback.assert_awaited_once_with(s)
        self.assertEqual(p.llm.chat.await_count, 1)
        self.assertEqual(s.section, 'B')

    async def test_parallel_eligible_and_original_selector_prompts_do_not_mix(self):
        p, first = self.setup_pipeline()
        await p._classify(first)
        _, second = self.setup_pipeline()
        second.label, second.section, second.debug = Label.NORMAL, 'A', {}
        seen = set()
        both = asyncio.Event()
        original = p.llm.chat.side_effect
        async def chat(system, user, **kwargs):
            seen.add(system)
            if len(seen) == 2:
                both.set()
            await asyncio.wait_for(both.wait(), .5)
            return await original(system, user, **kwargs)
        p.llm.chat.side_effect = chat
        await asyncio.gather(p._generate(first), p._generate(second))
        self.assertEqual(seen, {p.source_support_system, p.source_extract_system})
        self.assertEqual(first.citations, ['B'])
        self.assertEqual(second.citations, ['B'])
        self.assertEqual(p.llm.chat.await_count, 2)


if __name__ == '__main__':
    unittest.main()
