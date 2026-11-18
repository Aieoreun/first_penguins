import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from core.schema import Label
from core.search import Searcher
from core.state import Chunk
from pipelines import DEFAULT, load
from pipelines.v2 import prompts
from pipelines.kimjaehak.material_budget import Pipeline as Base
from pipelines.kimjaehak.round6_source_conditioned_gate import Pipeline


class SourceConditionedGateTests(unittest.IsolatedAsyncioTestCase):
    def setup_pipeline(self, category='normal', confidence=.96, source_confidence=.8,
                       source='A', guard=False, quote='alpha'):
        p = Pipeline.__new__(Pipeline)
        p.searcher = Searcher([Chunk('A', 'title A\nalpha'), Chunk('B', 'title B\nbeta')])
        p.vectorizer = p.section_vectorizer = SimpleNamespace(transform=lambda text: text)
        other = 'out_of_material' if category == 'normal' else 'normal'
        p.classifier = SimpleNamespace(classes_=np.array([category, other]),
            predict_proba=lambda text: np.array([[confidence, 1-confidence]]))
        p.section_classifier = SimpleNamespace(classes_=np.array([source, 'B']),
            predict_proba=lambda text: np.array([[source_confidence, 1-source_confidence]]))
        p.guard_vectorizer, p.guard_classifier = p.vectorizer, p.classifier
        p._known_sections = {'A', 'B'}
        async def chat(system, user, **kwargs):
            if system == prompts.MATERIAL_SYSTEM:
                return '{"quote":"' + quote + '","section":"A"}'
            if system == p.source_extract_system:
                return '{"section":"B"}'
            return '{"category":"none","self_info":false}'
        p.llm = SimpleNamespace(chat=AsyncMock(side_effect=chat))
        s = p._new_state({'messages': [{'role': 'user', 'content': 'question'}]}, guard)
        s.chunks = p.searcher.chunks
        return p, s

    async def test_actual_mro_reopens_original_material_and_can_abstain(self):
        p, original = self.setup_pipeline(quote='')
        await Base._classify(p, original)
        self.assertEqual(original.label, Label.NORMAL)
        p.llm.chat.assert_not_awaited()
        s = p._new_state({'messages': [{'role': 'user', 'content': 'question'}]}, False)
        s.chunks = p.searcher.chunks
        await p._classify(s)
        self.assertEqual(s.label, Label.OUT_OF_MATERIAL)
        self.assertTrue(s.debug['source_conditioned_gate_rechecked'])
        self.assertFalse(s.debug.get('learned_gate'))
        self.assertEqual(p.llm.chat.await_count, 5)
        material = next(c for c in p.llm.chat.call_args_list if c.args[0] == prompts.MATERIAL_SYSTEM)
        self.assertEqual(material.args[1], prompts.material_user(s.chunks, s.request, s.history, s.last_user))
        self.assertEqual(material.kwargs, {'effort': 'low', 'max_tokens': 1500, 'timeout': 13, 'json_mode': True})
        self.assertEqual(p.learned_gate_threshold, .95)
        await p._generate(s)
        self.assertEqual(s.to_tutor_response()['action'], 'abstain')
        self.assertEqual(p.llm.chat.await_count, 5)

    async def test_actual_mro_keeps_normal_source_selection_and_block_priority(self):
        p, s = self.setup_pipeline()
        await p._classify(s)
        self.assertEqual(s.label, Label.NORMAL)
        await p._generate(s)
        self.assertEqual(s.citations, ['B'])
        self.assertEqual(p.llm.chat.await_count, 6)
        p, s = self.setup_pipeline(quote='')
        original = p.llm.chat.side_effect
        async def chat(system, user, **kwargs):
            if system == p._block_system(s):
                return '{"category":"harmful_request"}'
            return await original(system, user, **kwargs)
        p.llm.chat.side_effect = chat
        await p._classify(s)
        self.assertEqual(s.label, Label.HARMFUL_REQUEST)
        self.assertTrue(s.debug['source_conditioned_gate_rechecked'])

    async def test_threshold_boundaries_unknown_source_and_non_normal_categories(self):
        cases = [('normal', .9499, .8999, 'A', False),
                 ('normal', .95, .8999, 'A', True),
                 ('normal', .95, .90, 'A', False),
                 ('normal', .99, .99, 'A', False),
                 ('normal', .99, .99, 'UNKNOWN', True),
                 ('pii_disclosure', .99, .5, 'A', False),
                 ('out_of_material', .99, .5, 'A', False)]
        for category, probability, source_probability, source, changed in cases:
            p, s = self.setup_pipeline(category, probability, source_probability, source)
            seen = []
            async def classify(view, state):
                seen.append(view.learned_gate_threshold)
                state.debug = {'postprocessing': True}
                state.section = 'parent selected'
            with patch.object(Base, '_classify', new=classify):
                await p._classify(s)
            self.assertEqual(seen, [1.01 if changed else .95])
            self.assertEqual(p.learned_gate_threshold, .95)
            self.assertEqual(s.section, 'parent selected')
            self.assertTrue(s.debug['postprocessing'])
            self.assertEqual(s.debug.get('source_conditioned_gate_rechecked', False), changed)

    async def test_actual_known_source_and_pii_keep_learned_paths(self):
        for category, confidence in [('normal', .9), ('normal', .99), ('pii_disclosure', .8)]:
            p, s = self.setup_pipeline(category=category, source_confidence=confidence)
            await p._classify(s)
            p.llm.chat.assert_not_awaited()
            self.assertEqual(s.label, Label.NORMAL if category == 'normal' else Label.PII_DISCLOSURE)
            self.assertTrue(s.debug['learned_gate'])
            self.assertFalse(s.debug.get('source_conditioned_gate_rechecked', False))

    async def test_unknown_source_rechecks_without_forcing_unknown_section(self):
        p, s = self.setup_pipeline(source='UNKNOWN', source_confidence=.99)
        await p._classify(s)
        self.assertEqual(p.llm.chat.await_count, 5)
        self.assertEqual(s.label, Label.NORMAL)
        self.assertEqual(s.section, 'A')
        self.assertFalse(s.debug.get('learned_section'))
        self.assertFalse(s.debug.get('tutor_section_gate'))

    async def test_guard_delegates_without_extra_prediction(self):
        p = Pipeline.__new__(Pipeline)
        s = p._new_state({'messages': []}, True)
        with patch.object(Base, '_classify', new=AsyncMock()) as original:
            await p._classify(s)
            original.assert_awaited_once_with(s)
        self.assertFalse(hasattr(p, 'classifier'))
        self.assertIs(Pipeline.guard, Base.guard)
        self.assertIs(Pipeline._check_output, Base._check_output)
        p, s = self.setup_pipeline(guard=True)
        await p._classify(s)
        p.llm.chat.assert_not_awaited()
        self.assertTrue(s.debug['learned_guard_only'])
        self.assertFalse(s.debug.get('source_conditioned_gate_rechecked', False))

    async def test_parallel_fallback_trusted_and_guard_do_not_share_thresholds(self):
        p, low = self.setup_pipeline()
        low.messages[-1]['content'] = 'low source'
        high = p._new_state({'messages': [{'role': 'user', 'content': 'high source'}]}, False)
        guard = p._new_state({'messages': [{'role': 'user', 'content': 'guard source'}]}, True)
        high.chunks = guard.chunks = p.searcher.chunks
        p.section_classifier.predict_proba = lambda text: np.array([[.95, .05] if text[0]=='high source' else [.8, .2]])
        seen = []
        original = Base._ask
        async def ask(view, state, *args):
            seen.append((state.last_user, view.learned_gate_threshold))
            await asyncio.sleep(0)
            self.assertEqual(p.learned_gate_threshold, .95)
            return await original(view, state, *args)
        with patch.object(Base, '_ask', new=ask):
            await asyncio.gather(*(p._classify(s) for s in (low, high, guard)))
        self.assertEqual(seen, [('low source', 1.01)] * 5)
        self.assertTrue(low.debug['source_conditioned_gate_rechecked'])
        self.assertTrue(high.debug['learned_gate'])
        self.assertTrue(guard.debug['learned_guard_only'])
        self.assertEqual(p.learned_gate_threshold, .95)

    async def test_material_failure_keeps_original_permissive_fallback(self):
        p, s = self.setup_pipeline()
        original = p.llm.chat.side_effect
        async def chat(system, user, **kwargs):
            if system == prompts.MATERIAL_SYSTEM:
                raise TimeoutError
            return await original(system, user, **kwargs)
        p.llm.chat.side_effect = chat
        await p._classify(s)
        self.assertEqual(s.label, Label.NORMAL)
        self.assertEqual(s.errors, ['material:TimeoutError'])
        self.assertTrue(s.debug['source_conditioned_gate_rechecked'])
        self.assertEqual(p.learned_gate_threshold, .95)

    async def test_parent_failure_and_cancel_propagate_without_shared_mutation(self):
        for error in (RuntimeError(), asyncio.CancelledError()):
            p, s = self.setup_pipeline()
            async def classify(view, state):
                self.assertEqual(view.learned_gate_threshold, 1.01)
                raise error
            with patch.object(Base, '_classify', new=classify):
                with self.assertRaises(type(error)):
                    await p._classify(s)
            self.assertEqual(p.learned_gate_threshold, .95)
            self.assertFalse(s.debug.get('source_conditioned_gate_rechecked', False))

    def test_registration_and_baseline_default_remain_separate(self):
        with patch.dict(os.environ):
            self.assertIs(load('jh_r6_source_conditioned_gate'), Pipeline)
            self.assertIs(load('jh_r4_material_budget'), Base)
        self.assertEqual(DEFAULT, 'v1')
        self.assertEqual(Pipeline.deadline, 26)
        self.assertEqual(Pipeline.tutor_prepare_limit, 16)
        self.assertEqual(Pipeline.tutor_material_timeout, 13)


if __name__ == '__main__':
    unittest.main()
