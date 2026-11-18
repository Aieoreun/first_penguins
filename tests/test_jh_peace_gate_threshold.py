"""Offline checks for endpoint-local thresholds through the real classifier MRO."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from core.schema import Label, parse_label
from core.state import Chunk
from pipelines.kimjaehak.adaptive import Pipeline as Adaptive
from pipelines.kimjaehak.guard_fitted import GuardFitting
from pipelines.kimjaehak.peace_gate_threshold import CurrentGate90, FastGate90
from pipelines.kimjaehak.peace_guard_scheduling import FastParallelInput
from pipelines.kimjaehak.peace_span_variants import CurrentSpan80


VARIANTS = ((CurrentGate90, CurrentSpan80), (FastGate90, FastParallelInput))


class GateThresholdTests(unittest.IsolatedAsyncioTestCase):
    def model(self, category, confidence):
        other = 'normal' if category != 'normal' else 'off_topic'
        return SimpleNamespace(classes_=np.array([category, other]),
            predict_proba=Mock(return_value=np.array([[confidence, 1 - confidence]])))

    def pipeline(self, cls, primary=('normal', .6), secondary=('normal', .6)):
        p = cls.__new__(cls)
        for name in ('guard_vectorizer', 'vectorizer', 'harmful_vectorizer', 'section_vectorizer'):
            setattr(p, name, Mock(transform=Mock(side_effect=lambda texts: texts)))
        p.guard_classifier = self.model(*primary)
        p.classifier = self.model(*secondary)
        p.harmful_classifier = self.model('refuse', .99)
        p.section_classifier = self.model('S', .6)
        p._known_sections = {'S'}
        p.llm = SimpleNamespace(chat=AsyncMock())
        return p

    def state(self, p, guard=True, candidate=None):
        body = {'messages': [{'role': 'user', 'content': '교재 내용을 설명해 주세요.'}]}
        if candidate is not None:
            body['candidate_answer'] = candidate
        state = p._new_state(body, guard)
        state.chunks = [Chunk('S', '교재 본문')]
        return state

    @staticmethod
    async def fallback(pipeline, state):
        state.label = Label.OFF_TOPIC
        state.debug = {'mock_fallback': True}

    async def test_primary_gate_keeps_precedence_with_or_without_candidate(self):
        for cls, _ in VARIANTS:
            for candidate in (None, '후보 답변'):
                p = self.pipeline(cls, ('off_topic', .90), ('normal', .99))
                s = self.state(p, candidate=candidate)
                await p._classify(s)
                self.assertEqual(s.label, Label.OFF_TOPIC)
                self.assertEqual(s.section, 'S')
                self.assertTrue(s.debug['learned_guard_only'])
                p.classifier.predict_proba.assert_not_called()
                p.llm.chat.assert_not_awaited()
                self.assertEqual(p.learned_gate_threshold, .95)

    async def test_secondary_gate_uses_same_threshold_after_primary_abstains(self):
        for cls, _ in VARIANTS:
            p = self.pipeline(cls, ('normal', .89), ('pii_disclosure', .90))
            s = self.state(p)
            await p._classify(s)
            self.assertEqual(s.label, Label.PII_DISCLOSURE)
            self.assertTrue(s.debug['learned_gate'])
            p.guard_classifier.predict_proba.assert_called_once()
            self.assertEqual(s.section, 'S')
            self.assertNotIn('tutor_section_gate', s.debug)
            p.llm.chat.assert_not_awaited()

    async def test_threshold_requires_explicit_true_marker_not_debug_or_candidate(self):
        for cls, _ in VARIANTS:
            for marker in (None, False, 1, 'true'):
                p = self.pipeline(cls, ('off_topic', .92), ('normal', .92))
                s = self.state(p, candidate='후보 답변')
                if marker is None:
                    del s._material_budget_guard
                else:
                    s._material_budget_guard = marker
                with patch.object(Adaptive, '_classify', new=self.fallback):
                    await p._classify(s)
                self.assertTrue(s.debug['mock_fallback'])
            # A real guard marker still applies after an earlier debug reset.
            p = self.pipeline(cls, secondary=('off_topic', .92))
            s = self.state(p)
            s.debug = {}
            await p._classify(s)
            self.assertTrue(s.debug['learned_gate'])
            self.assertEqual(s.label, Label.OFF_TOPIC)

    async def test_tutor_and_below_threshold_guard_keep_original_fallback(self):
        for cls, _ in VARIANTS:
            for guard, confidence in ((False, .92), (True, .8999)):
                p = self.pipeline(cls, ('off_topic', confidence), ('normal', confidence))
                s = self.state(p, guard=guard, candidate='candidate does not select endpoint')
                with patch.object(Adaptive, '_classify', new=self.fallback):
                    await p._classify(s)
                self.assertTrue(s.debug['mock_fallback'])
                if not guard:
                    p.guard_vectorizer.transform.assert_not_called()
                self.assertEqual(p.learned_gate_threshold, .95)

    async def test_existing_category_to_label_mapping_is_preserved(self):
        categories = ('normal', 'pii_disclosure', 'answer_seeking', 'academic_dishonesty',
                      'off_topic', 'out_of_material', 'jailbreak_injection',
                      'emotional_distress', 'low_signal_input')
        for cls, _ in VARIANTS:
            for category in categories:
                p = self.pipeline(cls, (category, .91))
                s = self.state(p)
                await p._classify(s)
                self.assertEqual(s.label, parse_label(category))
                self.assertEqual(s.category.value, category)
                p.llm.chat.assert_not_awaited()

    async def test_harmful_category_still_requires_unchanged_subtype_threshold(self):
        for cls, _ in VARIANTS:
            for action, confidence, expected in (
                    ('refuse', .94, Label.OFF_TOPIC),
                    ('refuse', .95, Label.HARMFUL_REQUEST),
                    ('redirect', .95, Label.HARMFUL_SPEECH)):
                p = self.pipeline(cls, ('harmful_content', .99), ('harmful_content', .92))
                p.harmful_classifier = self.model(action, confidence)
                s = self.state(p)
                with patch.object(Adaptive, '_classify', new=self.fallback):
                    await p._classify(s)
                self.assertEqual(s.label, expected)
                self.assertEqual(p.learned_harmful_threshold, .95)
                p.harmful_classifier.predict_proba.assert_called_once()
                self.assertNotIn('learned_guard_only', s.debug)

    async def test_concurrent_guard_tutor_share_models_without_threshold_mutation(self):
        shared_names = ('guard_vectorizer', 'guard_classifier', 'vectorizer', 'classifier',
                        'section_vectorizer', 'section_classifier',
                        'harmful_vectorizer', 'harmful_classifier', 'llm')
        for cls, _ in VARIANTS:
            p = self.pipeline(cls)
            states = [self.state(p, guard=True), self.state(p, guard=False)]
            entered, release = asyncio.Event(), asyncio.Event()
            seen = []

            async def classify(instance, state):
                seen.append((instance, state._material_budget_guard))
                state.label = Label.OFF_TOPIC
                if len(seen) == 2:
                    entered.set()
                await release.wait()
                self.assertEqual(instance.learned_gate_threshold,
                                 .90 if state._material_budget_guard else .95)
                self.assertEqual(p.learned_gate_threshold, .95)

            with patch.object(GuardFitting, '_classify', new=classify):
                tasks = [asyncio.create_task(p._classify(s)) for s in states]
                try:
                    await asyncio.wait_for(entered.wait(), 1)
                    self.assertNotIn('learned_gate_threshold', vars(p))
                    release.set()
                    await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
            for instance, guard in seen:
                self.assertEqual(instance is p, not guard)
                for name in shared_names:
                    self.assertIs(getattr(instance, name), getattr(p, name))

    def test_training_paths_scheduling_and_other_settings_are_inherited(self):
        for cls, parent in VARIANTS:
            self.assertEqual(cls.guard_gate_threshold, .90)
            for method in ('__init__', '_new_state', 'guard', 'tutor', '_run', '_search',
                           '_generate', '_check_output', '_prepare_within', '_collect_checks'):
                self.assertIs(getattr(cls, method), getattr(parent, method))
            for setting in ('learned_gate_threshold', 'learned_gate_c', 'guard_only_c',
                            'learned_harmful_threshold', 'learned_span_threshold',
                            'deadline', 'prepare_limit', 'request_slots', 'output_votes'):
                self.assertEqual(getattr(cls, setting), getattr(parent, setting))


if __name__ == '__main__':
    unittest.main()
