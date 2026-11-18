import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from core.schema import Label
from core.search import Searcher
from core.state import Chunk
from pipelines.kimjaehak.adaptive import Pipeline as Fallback
from pipelines.kimjaehak.material_budget import Pipeline as Base
from pipelines.kimjaehak.round6_normal_gate import Pipeline as Normal94
from pipelines.kimjaehak.round6_normal_sparse import Pipeline


def setup(cls, category='normal', confidence=.96, section_confidence=.92):
    pipeline = cls.__new__(cls)
    pipeline.searcher = Searcher(
        [Chunk('A', 'title A\nalpha'), Chunk('B', 'title B\nbeta')],
        [[1., 0.], [0., 1.]],
    )

    async def embed(*args, **kwargs):
        await asyncio.sleep(0)  # Interleave ordinary and skipped searches.
        return [[1., 0.]]

    pipeline.llm = SimpleNamespace(
        can_embed=True, embed=AsyncMock(side_effect=embed),
        chat=AsyncMock(side_effect=AssertionError('Unexpected chat')),
    )
    pipeline.vectorizer = pipeline.section_vectorizer = SimpleNamespace(
        transform=lambda text: text)
    pipeline.classifier = SimpleNamespace(
        classes_=np.array([category, 'off_topic']),
        predict_proba=lambda text: np.array([[confidence, 1 - confidence]]),
    )
    pipeline.section_classifier = SimpleNamespace(
        classes_=np.array(['B', 'A']),
        predict_proba=lambda text: np.array([[section_confidence, 1 - section_confidence]]),
    )
    pipeline.guard_vectorizer = pipeline.vectorizer
    pipeline.guard_classifier = pipeline.classifier
    pipeline._known_sections = {'A', 'B'}
    return pipeline


def state_for(pipeline, guard=False, text='alpha question'):
    return pipeline._new_state(
        {'messages': [{'role': 'user', 'content': text}]}, guard)


async def tutor_steps(pipeline, state):
    await pipeline._search(state)
    await pipeline._classify(state)
    await pipeline._generate(state)
    return state.to_tutor_response()


class NormalSparseTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_output_matches_normal94_and_embedding_boundary_stays_95(self):
        for category, confidence in (
            ('normal', .94), ('normal', .949999), ('normal', .95),
            ('normal', .99), ('pii_disclosure', .95),
        ):
            for section_confidence in (.90, .99):
                with self.subTest(category=category, confidence=confidence,
                                  section_confidence=section_confidence):
                    baseline = setup(Normal94, category, confidence, section_confidence)
                    combined = setup(Pipeline, category, confidence, section_confidence)
                    baseline_state, combined_state = state_for(baseline), state_for(combined)
                    expected = await tutor_steps(baseline, baseline_state)
                    actual = await tutor_steps(combined, combined_state)
                    self.assertEqual(actual, expected)
                    self.assertEqual(actual['citations'], ['B'])
                    self.assertTrue(actual['answer'])
                    self.assertEqual(combined_state.category.value, category)
                    self.assertTrue(combined_state.debug['learned_gate'])
                    self.assertEqual(
                        'tutor_normal_gate_threshold' in combined_state.debug, confidence < .95)
                    baseline.llm.embed.assert_awaited_once()
                    self.assertEqual(combined.llm.embed.await_count, int(confidence < .95))
                    baseline.llm.chat.assert_not_awaited()
                    combined.llm.chat.assert_not_awaited()
                    self.assertEqual(combined.learned_gate_threshold, .95)

    async def test_classification_relaxes_only_normal_in_new_band(self):
        for category, confidence in (
            ('normal', .939999), ('normal', .94),
            ('pii_disclosure', .945), ('emotional_distress', .945),
        ):
            with self.subTest(category=category, confidence=confidence):
                pipeline = setup(Pipeline, category, confidence)
                state = state_for(pipeline)
                fallback_calls = []

                async def fallback(view, current):
                    fallback_calls.append(view.learned_gate_threshold)
                    current.label = Label.NORMAL

                with patch.object(Fallback, '_classify', new=fallback):
                    await pipeline._search(state)
                    await pipeline._classify(state)
                self.assertEqual(fallback_calls, [] if category == 'normal' and confidence >= .94 else [.95])
                pipeline.llm.embed.assert_awaited_once()
                self.assertEqual(pipeline.learned_gate_threshold, .95)
                self.assertEqual(state.section, 'B')  # Existing section postprocessing survives.

    async def test_sparse_requires_original_source_and_input_conditions(self):
        for change in ('section', 'unknown', 'require_retrieved', 'category', 'guard'):
            with self.subTest(change=change):
                pipeline = setup(
                    Pipeline, category='emotional_distress' if change == 'category' else 'normal',
                    section_confidence=.8999 if change == 'section' else .92)
                if change == 'unknown':
                    pipeline._known_sections = {'A'}
                if change == 'require_retrieved':
                    pipeline.learned_section_require_retrieved = True
                state = state_for(pipeline, guard=change == 'guard')
                await pipeline._search(state)
                pipeline.llm.embed.assert_awaited_once()

    async def test_guard_real_classification_matches_base_and_guard_methods_are_inherited(self):
        for category in ('normal', 'pii_disclosure', 'emotional_distress'):
            baseline, combined = setup(Base, category), setup(Pipeline, category)
            states = []
            for pipeline in (baseline, combined):
                state = state_for(pipeline, guard=True)
                await pipeline._search(state)
                await pipeline._classify(state)
                states.append(state)
                pipeline.llm.embed.assert_awaited_once()
                pipeline.llm.chat.assert_not_awaited()
            self.assertEqual(states[0].to_guard_response(), states[1].to_guard_response())
            self.assertEqual(states[0].debug, states[1].debug)
            self.assertTrue(states[1].debug['learned_guard_only'])
        self.assertIs(Pipeline.guard, Base.guard)
        self.assertIs(Pipeline._check_output, Base._check_output)
        self.assertIs(Pipeline._generate, Base._generate)

    async def test_concurrent_requests_preserve_shared_threshold_models_and_client(self):
        pipeline = setup(Pipeline)
        pipeline.classifier.predict_proba = lambda texts: np.array([
            [.945, .055] if 'borderline' in text else [.96, .04] for text in texts])
        original = pipeline.__dict__.copy()
        low, high, guard = (state_for(pipeline, text='alpha borderline'),
                            state_for(pipeline), state_for(pipeline, guard=True))

        async def guard_steps():
            await pipeline._search(guard)
            await pipeline._classify(guard)

        await asyncio.gather(tutor_steps(pipeline, low), tutor_steps(pipeline, high), guard_steps())
        self.assertEqual(pipeline.llm.embed.await_count, 2)
        pipeline.llm.chat.assert_not_awaited()
        self.assertEqual(low.to_tutor_response(), high.to_tutor_response())
        self.assertEqual(low.debug['tutor_normal_gate_threshold'], .94)
        self.assertNotIn('tutor_normal_gate_threshold', high.debug)
        self.assertNotIn('tutor_normal_gate_threshold', guard.debug)
        self.assertTrue(guard.debug['learned_guard_only'])
        self.assertEqual(pipeline.learned_gate_threshold, .95)
        self.assertTrue(pipeline.llm.can_embed)
        self.assertTrue(pipeline.searcher.has_embeddings)
        self.assertEqual(pipeline.__dict__.keys(), original.keys())
        for name, value in original.items():
            self.assertIs(pipeline.__dict__[name], value, name)


if __name__ == '__main__':
    unittest.main()
