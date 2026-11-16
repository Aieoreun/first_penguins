import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from core.schema import Label
from core.state import State
from pipelines.kimjaehak.tutor_section_gate import ContextFast, Pipeline


class TutorSectionGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_guard_and_blocked_tutor_do_not_use_extra_section_gate(self):
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.section_classifier = SimpleNamespace(predict_proba=Mock())
        async def parent(p, state):
            state.section = 'old'
        with patch.object(ContextFast, '_classify', parent):
            for guard, label in [(True,Label.NORMAL),(False,Label.HARMFUL_REQUEST),(False,Label.OUT_OF_MATERIAL)]:
                state = State(messages=[], label=label)
                state._section_gate_guard = guard
                await pipeline._classify(state)
                self.assertEqual(state.section, 'old')
        pipeline.section_classifier.predict_proba.assert_not_called()

    async def test_tutor_gate_changes_only_source_at_required_confidence(self):
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.section_classifier = SimpleNamespace(classes_=np.array(['old','new']),
            predict_proba=Mock(return_value=np.array([[.08,.92]])))
        pipeline.section_vectorizer = SimpleNamespace(transform=Mock())
        pipeline._known_sections = {'old','new'}
        state = State(messages=[{'role':'user','content':'공부 질문'}],label=Label.NORMAL)
        async def parent(p, state):
            state.section = 'old'
        with patch.object(ContextFast, '_classify', parent):
            await pipeline._classify(state)
            self.assertEqual(state.section, 'new')
            self.assertEqual(state.label, Label.NORMAL)
            pipeline.section_classifier.predict_proba.return_value = np.array([[.11,.89]])
            await pipeline._classify(state)
            self.assertEqual(state.section, 'old')

    def test_loader_keeps_guard_output_and_generation(self):
        from pipelines import load
        self.assertIs(load('jh_r3_section90'),Pipeline)
        for method in ('guard','_check_output','_generate','_search'):
            self.assertIs(getattr(Pipeline,method),getattr(ContextFast,method))
