import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from core.schema import Label
from core.state import State
from pipelines.kimjaehak.tutor_ranked import ContextFast, Pipeline


class TutorRankedTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.pipeline=Pipeline.__new__(Pipeline)
        self.pipeline.section_features=SimpleNamespace(ids=['old','new'], transform=Mock(return_value=np.zeros((2,24))))
        self.pipeline.section_ranker=SimpleNamespace(predict_proba=Mock(return_value=np.array([[.9,.1],[.2,.8]])))
        self.state=State(messages=[{'role':'user','content':'question'}],label=Label.NORMAL,section='old')

    async def run_rank(self):
        with patch.object(ContextFast,'_generate',new_callable=AsyncMock) as generate:
            await self.pipeline._generate(self.state)
            generate.assert_awaited_once_with(self.state)

    async def test_uncertain_source_changes_before_generation(self):
        await self.run_rank()
        self.assertEqual(self.state.section,'new')
        self.assertEqual(self.state.label,Label.NORMAL)

    async def test_confident_learned_section_remains(self):
        self.state.debug['learned_section']=True
        await self.run_rank()
        self.assertEqual(self.state.section,'old')
        self.pipeline.section_features.transform.assert_not_called()

    async def test_policy_response_does_not_use_ranker(self):
        self.state.label=Label.HARMFUL_REQUEST
        await self.run_rank()
        self.assertEqual(self.state.section,'old')
        self.pipeline.section_features.transform.assert_not_called()

    async def test_feature_failure_preserves_original_source(self):
        self.pipeline.section_features.transform.side_effect=ValueError
        await self.run_rank()
        self.assertEqual(self.state.section,'old')
        self.assertIn('tutor_ranker:ValueError',self.state.errors)

    def test_guard_and_input_methods_unchanged(self):
        from pipelines import load
        self.assertIs(load('jh_r3_tutor_ranked'),Pipeline)
        for name in ('guard','_classify','_check_output','_search','_new_state'):
            self.assertIs(getattr(Pipeline,name),getattr(ContextFast,name))
