import asyncio
import unittest
from unittest.mock import patch

from pipelines.kimjaehak.tutor_fitting import ContextFast, Pipeline, StrongerTutor


class TutorFittingTests(unittest.IsolatedAsyncioTestCase):
    async def test_models_remain_separate_and_request_capacity_shared(self):
        def initialize(obj, searcher, llm):
            obj._slots = asyncio.Semaphore(obj.request_slots)
            obj.classifier = object()
        async def tutor(obj, body):
            return obj.learned_gate_c, obj.classifier, body
        with patch.object(ContextFast,'__init__',initialize), patch.object(StrongerTutor,'tutor',tutor):
            pipeline = Pipeline(None,None)
            outer = pipeline.classifier
            result = await pipeline.tutor({'messages':[]})
        self.assertEqual(pipeline.learned_gate_c,1000)
        self.assertEqual(result[0],3000)
        self.assertIsNot(outer,result[1])
        self.assertIs(pipeline.classifier,outer)
        self.assertIs(pipeline._slots,pipeline._tutor_pipeline._slots)

    def test_guard_helpers_are_the_original_implementation(self):
        from pipelines import load
        self.assertIs(load('jh_r3_tutor_fit3000'),Pipeline)
        for method in ('guard','_classify','_search','_check_output'):
            self.assertIs(getattr(Pipeline,method),getattr(ContextFast,method))
