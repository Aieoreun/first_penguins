import asyncio
import unittest
from unittest.mock import patch

from pipelines.kimjaehak.guard_fitted import ContextFast
from pipelines.kimjaehak.input_examples import DenseTen
from pipelines.kimjaehak.tutor_split import DenseOnly, MaterialOnly, Pipeline


class TutorSplitTests(unittest.IsolatedAsyncioTestCase):
    def make_pipeline(self, cls=Pipeline):
        def initialize(instance, searcher, llm):
            instance.searcher, instance.llm = searcher, llm
            instance._slots = asyncio.Semaphore(2)
        with patch.object(ContextFast, "__init__", initialize), patch.object(cls.tutor_type, "__init__", initialize):
            return cls(object(), object())

    def test_loader_and_guard_inheritance(self):
        from pipelines import load
        self.assertIs(load("jh_r2_tutor_split"), Pipeline)
        for name in ("guard", "_run", "_classify", "_search", "_check_output"):
            self.assertIs(getattr(Pipeline, name), getattr(ContextFast, name))
        self.assertIs(Pipeline.tutor_type, DenseTen)

    async def test_tutor_always_uses_new_path_without_mutating_body(self):
        pipeline = self.make_pipeline()
        body = {"messages": [], "candidate_answer": "optional field must not select guard"}
        with patch.object(DenseTen, "tutor", return_value="new tutor") as tutor:
            self.assertEqual(await pipeline.tutor(body), "new tutor")
            tutor.assert_awaited_once_with(body)
        self.assertEqual(body["candidate_answer"], "optional field must not select guard")

    async def test_guard_without_candidate_uses_original_path(self):
        pipeline = self.make_pipeline()
        body = {"messages": []}
        with patch.object(ContextFast, "_run", return_value="old guard") as run:
            self.assertEqual(await pipeline.guard(body), "old guard")
            run.assert_awaited_once_with(body, True)

    async def test_shared_capacity_across_endpoints_and_cancellation(self):
        pipeline = self.make_pipeline()
        self.assertIs(pipeline._slots, pipeline._tutor_pipeline._slots)
        self.assertIs(pipeline.llm, pipeline._tutor_pipeline.llm)
        active = peak = 0
        entered = asyncio.Event()
        async def run(instance, body, guard=False):
            nonlocal active, peak
            async with instance._slots:
                active += 1
                peak = max(peak, active)
                entered.set()
                try:
                    await asyncio.sleep(.01)
                    return guard
                finally:
                    active -= 1
        with patch.object(ContextFast, "_run", run), patch.object(DenseTen, "tutor", run):
            tasks = [asyncio.create_task(pipeline.guard({}) if i % 2 else pipeline.tutor({})) for i in range(8)]
            await entered.wait()
            tasks[0].cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.assertEqual(await pipeline.guard({}), True)
        self.assertEqual(peak, 2)
        self.assertEqual(active, 0)
        self.assertEqual(pipeline._slots._value, 2)

    def test_tutor_settings_do_not_leak_into_guard(self):
        pipeline = self.make_pipeline()
        self.assertEqual(pipeline._tutor_pipeline.material_max_chunks, 10)
        self.assertTrue(pipeline._tutor_pipeline.dense_input_examples)
        self.assertIsNone(pipeline.material_max_chunks)
        self.assertFalse(hasattr(pipeline, "policy_examples"))

    def test_ablation_variants_isolate_each_change(self):
        from pipelines import load
        self.assertIs(load("jh_r2_tutor_dense"), DenseOnly)
        self.assertIs(load("jh_r2_tutor_material10"), MaterialOnly)
        dense, material = self.make_pipeline(DenseOnly), self.make_pipeline(MaterialOnly)
        self.assertTrue(dense._tutor_pipeline.dense_input_examples)
        self.assertIsNone(dense._tutor_pipeline.material_max_chunks)
        self.assertFalse(hasattr(material._tutor_pipeline, "dense_input_examples"))
        self.assertEqual(material._tutor_pipeline.material_max_chunks, 10)
        for pipeline in (dense, material):
            self.assertIsNone(pipeline.material_max_chunks)
            self.assertIs(pipeline._slots, pipeline._tutor_pipeline._slots)
