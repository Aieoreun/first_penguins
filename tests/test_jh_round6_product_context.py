import asyncio
from copy import deepcopy
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
from pipelines.kimjaehak.calibrated import BLOCK_SYSTEM
from pipelines.kimjaehak.material_budget import Pipeline as Base
from pipelines.kimjaehak.round6_product_context import Pipeline, PRODUCT_CONTEXT_RULE


class ProductContextTests(unittest.IsolatedAsyncioTestCase):
    def setup_pipeline(self, cls=Pipeline, guard=False, quote='', block='none'):
        p = cls.__new__(cls)
        p.searcher = Searcher([Chunk('A', 'title A\ntextbook fact')])
        p.vectorizer = p.section_vectorizer = SimpleNamespace(transform=lambda text: text)
        p.classifier = SimpleNamespace(classes_=np.array(['normal', 'out_of_material']),
            predict_proba=lambda text: np.array([[.6, .4]]))
        p.section_classifier = SimpleNamespace(classes_=np.array(['A', 'B']),
            predict_proba=lambda text: np.array([[.6, .4]]))
        p.guard_vectorizer, p.guard_classifier = p.vectorizer, p.classifier
        p._known_sections = {'A'}
        async def chat(system, user, **kwargs):
            await asyncio.sleep(0)
            if system == prompts.MATERIAL_SYSTEM:
                return '{"quote":"' + quote + '","section":"A"}'
            if system.startswith(BLOCK_SYSTEM):
                return '{"category":"' + block + '"}'
            return '{"category":"none","self_info":false}'
        p.llm = SimpleNamespace(chat=AsyncMock(side_effect=chat))
        s = p._new_state({'messages': [{'role': 'user', 'content': 'textbook product question'}]}, guard)
        s.chunks = p.searcher.chunks
        return p, s

    async def test_actual_mro_keeps_empty_material_abstain_and_original_calls(self):
        p, s = self.setup_pipeline()
        before = deepcopy((s.messages, s.request, s.chunks))
        await p._classify(s)
        self.assertEqual(s.label, Label.OUT_OF_MATERIAL)
        self.assertEqual((s.messages, s.request, s.chunks), before)
        self.assertEqual(p.llm.chat.await_count, 5)
        base, original = self.setup_pipeline(Base)
        await base._classify(original)
        for changed, unchanged in zip(p.llm.chat.call_args_list, base.llm.chat.call_args_list):
            self.assertEqual(changed.args[1:], unchanged.args[1:])
            self.assertEqual(changed.kwargs, unchanged.kwargs)
            expected = unchanged.args[0]
            if expected == BLOCK_SYSTEM:
                expected += PRODUCT_CONTEXT_RULE
            self.assertEqual(changed.args[0], expected)
        await p._generate(s)
        self.assertEqual(s.to_tutor_response()['action'], 'abstain')
        self.assertEqual(p.llm.chat.await_count, 5)

    async def test_actual_jailbreak_vote_retains_priority_over_material(self):
        for quote in ('', 'textbook fact'):
            p, s = self.setup_pipeline(quote=quote, block='jailbreak_injection')
            await p._classify(s)
            self.assertEqual(s.label, Label.JAILBREAK_INJECTION)
            self.assertEqual(s.to_tutor_response()['action'], 'refuse')
            self.assertEqual(p.llm.chat.await_count, 5)

    async def test_supported_normal_has_no_forced_category(self):
        p, s = self.setup_pipeline(quote='textbook fact')
        await p._classify(s)
        self.assertEqual(s.label, Label.NORMAL)
        self.assertFalse(s.debug.get('learned_gate'))
        self.assertEqual(s.debug['block_votes'], ['none', 'none'])

    async def test_guard_and_missing_identity_use_exact_parent_prompt(self):
        p, s = self.setup_pipeline(guard=True)
        base, original = self.setup_pipeline(Base, guard=True)
        self.assertIs(p._block_system(s), base._block_system(original))
        await p._classify(s)
        await base._classify(original)
        self.assertEqual(p.llm.chat.call_args_list, base.llm.chat.call_args_list)
        self.assertEqual(s.to_tutor_response(), original.to_tutor_response())
        del s._material_budget_guard
        self.assertIs(p._block_system(s), BLOCK_SYSTEM)
        self.assertIs(Pipeline.guard, Base.guard)
        self.assertIs(Pipeline._check_output, Base._check_output)

    async def test_concurrent_tutor_and_guard_keep_request_local_identity(self):
        p, tutor = self.setup_pipeline(quote='textbook fact')
        guard = p._new_state({'messages': [{'role': 'user', 'content': 'guard question'}]}, True)
        guard.chunks = p.searcher.chunks
        await asyncio.gather(p._classify(tutor), p._classify(guard))
        blocks = [c for c in p.llm.chat.call_args_list if c.args[0].startswith(BLOCK_SYSTEM)]
        self.assertEqual(len(blocks), 4)
        for call in blocks:
            is_guard = 'guard question' in call.args[1]
            self.assertEqual(call.args[0], BLOCK_SYSTEM if is_guard else Pipeline.tutor_block_system)
        self.assertEqual(p.llm.chat.await_count, 10)
        self.assertEqual(Base._block_system(p, tutor), BLOCK_SYSTEM)

    async def test_learned_gate_still_skips_all_checks(self):
        p, s = self.setup_pipeline()
        p.classifier.predict_proba = lambda text: np.array([[.99, .01]])
        await p._classify(s)
        self.assertTrue(s.debug['learned_gate'])
        p.llm.chat.assert_not_awaited()

    def test_registration_parent_methods_and_budgets(self):
        with patch.dict(os.environ):
            self.assertIs(load('jh_r6_product_context'), Pipeline)
            self.assertIs(load('jh_r4_material_budget'), Base)
        self.assertEqual(DEFAULT, 'v1')
        self.assertIs(Pipeline._classify, Base._classify)
        self.assertIs(Pipeline._ask, Base._ask)
        self.assertIs(Pipeline._generate, Base._generate)
        self.assertEqual((Pipeline.deadline, Pipeline.tutor_prepare_limit, Pipeline.tutor_material_timeout), (26, 16, 13))


if __name__ == '__main__':
    unittest.main()
