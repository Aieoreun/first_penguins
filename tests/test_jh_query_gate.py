import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.state import Chunk, State
from pipelines.kimjaehak.query_gate import ContextFast, Pipeline, use_last


class QueryGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_original_dialogue_is_preserved_for_classification(self):
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.searcher = SimpleNamespace()
        state = State(messages=[{'role':'user','content':'지난 질문'},
                                {'role':'assistant','content':'지난 답변'},
                                {'role':'user','content':'독립된 새 질문'}])
        old = copy.deepcopy(state.messages)
        async def search(p, selected, extra=''):
            self.assertEqual(selected.messages, old[-1:])
            selected.chunks = [Chunk('new', '교재')]
            selected.errors.append('embed:TimeoutError')
        with patch('pipelines.kimjaehak.query_gate.query_features', return_value={}), \
                patch('pipelines.kimjaehak.query_gate.use_last', return_value=True), \
                patch.object(ContextFast, '_search', search):
            await pipeline._search(state)
        self.assertEqual(state.messages, old)
        self.assertEqual(state.chunks[0].section_id, 'new')
        self.assertEqual(state.errors, ['embed:TimeoutError'])
        self.assertIsNone(state.label)

    async def test_guard_uses_original_search_without_gate(self):
        pipeline = Pipeline.__new__(Pipeline)
        state = State(messages=[{'role':'user','content':'앞'}, {'role':'user','content':'뒤'}])
        state._query_guard_endpoint = True
        async def search(p, received, extra=''):
            self.assertIs(received, state)
        with patch('pipelines.kimjaehak.query_gate.query_features') as features, \
                patch.object(ContextFast, '_search', search):
            await pipeline._search(state)
        features.assert_not_called()

    def test_ambiguous_or_weak_last_question_does_not_drop_history(self):
        good = dict(base='old', last='new', score=15, gap=.35, terms=8)
        self.assertTrue(use_last(good))
        for change in ({'score':11}, {'gap':.29}, {'terms':3}, {'last':'old'}):
            self.assertFalse(use_last({**good, **change}))

    def test_guard_and_answer_methods_unchanged(self):
        from pipelines import load
        self.assertIs(load('jh_r3_query_gate'), Pipeline)
        for method in ('guard', '_classify', '_check_output', '_generate'):
            self.assertIs(getattr(Pipeline, method), getattr(ContextFast, method))
