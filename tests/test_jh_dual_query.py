import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core.schema import Label
from core.state import Chunk, State
from pipelines.kimjaehak.dual_query import ContextFast, Pipeline, Sources


class DualQueryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.pipeline = Pipeline.__new__(Pipeline)
        self.pipeline.searcher = SimpleNamespace(search=Mock(return_value=[Chunk('new','new text')]),
                                                section_text=lambda sid:sid+' material')
        self.state = State(messages=[{'role':'user','content':'앞 질문'}, {'role':'user','content':'새 질문'}],
                           label=Label.NORMAL, section='old')

    async def test_search_adds_other_view_without_altering_dialogue(self):
        old = copy.deepcopy(self.state.messages)
        async def search(p, state, extra=''):
            state.chunks = [Chunk('old','old text'), Chunk('rest','rest text')]
        with patch.object(ContextFast, '_search', search):
            await self.pipeline._search(self.state)
        self.assertEqual(self.state.messages, old)
        self.assertEqual([c.section_id for c in self.state.chunks], ['old','new','rest'])
        self.assertEqual(self.state._dual_sections, ['old','new'])

    async def test_guard_never_adds_tutor_search(self):
        self.state._dual_guard = True
        async def search(p, state, extra=''):
            state.chunks = [Chunk('old','old text')]
        with patch.object(ContextFast, '_search', search):
            await self.pipeline._search(self.state)
        self.pipeline.searcher.search.assert_not_called()

    async def test_generation_uses_own_two_sources_and_preserves_deadline(self):
        self.state._dual_sections = ['old','new']
        self.state.round2_deadline = 12345
        self.state.chunks = [Chunk('original','original')]
        async def generate(p, selected):
            self.assertEqual(selected.round2_deadline, 12345)
            self.assertEqual(selected.section, 'old')
            self.assertEqual([c.section_id for c in selected.chunks], ['new'])
            selected.section, selected.answer, selected.citations = 'new', '새 답변', ['new']
            selected.errors.append('round2_generate:ValueError')
        with patch.object(Sources, '_generate', generate):
            await self.pipeline._generate(self.state)
        self.assertEqual(self.state.section,'new')
        self.assertEqual(self.state.chunks[0].section_id,'original')
        self.assertEqual(self.state.citations,['new'])
        self.assertIn('round2_generate:ValueError',self.state.errors)

    async def test_confident_sources_and_policy_responses_keep_original_generation(self):
        self.state._dual_sections = ['old','new']
        called=[]
        async def generate(p, state):
            called.append(state)
        with patch.object(ContextFast, '_generate', generate):
            self.state.debug['learned_section'] = True
            await self.pipeline._generate(self.state)
            self.state.debug = {}
            self.state.label = Label.ACADEMIC_DISHONESTY
            await self.pipeline._generate(self.state)
        self.assertEqual(called,[self.state,self.state])

    def test_loader_preserves_guard_output_and_classification(self):
        from pipelines import load
        self.assertIs(load('jh_r3_dual_query'),Pipeline)
        for method in ('guard','_classify','_check_output'):
            self.assertIs(getattr(Pipeline,method),getattr(ContextFast,method))
