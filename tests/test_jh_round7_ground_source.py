import json
import unittest
from unittest.mock import Mock
from core.schema import Label
from core.search import Searcher
from core.state import Chunk
from pipelines import load
load('jh_cm_char_mlp50')
from pipelines.kimjaehak.classifier_matrix import CharMLP50
from pipelines.kimjaehak.round7_ground_source import InputGroundSource, UnionGroundSource
from pipelines.kimjaehak.context_output import Pipeline as ContextOutput
from tests import test_jh_reasoning_ablation as reasoning


class SourceTests(unittest.IsolatedAsyncioTestCase):
    pipeline = reasoning.ReasoningTests.pipeline
    state = reasoning.ReasoningTests.state

    def ready(self, cls, guard=True):
        p=self.pipeline(cls)
        p.searcher=Searcher([Chunk('A','original material'),Chunk('B','question material')])
        s=self.state(p,guard);s.label=Label.NORMAL
        p.searcher.search=Mock(return_value=[Chunk('B','question material')])
        return p,s

    async def test_actual_ground_call_uses_input_source_without_mutating_state(self):
        p,s=self.ready(InputGroundSource)
        before=(s.section,list(s.chunks),list(s.messages),s.candidate_answer)
        p.llm.chat.return_value='{"unsupported":[]}'
        await ContextOutput._check_output(p,s)
        payload=json.loads(p.llm.chat.await_args.args[1])
        self.assertEqual([r['section'] for r in payload['교재']],['B'])
        self.assertEqual(payload['대화'],s.messages)
        self.assertEqual(payload['후보 답변'],s.candidate_answer)
        p.searcher.search.assert_called_once_with('\n'.join(m['content'] for m in s.messages),bm25_top=1,embedding_top=0)
        self.assertEqual((s.section,s.chunks,s.messages,s.candidate_answer),before)
        self.assertTrue(s.debug['r7_ground_source_changed'])

    def test_union_keeps_original_first_deduplicates_and_empty_search_falls_back(self):
        p,s=self.ready(UnionGroundSource)
        self.assertEqual([r['section'] for r in p._ground_payload(s)['교재']],['A','B'])
        p.searcher.search.return_value=[Chunk('A','original material')]
        self.assertEqual([r['section'] for r in p._ground_payload(s)['교재']],['A'])
        self.assertFalse(s.debug['r7_ground_source_changed'])
        p.searcher.search.return_value=[]
        self.assertEqual(p._ground_payload(s),CharMLP50._ground_payload(p,s))

    def test_tutor_and_unrelated_methods_are_unchanged(self):
        for name,cls in [('jh_r7_ground_input',InputGroundSource),('jh_r7_ground_union',UnionGroundSource)]:
            self.assertIs(load(name),cls)
            p,s=self.ready(cls,False)
            self.assertEqual(p._ground_payload(s),CharMLP50._ground_payload(p,s))
            p.searcher.search.assert_not_called()
            for method in ('tutor','guard','_classify','_search','_generate','_fusion_ground','_fusion_safety'):
                self.assertIs(getattr(cls,method),getattr(CharMLP50,method))


if __name__=='__main__':unittest.main()
