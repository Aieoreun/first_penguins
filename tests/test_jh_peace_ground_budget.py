import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from core.search import Searcher
from core.state import Chunk
from pipelines.kimjaehak.peace_ground_budget import LowThree, MediumThree
from pipelines.kimjaehak.peace_span_variants import RoutedSpan80


class GroundBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_ground_votes_use_bounded_chat_and_same_span_contract(self):
        for cls, effort, tokens, timeout in ((LowThree, 'low', 1500, 13),
                                            (MediumThree, 'medium', 2000, 20)):
            with self.subTest(candidate=cls.__name__):
                p = cls.__new__(cls)
                p.llm = SimpleNamespace(chat=AsyncMock(return_value='{"unsupported":["wrong"]}'))
                p.searcher = Searcher([Chunk('A', 'textbook')])
                p.span_tagger = SimpleNamespace(predict=lambda text: ([], .5))
                state = p._new_state({'messages':[{'role':'user','content':'lesson question'}],
                                      'candidate_answer':'wrong statement'}, True)
                state.chunks, state.section = p.searcher.chunks, 'A'
                await p._fusion_ground(state)
                self.assertEqual(p.llm.chat.await_count, 3)
                self.assertFalse(state.grounded)
                self.assertEqual(state.unsupported, ['wrong'])
                for call in p.llm.chat.call_args_list:
                    self.assertEqual(call.kwargs, {'effort':effort,'max_tokens':tokens,
                                                  'timeout':timeout,'json_mode':True})
                    self.assertEqual(json.loads(call.args[1])['후보 답변'], 'wrong statement')

    def test_input_safety_and_tutor_are_inherited_without_base_mutation(self):
        for cls in (LowThree, MediumThree):
            for method in ('tutor', '_classify', '_fusion_safety', '_generate'):
                self.assertIs(getattr(cls, method), getattr(RoutedSpan80, method))
            self.assertEqual(cls.guard_deadline, 28)
            self.assertEqual(cls.learned_span_threshold, .8)
        self.assertEqual(RoutedSpan80.output_votes, 5)
        self.assertEqual(RoutedSpan80.guard_ground_timeout, 13)


if __name__ == '__main__':
    unittest.main()
