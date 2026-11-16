import unittest
from pipelines.kimjaehak.draft_selection import select


class DraftSelectionTests(unittest.TestCase):
    def test_consensus_ignores_unrelated_outlier(self):
        self.assertIn(select(['저축의 이자는 3퍼센트입니다.','저축 이자는 3퍼센트입니다.','파이썬에서 print 함수로 출력합니다.'],'consensus'),(0,1))

    def test_equal_answers_and_two_way_ties_keep_first(self):
        self.assertEqual(select(['same','same'],'consensus'),0)
        self.assertEqual(select(['first answer','other words'],'consensus'),0)

    def test_empty_texts_are_supported_and_longest_ignores_whitespace(self):
        self.assertEqual(select(['',''],'consensus'),0)
        self.assertEqual(select(['a          ','long'],'longest'),1)

    def test_invalid_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError):select([],'consensus')
        with self.assertRaises(ValueError):select(['a'],'oracle')
