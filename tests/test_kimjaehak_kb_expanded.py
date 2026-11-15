import unittest

from pipelines.kimjaehak.kb_expanded import ExpandedIndex, Fitted
from tests.test_pipeline import SEARCHER


class KBExpandedTests(unittest.TestCase):
    def test_empty_auxiliary_index_preserves_bm25(self):
        index = ExpandedIndex(SEARCHER,[])
        for query in ("다람쥐 첫걸음 통장의 이체 한도", "fruit_order tuple", "점수와 range 조건", ""):
            old = SEARCHER.search(query)
            new = index.best_chunk(query,.1)
            self.assertEqual(new.section_id if new else None,old[0].section_id if old else None)

    def test_unknown_generated_section_is_rejected(self):
        with self.assertRaises(ValueError):
            ExpandedIndex(SEARCHER,[{"section":"INVALID","questions":["question"],"error":None}])

    def test_fitted_composition_retains_tutor_only_section_policy(self):
        self.assertEqual(Fitted.learned_gate_threshold,.95)
        self.assertFalse(Fitted.learned_section_use_for_output)
        self.assertEqual(Fitted.query_expansion_weight,.1)


if __name__ == "__main__":
    unittest.main()
