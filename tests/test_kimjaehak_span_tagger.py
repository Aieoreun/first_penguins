import unittest

from pipelines.kimjaehak.span_tagger import decode_mask, refine_spans, span_mask


class SpanTaggerTests(unittest.TestCase):
    def test_exact_spans_preserve_characters_and_multiple_occurrences(self):
        text = "첫째는 12일, 둘째도 12일이에요."
        self.assertEqual(decode_mask(text, span_mask(text, ["12일"])), ["12일", "12일"])
        with self.assertRaises(ValueError):
            span_mask(text, ["없는 구절"])

    def test_refinement_requires_confidence_and_agreement_and_keeps_grounded(self):
        text = "기간은 12일이고 금액은 5천 원이에요."
        original, proposed = ["기간은 12일이고"], ["12일이고"]
        self.assertEqual(refine_spans(text, original, proposed, .99, .95), (proposed, True))
        self.assertEqual(refine_spans(text, original, proposed, .8, .95), (original, False))
        self.assertEqual(refine_spans(text, original, ["5천 원"], .99, .95), (original, False))
        self.assertEqual(refine_spans(text, [], proposed, .99, .95), ([], False))
        self.assertEqual(refine_spans(text, original, [], .99, .95), (original, False))


if __name__ == "__main__":
    unittest.main()
