"""Keep the tested organizer proxy limits across registered R6 candidates."""
import unittest

from pipelines import load
from pipelines.kimjaehak.config import VARIANTS


class Round6Contracts(unittest.TestCase):
    def test_registered_candidates_keep_nonstream_token_caps_and_deadline(self):
        for name in VARIANTS:
            if not name.startswith('jh_r6_'):
                continue
            with self.subTest(variant=name):
                pipeline = load(name)
                self.assertEqual(pipeline.deadline, 26)
                self.assertFalse(getattr(pipeline, 'source_stream_early_finish', False))
                for field in ('source_selection_tokens', 'joint_evidence_tokens', 'source_support_tokens',
                              'answer_max_tokens', 'short_max_tokens', 'material_max_tokens'):
                    value = getattr(pipeline, field, None)
                    if value is not None:
                        self.assertGreater(value, 0, field)
                        self.assertLessEqual(value, 2000, field)


if __name__ == '__main__':
    unittest.main()
