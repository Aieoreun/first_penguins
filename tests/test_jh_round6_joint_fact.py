import unittest
from types import SimpleNamespace

from pipelines.kimjaehak.round6_joint_evidence import Pipeline as Original
from pipelines.kimjaehak.round6_joint_fact import Pipeline
from pipelines.kimjaehak.round6_joint_quote import Pipeline as Strict


class FactTests(unittest.TestCase):
    def test_same_prompt_but_original_adapter_ignores_unverified_quote(self):
        p = Pipeline.__new__(Pipeline)
        p.searcher = SimpleNamespace(section_text=lambda sid: 'actual KB content')
        raw = {'supported': True, 'section': 'A', 'quote': 'invented model text'}
        result, source, diagnostic = p._adapt_joint_result(raw, ['A'])
        self.assertEqual(result, {'quote': 'actual KB content', 'section': 'A'})
        self.assertEqual(source, 'A')
        self.assertIsNone(diagnostic)
        self.assertEqual(Pipeline.joint_evidence_system, Strict.joint_evidence_system)
        self.assertIs(Pipeline._adapt_joint_result, Original._adapt_joint_result)
        self.assertIs(Pipeline._generate, Original._generate)
        self.assertIs(Pipeline.guard, Original.guard)
        self.assertFalse(getattr(Pipeline, 'tutor_joint_quote', False))

    def test_unsupported_and_invalid_decisions_still_use_original_contract(self):
        p = Pipeline.__new__(Pipeline)
        self.assertEqual(p._adapt_joint_result({'supported': False, 'section': 'A'}, ['A']),
                         ({'quote': '', 'section': 'A'}, None, None))
        with self.assertRaises(ValueError):
            p._adapt_joint_result({'supported': True, 'section': 'invented'}, ['A'])


if __name__ == '__main__':
    unittest.main()
