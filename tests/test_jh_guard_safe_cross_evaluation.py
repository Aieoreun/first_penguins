from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import jh_guard_safe_cross_diagnostic as builder
from experiments import jh_guard_safe_cross_evaluation as evaluation


def response(safe=True):
    return {'input': {'decision': 'allow', 'category': 'normal'},
            'output': {'safe': safe, 'grounded': True, 'unsupported': []}}


class SafeCrossEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items, cls.metadata = evaluation.load(builder.ROOT)

    def rows(self):
        # Synthetic verdicts only: no model outputs or dataset evaluation.
        return [{'id': i['id'], 'response': response(i['expected_safe']), 'error': None}
                for i in self.items]

    def test_load_exact_approved_eight_and_preserve_original_cache(self):
        self.assertEqual(len(self.items), 8)
        self.assertEqual(self.metadata['pair_count'], 4)
        self.assertFalse(self.metadata['training_allowed'])
        self.assertFalse(self.metadata['deployment_allowed'])
        self.assertFalse(self.metadata['official_score'])
        proposal = builder.build()
        self.assertEqual(self.items, proposal['items'])
        self.assertEqual(proposal['status'], 'pending_parent_review')
        self.assertEqual(self.metadata['review_sha256'], evaluation.REVIEW_SHA256)
        self.assertEqual(set(self.metadata['accepted_pair_ids']), {i['pair_id'] for i in self.items})

    def test_each_source_pin_and_review_checked(self):
        sources = self.metadata['source_files']
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for relative in sources:
                dest = root / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes((builder.ROOT / relative).read_bytes())
            self.assertEqual(evaluation.load(root)[0], self.items)
            paths = [evaluation.REVIEW_PATH, 'public/guard.jsonl',
                     'experiments/jh_guard_safe_cross_diagnostic.py', 'kb/python/PY-03-02.md',
                     'experiments/.cache/jh_guard_safe_cross_proposal_v1/proposal.json',
                     'experiments/.cache/jh_guard_safe_cross_proposal_v1/proposed_inputs.jsonl',
                     'experiments/.cache/jh_guard_safe_cross_proposal_v1/manifest.json']
            for relative in paths:
                with self.subTest(path=relative):
                    path = root / relative
                    original = path.read_bytes()
                    path.write_bytes(original + b' ')
                    with self.assertRaises(ValueError):
                        evaluation.load(root)
                    path.write_bytes(original)

    def test_regeneration_mismatch_rejected_even_with_pinned_artifacts(self):
        changed = builder.build()
        changed['items'][0]['candidate_answer'] += 'changed'
        with patch.object(builder, 'build', return_value=changed), self.assertRaises(ValueError):
            evaluation.load(builder.ROOT)

    def test_perfect_and_all_true_synthetic_verdict_metrics(self):
        result = evaluation.score(self.items, self.rows())
        for metric in ('safe_accuracy', 'control_safety', 'unsafe_detection', 'paired_accuracy',
                       'input_normal_accuracy'):
            self.assertEqual(result[metric], 1.)
        self.assertTrue(result['collection_valid'])
        self.assertFalse(result['official_score'])
        rows = self.rows()
        for row in rows:
            row['response']['output']['safe'] = True
        result = evaluation.score(self.items, rows)
        self.assertEqual(result['safe_accuracy'], .5)
        self.assertEqual(result['control_safety'], 1.)
        self.assertEqual(result['unsafe_detection'], 0.)
        self.assertEqual(result['paired_accuracy'], 0.)

    def test_missing_duplicate_nonbool_and_malformed_count_as_wrong(self):
        rows = self.rows()[1:]
        rows.append(deepcopy(rows[0]))  # Duplicate all copies, even if both correct.
        rows[1]['response']['output']['safe'] = 1
        rows[2]['response']['output']['unsupported'] = 'not-a-list'
        result = evaluation.score(self.items, rows)
        self.assertEqual(result['errors'], {'missing': 1, 'duplicate_id': 1,
                                           'transport_error': 0, 'invalid_response': 2})
        self.assertEqual(result['valid_count'], 4)
        self.assertEqual(result['safe_accuracy'], .5)
        self.assertEqual(result['paired_accuracy'], .5)
        self.assertEqual(result['input_normal_accuracy'], .75)
        self.assertFalse(result['collection_valid'])
        # An empty/missing response or string bool cannot pass the parser.
        for invalid in (None, {}, {'output': {'safe': False}}, response('false')):
            rows = self.rows()
            rows[0]['response'] = invalid
            self.assertEqual(evaluation.score(self.items, rows)['errors']['invalid_response'], 1)

    def test_transport_error_unknown_and_malformed_rows_exposed(self):
        rows = self.rows()
        rows[0]['error'] = 'TimeoutError'
        rows.extend([{'id': 'unexpected', 'response': response()}, {'response': response()}, None])
        result = evaluation.score(self.items, rows)
        self.assertEqual(result['n'], 8)
        self.assertEqual(result['errors']['transport_error'], 1)
        self.assertEqual(result['unexpected_ids'], ['unexpected'])
        self.assertEqual(result['malformed_row_count'], 2)
        self.assertEqual(result['safe_accuracy'], 7/8)
        self.assertFalse(result['collection_valid'])

    def test_input_auxiliary_and_grounding_values_do_not_change_safety(self):
        rows = self.rows()
        for row in rows:
            row['response']['output']['grounded'] = False
            row['response']['output']['unsupported'] = ['not evaluated as a span']
        rows[0]['response']['input'] = {'decision': 'block', 'category': 'harmful_content'}
        result = evaluation.score(self.items, rows)
        self.assertEqual(result['safe_accuracy'], 1.)
        self.assertEqual(result['paired_accuracy'], 1.)
        self.assertEqual(result['input_normal_accuracy'], 7/8)
        self.assertNotIn('span_f1', result)
        self.assertNotIn('guard_score', result)

    def test_invalid_expected_ids_or_pair_balance_fail(self):
        for bad in (self.items + [self.items[0]], self.items[:-1], []):
            with self.assertRaises(ValueError):
                evaluation.score(bad, [])
        bad = deepcopy(self.items)
        bad[0]['expected_safe'] = 'true'
        with self.assertRaises(ValueError):
            evaluation.score(bad, [])


if __name__ == '__main__':
    unittest.main()
