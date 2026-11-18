"""Reproduction/provenance checks only; these tests never score a model."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import jh_guard_intent_pairs as diagnostic


class IntentPairTests(unittest.TestCase):
    def test_real_proposal_is_input_only_balanced_and_changes_last_request_only(self):
        before = deepcopy(diagnostic.PAIRS)
        proposals, inputs, manifest = diagnostic.build()
        self.assertEqual(before, diagnostic.PAIRS)
        self.assertEqual((len(proposals), len(inputs)), (8, 16))
        counts = Counter(row['labels']['category'] for row in inputs)
        self.assertEqual(counts, {'normal': 8, 'harmful_content': 2,
            'jailbreak_injection': 2, 'academic_dishonesty': 2, 'off_topic': 2})
        for normal, blocked in zip(inputs[::2], inputs[1::2]):
            self.assertEqual(normal['pair_id'], blocked['pair_id'])
            self.assertEqual(normal['messages'][:-1], blocked['messages'][:-1])
            self.assertNotEqual(normal['messages'][-1], blocked['messages'][-1])
            for row in (normal, blocked):
                self.assertNotIn('candidate_answer', row)
                self.assertEqual(set(row['labels']), {'category', 'decision'})
                self.assertEqual(set(row), {'id', 'endpoint', 'pair_id', 'contrast_kind', 'messages', 'labels'})
        self.assertEqual(manifest['review_status'], 'pending_parent_review_before_evaluation')
        for name in ('evaluation_allowed', 'training_allowed', 'deployment_allowed', 'independent_holdout'):
            self.assertIs(manifest[name], False)

    def test_exact_support_and_policy_quotes_have_reproducible_hashes_and_lines(self):
        proposals, _, manifest = diagnostic.build()
        for pair in proposals:
            for source in pair['kb_sources'] + pair['policy_sources']:
                path = diagnostic.ROOT / source['path']
                self.assertEqual(diagnostic.digest(path.read_bytes()), source['source_sha256'])
                self.assertEqual(diagnostic.digest(source['quote'].encode()), source['quote_sha256'])
                self.assertEqual(path.read_text()[:path.read_text().index(source['quote'])].count('\n') + 1,
                                 source['line'])
        self.assertEqual(manifest['source_files'], diagnostic.SOURCE_SHA)

    def test_deterministic_write_verify_and_no_overwrite(self):
        payload = diagnostic.artifacts()
        self.assertEqual(payload, diagnostic.artifacts())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with patch.object(diagnostic, 'artifacts', return_value=payload):
                dest = diagnostic.write(root)
                self.assertEqual(diagnostic.verify(root), dest)
                self.assertEqual({p.name for p in dest.iterdir()}, set(payload))
                manifest = json.loads((dest / 'manifest.json').read_text())
                self.assertEqual(manifest['dataset_sha256'], diagnostic.digest((dest/'inputs.jsonl').read_bytes()))
                self.assertEqual(manifest['proposal_sha256'], diagnostic.digest((dest/'proposal.json').read_bytes()))
                with self.assertRaises(FileExistsError):
                    diagnostic.write(root)
                (dest/'inputs.jsonl').write_text('tampered')
                with self.assertRaises(ValueError):
                    diagnostic.verify(root)
                with self.assertRaises(ValueError):
                    diagnostic.write(root, 'outside-cache')

    def test_changed_sources_bad_quotes_duplicate_ids_and_missing_review_rejected(self):
        expected = dict(diagnostic.SOURCE_SHA)
        expected['SPEC.md'] = '0' * 64
        with self.assertRaises(ValueError):
            diagnostic.build(expected_sources=expected)
        for mutation in ('quote', 'duplicate', 'rationale', 'history', 'same_request', 'category'):
            pairs = deepcopy(diagnostic.PAIRS)
            if mutation == 'quote':
                pairs[0]['kb_sources'][0]['quote'] = 'This quote is not in the source.'
            elif mutation == 'duplicate':
                pairs[1]['pair_id'] = pairs[0]['pair_id']
            elif mutation == 'rationale':
                pairs[0]['ambiguity_review'] = ''
            elif mutation == 'history':
                pairs[0]['history'][0]['role'] = 'system'
            elif mutation == 'same_request':
                pairs[0]['blocked_request'] = pairs[0]['normal_request']
            else:
                pairs[0]['blocked_category'] = 'out_of_material'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                diagnostic.build(pairs=pairs)

    def test_output_symlink_cannot_escape_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            cache = root/'experiments/.cache'
            cache.mkdir(parents=True)
            outside = root/'outside'
            outside.mkdir()
            (cache/'redirect').symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                diagnostic.cache_path(root, 'experiments/.cache/redirect/child')

    def test_revision_changes_only_eighth_pair_and_drops_candidate_specific_policy(self):
        proposals, inputs, manifest = diagnostic.build()
        self.assertEqual(manifest['version'], 2)
        self.assertEqual(manifest['previous_version']['changed_pair_ids'], ['intent-08'])
        self.assertNotIn('pipelines/kimjaehak/precise_output.py', manifest['source_files'])
        self.assertEqual({s['path'] for s in proposals[-1]['kb_sources']}, {'kb/python/PY-02-04.md'})
        prior = (deepcopy(proposals), deepcopy(inputs))
        prior[0][-1]['topic'] = 'old eighth pair'
        prior[1][-1]['messages'][-1]['content'] = 'old eighth request'
        with patch.object(diagnostic, 'preserved_previous', return_value=prior):
            self.assertEqual(diagnostic.build()[:2], (proposals, inputs))
            prior[0][0]['topic'] = 'forbidden change to first pair'
            with self.assertRaises(ValueError):
                diagnostic.build()

    def test_fresh_clone_rebuilds_without_v1_and_partial_or_changed_v1_is_rejected(self):
        expected = diagnostic.artifacts()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for name in diagnostic.SOURCE_SHA:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((diagnostic.ROOT / name).read_bytes())
            self.assertEqual(diagnostic.artifacts(root), expected)
            prior = root / diagnostic.PREVIOUS_VERSION['path']
            prior.mkdir(parents=True)
            with self.assertRaises(ValueError):
                diagnostic.build(root)
            for name in diagnostic.PREVIOUS_VERSION['files']:
                (prior / name).write_text('{}')
            with self.assertRaises(ValueError):
                diagnostic.build(root)


if __name__ == '__main__':
    unittest.main()
