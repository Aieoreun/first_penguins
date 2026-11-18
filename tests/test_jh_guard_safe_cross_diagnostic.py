from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import jh_guard_safe_cross_diagnostic as diagnostic


class SafeCrossProposalTests(unittest.TestCase):
    def test_verbatim_sources_balanced_pairs_and_no_grounding_labels(self):
        proposal = diagnostic.build()
        rows = {r['id']: r for r in map(json.loads,
                (diagnostic.ROOT / 'public/guard.jsonl').read_text().splitlines())}
        self.assertEqual(proposal['status'], 'pending_parent_review')
        self.assertFalse(proposal['model_predictions_inspected'])
        self.assertFalse(proposal['evaluation_performed'])
        self.assertEqual(proposal['api_calls'], 0)
        self.assertEqual(len(proposal['pairs']), 4)
        self.assertEqual(len(proposal['items']), 8)
        items = {item['id']: item for item in proposal['items']}
        for pair in proposal['pairs']:
            control, unsafe = [items[x] for x in pair['item_ids']]
            question = rows[pair['question_source']['public_source_id']]
            self.assertEqual(control['messages'], question['messages'])
            self.assertEqual(unsafe['messages'], control['messages'])
            self.assertTrue(control['expected_safe'])
            self.assertFalse(unsafe['expected_safe'])
            support = pair['question_kb_support']
            self.assertIn(support['quote'], (diagnostic.ROOT / support['path']).read_text())
            for item in (control, unsafe):
                source = item['candidate_source']
                original = rows[source['public_source_id']]
                self.assertEqual(item['candidate_answer'], original['candidate_answer'])
                self.assertEqual(source['candidate_sha256'], diagnostic.digest(original['candidate_answer'].encode()))
                self.assertEqual(source['original_labels'], original['labels'])
                self.assertEqual(item['expected_input'], {'category': 'normal', 'decision': 'allow'})
                self.assertNotIn('labels', item)
                self.assertNotIn('grounded', item)
                self.assertNotIn('unsupported', item)
        self.assertEqual(proposal, diagnostic.build())

    def test_changed_public_or_invalid_kb_quote_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'public').mkdir()
            (root / 'public/guard.jsonl').write_text('changed')
            with self.assertRaises(ValueError):
                diagnostic.build(root)
        specs = deepcopy(diagnostic.PAIRS)
        specs[0]['quote'] = 'NOT_A_KB_QUOTE'
        with patch.object(diagnostic, 'PAIRS', specs), self.assertRaises(ValueError):
            diagnostic.build()

    def test_ambiguous_unsafe_category_and_non_normal_question_rejected(self):
        for field, source_id in [('unsafe_id', 'pub-1689289f'), ('question_id', 'pub-1499aec9')]:
            specs = deepcopy(diagnostic.PAIRS)
            specs[0][field] = source_id
            with patch.object(diagnostic, 'PAIRS', specs), self.assertRaises(ValueError):
                diagnostic.build()

    def test_write_is_immutable_and_records_hash(self):
        proposal = diagnostic.build()
        with tempfile.TemporaryDirectory() as temp, patch.object(diagnostic, 'ROOT', Path(temp)), \
                patch.object(diagnostic, 'build', return_value=proposal):
            dest = diagnostic.write_proposal('test')
            manifest = json.loads((dest / 'manifest.json').read_text())
            self.assertEqual(manifest['proposal_sha256'], diagnostic.digest((dest / 'proposal.json').read_bytes()))
            self.assertEqual(manifest['item_count'], 8)
            self.assertIn('parent review pending', (dest / 'PROPOSAL.md').read_text())
            with self.assertRaises(FileExistsError):
                diagnostic.write_proposal('test')
            with self.assertRaises(ValueError):
                diagnostic.write_proposal('../escape')


if __name__ == '__main__':
    unittest.main()
