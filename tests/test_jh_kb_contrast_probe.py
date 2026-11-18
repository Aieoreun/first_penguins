from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments import jh_kb_contrast_probe as probe


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical_digest(value):
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(',', ':')).encode())


class ContrastProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.cache = self.root / 'experiments/.cache/draft'
        self.cache.mkdir(parents=True)
        self.source = self.root / 'kb/python/PY-01-01.md'
        self.source.parent.mkdir(parents=True)
        self.source.write_text('# PY-01-01 Synthetic source\nThe blue basket holds three apples.\n')
        quote = 'The blue basket holds three apples.'
        self.proposal_path = self.cache / 'proposal.json'
        self.review_path = self.cache / 'review.json'
        entries = [{'path': 'kb/python/PY-01-01.md',
                    'sha256': probe.sha(self.source), 'bytes': self.source.stat().st_size}]
        pairs = []
        for number in range(3):
            positive = f'Synthetic basket question {number}: what is its content?'
            negative = f'Synthetic basket question {number}: what is its delivery date?'
            pairs.append({
                'id': f'contrast-{number}', 'domain': 'python', 'source_section': 'PY-01-01',
                'question_pair_sha256': digest(json.dumps(
                    [positive, negative], ensure_ascii=False, separators=(',', ':')).encode()),
                'supported': {
                    'question': positive,
                    'labels': {'category': 'normal', 'action': 'answer',
                               'evidence': ['PY-01-01'], 'gold_facts': ['Three apples.']},
                    'source_path': entries[0]['path'], 'source_sha256': entries[0]['sha256'],
                    'support_quote': quote, 'quote_sha256': digest(quote.encode()),
                },
                'unsupported': {
                    'question': negative,
                    'labels': {'category': 'out_of_material', 'action': 'abstain',
                               'evidence': [], 'gold_facts': []},
                },
            })
        self.proposal = {
            'scope': 'manual_organizer_kb_contrast_development_not_private_not_holdout',
            'design_constraints': {
                'only_organizer_kb_used': True,
                'public_questions_not_read_or_edited_during_design': True,
                'model_predictions_not_inspected_during_design': True,
                'pipeline_not_evaluated': True, 'api_calls': 0,
                'no_runtime_id_or_answer_lookup': True, 'independent_review_required': True,
            },
            'kb_files': entries, 'kb_manifest_sha256': canonical_digest(entries), 'pairs': pairs,
        }
        self.review = {
            'scope': 'independent_manual_kb_contrast_review',
            'kb_manifest_sha256': self.proposal['kb_manifest_sha256'],
            'model_predictions_inspected': False, 'api_calls': 0,
            'accepted_pair_ids': ['contrast-0', 'contrast-2'],
            'excluded': [{'id': 'contrast-1', 'reason': 'Synthetic exclusion.'}],
        }
        self.save()

    def save(self, *, relink=True):
        self.proposal_path.write_text(json.dumps(self.proposal, ensure_ascii=False))
        if relink:
            self.review['proposal_sha256'] = probe.sha(self.proposal_path)
        self.review_path.write_text(json.dumps(self.review, ensure_ascii=False))

    def build(self):
        return probe.build_items(self.proposal_path, self.review_path, root=self.root)

    def test_build_preserves_balanced_pairs_and_keeps_annotations_outside_messages(self):
        # Extra source annotations must not become model messages or candidates.
        positive = self.proposal['pairs'][0]['supported']
        positive['messages'] = [{'role': 'system', 'content': 'DO NOT FORWARD GOLD'}]
        positive['candidate_answer'] = 'DO NOT FORWARD GOLD'
        self.save()
        items, metadata = self.build()
        self.assertEqual([x['id'] for x in items], [
            'contrast-0-supported', 'contrast-0-unsupported',
            'contrast-2-supported', 'contrast-2-unsupported'])
        for item in items:
            number = int(item['source_id'].rsplit('-', 1)[1])
            source = self.proposal['pairs'][number][item['contrast_role']]
            self.assertEqual(item['endpoint'], 'tutor')
            self.assertEqual(item['messages'], [{'role': 'user', 'content': source['question']}])
            self.assertEqual(item['labels'], source['labels'])
            self.assertNotIn('candidate_answer', item)
            self.assertEqual(set(item['messages'][0]), {'role', 'content'})
            self.assertEqual('label_support_quote' in item, item['contrast_role'] == 'supported')
        items[0]['labels']['gold_facts'].append('Mutated diagnostic copy')
        self.assertEqual(positive['labels']['gold_facts'], ['Three apples.'])
        self.assertEqual(metadata['scope'], 'provided_kb_fact_diagnostic')
        self.assertFalse(metadata['independent_source_holdout'])
        self.assertFalse(metadata['public_questions_in_generation'])
        self.assertIsNone(metadata['generation_model'])
        self.assertEqual(metadata['source_sha256'], probe.sha(self.proposal_path))
        self.assertEqual(metadata['review_sha256'], probe.sha(self.review_path))

    def test_changed_proposal_requires_new_review_link(self):
        self.proposal['pairs'][0]['supported']['question'] += ' edited'
        self.save(relink=False)
        with self.assertRaises(ValueError):
            self.build()

    def test_review_scope_provenance_and_blindness_flags_are_required(self):
        for field, value in (
            ('scope', 'unreviewed'), ('proposal_sha256', 'wrong'),
            ('kb_manifest_sha256', 'wrong'), ('model_predictions_inspected', True),
            ('model_predictions_inspected', None), ('api_calls', 1),
        ):
            with self.subTest(field=field, value=value):
                review = deepcopy(self.review)
                review[field] = value
                self.review_path.write_text(json.dumps(review))
                with self.assertRaises(ValueError):
                    self.build()

    def test_design_cannot_have_inspected_predictions_or_run_models(self):
        original = deepcopy(self.proposal)
        for field, value in (
            ('model_predictions_not_inspected_during_design', False),
            ('pipeline_not_evaluated', False), ('api_calls', 1),
        ):
            with self.subTest(field=field):
                self.proposal = deepcopy(original)
                self.proposal['design_constraints'][field] = value
                self.save()
                with self.assertRaises(ValueError):
                    self.build()

    def test_manifest_hash_is_recomputed_even_if_review_repeats_bad_value(self):
        self.proposal['kb_manifest_sha256'] = 'same-but-not-valid'
        self.review['kb_manifest_sha256'] = 'same-but-not-valid'
        self.save()
        with self.assertRaises(ValueError):
            self.build()

    def test_manifest_byte_count_matches_real_kb_even_after_hash_relink(self):
        self.proposal['kb_files'][0]['bytes'] += 1
        self.proposal['kb_manifest_sha256'] = canonical_digest(self.proposal['kb_files'])
        self.review['kb_manifest_sha256'] = self.proposal['kb_manifest_sha256']
        self.save()
        with self.assertRaises(ValueError):
            self.build()

    def test_pair_content_hash_is_checked_after_review_relink(self):
        self.proposal['pairs'][0]['question_pair_sha256'] = 'wrong'
        self.save()
        with self.assertRaises(ValueError):
            self.build()

    def test_support_file_must_match_cited_section(self):
        pair = self.proposal['pairs'][0]
        pair['source_section'] = 'PY-99-99'
        pair['supported']['labels']['evidence'] = ['PY-99-99']
        self.save()
        with self.assertRaises(ValueError):
            self.build()

    def test_exact_quote_from_sections_attached_csv_is_supported(self):
        table = self.source.parent / 'PY-01-01-table1.csv'
        table.write_text('Object,Contents\nBlue basket,Three apples\n')
        entries = self.proposal['kb_files']
        entries.append({'path': str(table.relative_to(self.root)),
                        'sha256': probe.sha(table), 'bytes': table.stat().st_size})
        entries.sort(key=lambda row: row['path'])
        self.proposal['kb_manifest_sha256'] = canonical_digest(entries)
        self.review['kb_manifest_sha256'] = self.proposal['kb_manifest_sha256']
        positive = self.proposal['pairs'][0]['supported']
        positive.update(source_path=str(table.relative_to(self.root)),
                        source_sha256=probe.sha(table), support_quote='Blue basket,Three apples',
                        quote_sha256=digest(b'Blue basket,Three apples'))
        self.save()
        items, _ = self.build()
        self.assertEqual(items[0]['label_support_quote'], 'Blue basket,Three apples')
        self.assertEqual(items[0]['labels']['evidence'], ['PY-01-01'])

    def test_root_symlink_is_normalized_before_provenance_paths(self):
        alias = self.root / 'workspace_alias'
        alias.symlink_to(self.root, target_is_directory=True)
        items, metadata = probe.build_items(
            alias / self.proposal_path.relative_to(self.root),
            alias / self.review_path.relative_to(self.root), root=alias)
        self.assertEqual(len(items), 4)
        self.assertEqual(metadata['source'], 'experiments/.cache/draft/proposal.json')

    def test_positive_facts_are_nonempty_strings_in_nonempty_list(self):
        original = deepcopy(self.proposal)
        for value in ('A truthy string', {'fact': 'A mapping'}, [''], ['   '], [2]):
            with self.subTest(value=value):
                self.proposal = deepcopy(original)
                self.proposal['pairs'][0]['supported']['labels']['gold_facts'] = value
                self.save()
                with self.assertRaises(ValueError):
                    self.build()

    def test_optional_additional_quotes_are_exact_source_substrings(self):
        pos = self.proposal['pairs'][0]['supported']
        pos['additional_support_quotes'] = ['The blue basket']
        self.save()
        self.build()
        for value in (['The red basket'], [''], [4], 'The blue basket'):
            with self.subTest(value=value):
                pos['additional_support_quotes'] = value
                self.save()
                with self.assertRaises(ValueError):
                    self.build()

    def test_kb_content_and_inventory_tampering_are_rejected(self):
        original = self.source.read_bytes()
        self.source.write_text('Changed textbook')
        with self.assertRaises(ValueError):
            self.build()
        self.source.write_bytes(original)
        additional = self.source.parent / 'PY-01-02.md'
        additional.write_text('Unexpected source')
        with self.assertRaises(ValueError):
            self.build()
        additional.unlink()
        self.source.unlink()
        with self.assertRaises(ValueError):
            self.build()

    def test_review_accounts_for_every_pair_once_and_accepts_something(self):
        for accepted, excluded in (
            (['contrast-0'], ['contrast-1']),  # Missing pair.
            (['contrast-0', 'contrast-0', 'contrast-2'], ['contrast-1']),
            (['contrast-0', 'contrast-2'], ['contrast-0', 'contrast-1']),
            (['contrast-0', 'unknown'], ['contrast-1', 'contrast-2']),
            ([], ['contrast-0', 'contrast-1', 'contrast-2']),
        ):
            with self.subTest(accepted=accepted, excluded=excluded):
                self.review['accepted_pair_ids'] = accepted
                self.review['excluded'] = [{'id': x} for x in excluded]
                self.save()
                with self.assertRaises(ValueError):
                    self.build()

    def test_duplicate_proposal_ids_and_questions_are_rejected(self):
        original = deepcopy(self.proposal)
        self.proposal['pairs'][2]['id'] = 'contrast-0'
        self.save()
        with self.assertRaises(ValueError):
            self.build()
        self.proposal = original
        self.proposal['pairs'][2]['supported']['question'] = self.proposal['pairs'][0]['supported']['question']
        pair = self.proposal['pairs'][2]
        pair['question_pair_sha256'] = digest(json.dumps(
            [pair['supported']['question'], pair['unsupported']['question']],
            ensure_ascii=False, separators=(',', ':')).encode())
        self.save()
        with self.assertRaises(ValueError):
            self.build()

    def test_positive_quote_must_be_exact_and_have_valid_hashes(self):
        original = deepcopy(self.proposal)
        for field, value in (
            ('support_quote', ''), ('support_quote', 'The blue basket holds four apples.'),
            ('quote_sha256', 'wrong'), ('source_sha256', 'wrong'),
        ):
            with self.subTest(field=field):
                self.proposal = deepcopy(original)
                pos = self.proposal['pairs'][0]['supported']
                pos[field] = value
                if field == 'support_quote':
                    pos['quote_sha256'] = digest(value.encode())
                self.save()
                with self.assertRaises(ValueError):
                    self.build()

    def test_category_action_and_evidence_contracts_are_enforced(self):
        original = deepcopy(self.proposal)
        for kind, field, value in (
            ('supported', 'category', 'out_of_material'), ('supported', 'action', 'abstain'),
            ('supported', 'evidence', ['WRONG']), ('supported', 'gold_facts', []),
            ('unsupported', 'category', 'normal'), ('unsupported', 'action', 'answer'),
            ('unsupported', 'evidence', ['PY-01-01']), ('unsupported', 'gold_facts', ['Invented']),
        ):
            with self.subTest(kind=kind, field=field):
                self.proposal = deepcopy(original)
                self.proposal['pairs'][0][kind]['labels'][field] = value
                self.save()
                with self.assertRaises(ValueError):
                    self.build()

    def test_paths_must_stay_inside_expected_roots(self):
        outside = self.root / 'outside.json'
        outside.write_text(self.proposal_path.read_text())
        with self.assertRaises(ValueError):
            probe.build_items(outside, self.review_path, root=self.root)
        self.proposal['pairs'][0]['supported']['source_path'] = 'outside.json'
        self.proposal['pairs'][0]['supported']['source_sha256'] = probe.sha(outside)
        self.save()
        with self.assertRaises(ValueError):
            self.build()

    def test_build_only_writes_hashed_diagnostic_without_collection(self):
        original_build = probe.build_items
        argv = ['jh_kb_contrast_probe', '--proposal', str(self.proposal_path),
                '--review', str(self.review_path), '--name', 'synthetic',
                '--variant', next(iter(probe.VARIANTS)), '--build-only']
        with (patch.object(probe, 'ROOT', self.root), patch.object(
            probe, 'build_items', side_effect=lambda proposal, review: original_build(
                proposal, review, root=self.root)
        ), patch.object(probe, 'run_variant', side_effect=AssertionError('No API permitted')) as collect,
            patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO())):
            probe.main()
        collect.assert_not_called()
        dest = self.root / 'experiments/.cache/jh_kb_contrast_synthetic'
        dataset = dest / 'derived_inputs.jsonl'
        metadata = json.loads((dest / 'derivation.json').read_text())
        self.assertEqual(metadata['count'], 4)
        self.assertEqual(metadata['dataset_sha256'], probe.sha(dataset))
        self.assertEqual(len(dataset.read_text().splitlines()), 4)
        self.assertEqual(set(p.name for p in dest.iterdir()), {'derived_inputs.jsonl', 'derivation.json'})


if __name__ == '__main__':
    unittest.main()
