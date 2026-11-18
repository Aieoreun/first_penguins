from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import jh_guard_fusion_datasets as datasets


def guard(identifier='g', candidate=True):
    row = {'id':identifier, 'endpoint':'guard', 'messages':[{'role':'user','content':'Synthetic question'}],
           'labels':{'category':'normal','decision':'allow'}}
    if candidate:
        row['candidate_answer'] = 'Synthetic unsupported detail.'
        row['labels'].update(grounded=False, safe=True, unsupported=['unsupported detail'])
    return row


class DatasetTests(unittest.TestCase):
    def test_alternative_filters_by_parent_not_rewritten_endpoint(self):
        parents = {'g': guard('g'), 't': {**guard('t', False), 'endpoint': 'tutor'}}
        rows = [{**guard('g-p1', False), 'source_id': 'g'},
                {**guard('t-p1', False), 'source_id': 't'}]
        spec = {'raw_count': 2, 'count': 1, 'candidate_count': 0}
        items = datasets.validate_rows(rows, spec, 'alternative', parents)
        self.assertEqual(items, rows[:1])
        self.assertIs(items[0], rows[0])
        self.assertNotIn('candidate_answer', items[0])
        with self.assertRaises(ValueError):
            datasets.validate_rows(rows, spec, 'alternative')
        for field, value in (('candidate_answer', 'Do not restore original answer'),
                             ('labels', {'category':'normal','decision':'allow','safe':True}),
                             ('source_id', 'missing'), ('messages', [])):
            broken = deepcopy(rows)
            broken[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                datasets.validate_rows(broken, spec, 'alternative', parents)

    def test_alternative_rejects_label_drift_and_repeated_parent(self):
        parents = {'g': guard('g')}
        rows = [{**guard('g-p1', False), 'source_id':'g'}]
        spec = {'raw_count':1,'count':1,'candidate_count':0}
        for field, value in (('category','off_topic'), ('decision','block')):
            broken = deepcopy(rows)
            broken[0]['labels'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                datasets.validate_rows(broken, spec, 'alternative', parents)
        with self.assertRaises(ValueError):
            datasets.validate_rows(rows + [{**rows[0], 'id':'g-p2'}],
                                   {'raw_count':2,'count':2,'candidate_count':0}, 'alternative', parents)

    def test_alternative_summary_never_reports_output_or_aggregate_score(self):
        result = {'guard':99, 'guard_details':{'block_f1':.9, 'category_accuracy':.8,
                  'false_block':2, 'missed_block':3, 'safe_accuracy':None}, 'runtime':{}}
        metadata = {'dataset':'alternative', 'scope':'input-only diagnostic',
                    'primary_metric':'input_only', 'interpretation':'No output labels'}
        summary = datasets.summarize_result('fusion', result, metadata)
        self.assertEqual(set(summary['input_metrics']),
                         {'block_f1','category_accuracy','false_block','missed_block'})
        self.assertNotIn('guard', summary)
        self.assertNotIn('guard_details', summary)
        self.assertEqual(summary['interpretation'], 'No output labels')

    def test_paraphrase_filters_guard_without_replacing_candidate_or_labels(self):
        rows = [guard('g1'), {'id':'t','endpoint':'tutor'}, guard('g2', False)]
        spec = {'raw_count':3,'count':2,'candidate_count':1}
        items = datasets.validate_rows(rows, spec, 'paraphrase')
        self.assertEqual(items, [rows[0],rows[2]])
        self.assertIs(items[0], rows[0])
        self.assertEqual(items[0]['labels']['unsupported'], ['unsupported detail'])

    def test_counts_duplicate_ids_and_span_annotation_are_checked(self):
        spec = {'raw_count':2,'count':2,'candidate_count':2}
        for rows in ([guard('a')], [guard('a'),guard('a')]):
            with self.assertRaises(ValueError):
                datasets.validate_rows(rows, spec, 'novel')
        rows = [guard('a'), guard('b')]
        rows[1]['labels']['unsupported'] = ['not in answer']
        with self.assertRaises(ValueError):
            datasets.validate_rows(rows, spec, 'novel')

    def test_refusal_requires_complete_safe_unsafe_pairs_and_forced_grounding(self):
        rows = [guard('p/refusal'), guard('p/disclaimer')]
        for row in rows:
            row['labels'].update(decision='block',grounded=True,unsupported=[],safe=row['id'].endswith('/refusal'))
        spec = {'raw_count':2,'count':2,'candidate_count':2}
        self.assertEqual(datasets.validate_rows(rows, spec, 'refusal'), rows)
        for field, value in (('safe',True),('grounded',False),('unsupported',['unsupported detail'])):
            broken = deepcopy(rows)
            broken[1]['labels'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                datasets.validate_rows(broken, spec, 'refusal')

    def test_refusal_summary_uses_safety_not_artificial_guard_aggregate(self):
        result = {'guard':99, 'guard_details':{'safe_accuracy':.75,'grounded_accuracy':1},'runtime':{}}
        metadata = {'dataset':'refusal','scope':'safety diagnostic','primary_metric':'safe_accuracy',
                    'interpretation':'Forced grounding labels; safety only'}
        result = datasets.summarize_result('fusion', result, metadata)
        self.assertEqual(result['safe_accuracy'], .75)
        self.assertNotIn('guard', result)
        self.assertNotIn('guard_details', result)
        self.assertIn('safety only', result['interpretation'])

    def test_pinned_source_and_manifest_lineage_reject_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'data.jsonl'
            source.write_text(json.dumps(guard())+'\n')
            parent = root / 'parent.jsonl'
            parent.write_text('parent fixture')
            manifest = root / 'manifest.json'
            meta = {'scope':'frozen fixture','count':1,'dataset_sha256':datasets.digest(source.read_bytes()),
                    'source':'parent.jsonl','source_sha256':datasets.digest(parent.read_bytes()),
                    'independent_source_holdout':False}
            manifest.write_text(json.dumps(meta))
            spec = {'path':'data.jsonl','manifest':'manifest.json','manifest_scope':'frozen fixture',
                    'raw_count':1,'count':1,'candidate_count':1,'scope':'manual synthetic diagnostic',
                    'primary_metric':'guard','old_input_predictions_fixed':True,
                    'files':{p.name:datasets.digest(p.read_bytes()) for p in (source,parent,manifest)}}
            with patch.dict(datasets.DATASETS, {'fixture':spec}):
                rows, frozen = datasets.load_dataset('fixture', root)
                self.assertEqual(len(rows),1)
                self.assertIs(frozen['input_predictions_fixed'],False)
                self.assertIs(frozen['historical_component_input_predictions_fixed'],True)
                original = parent.read_bytes()
                parent.write_text('changed')
                with self.assertRaises(ValueError):
                    datasets.load_dataset('fixture', root)
                parent.write_bytes(original)
                meta['scope'] = 'changed even after file hash relink'
                manifest.write_text(json.dumps(meta))
                spec['files']['manifest.json'] = datasets.digest(manifest.read_bytes())
                with self.assertRaises(ValueError):
                    datasets.load_dataset('fixture', root)


if __name__ == '__main__':
    unittest.main()
