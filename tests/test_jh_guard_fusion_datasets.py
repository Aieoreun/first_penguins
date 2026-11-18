from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import jh_guard_fusion_datasets as datasets
from experiments import jh_guard_counterfactual as counterfactual
from experiments import jh_guard_relational_diagnostic as relational


def guard(identifier='g', candidate=True):
    row = {'id':identifier, 'endpoint':'guard', 'messages':[{'role':'user','content':'Synthetic question'}],
           'labels':{'category':'normal','decision':'allow'}}
    if candidate:
        row['candidate_answer'] = 'Synthetic unsupported detail.'
        row['labels'].update(grounded=False, safe=True, unsupported=['unsupported detail'])
    return row


class DatasetTests(unittest.TestCase):
    def test_relational_exact_proposal_regeneration_and_review_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proposals = [{'pair_id':'r','manual':'synthetic reason'}]
            rows = [guard('r/control'),guard('r/counterfactual')]
            base_manifest = {'proposal_sha256':'proposal','kb_manifest_sha256':'kb'}
            manifest = {**base_manifest,'dataset_sha256':datasets.digest(
                ''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows).encode())}
            review = {'scope':'parent_source_review_before_relational_scoring','accepted':True,
                      'new_relational_predictions_observed':False,'training_allowed':False,
                      'deployment_allowed':False,'independent_source_holdout':False,
                      'accepted_pair_ids':['r'],**manifest}
            spec = {'review':'review.json','proposal':'proposal.json'}
            (root/'review.json').write_text(json.dumps(review))
            (root/'proposal.json').write_text(json.dumps(proposals))
            with patch.object(relational,'build',side_effect=lambda root: (
                    deepcopy(proposals),deepcopy(rows),deepcopy(base_manifest))):
                datasets.validate_relational(root,spec,rows,manifest)
                for key,value in (('accepted',False),('accepted_pair_ids',['changed']),
                                  ('dataset_sha256','changed'),('proposal_sha256','changed'),
                                  ('kb_manifest_sha256','changed'),('new_relational_predictions_observed',True)):
                    (root/'review.json').write_text(json.dumps({**review,key:value}))
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        datasets.validate_relational(root,spec,rows,manifest)
                (root/'review.json').write_text(json.dumps(review))
                (root/'proposal.json').write_text(json.dumps([{'pair_id':'r','manual':'changed'}]))
                with self.assertRaises(ValueError):
                    datasets.validate_relational(root,spec,rows,manifest)

    def test_relational_group_metrics_keep_whole_set_weighting_and_invalids(self):
        items,rows = [],[]
        for pair,group in (('code','code_interpretation'),('course1','course_relation'),('course2','course_relation')):
            for kind in ('control','counterfactual'):
                item = guard(pair+'/'+kind)
                item.update(pair_id=pair,contrast_kind=kind,error_group=group)
                item['labels']['grounded'] = kind == 'control'
                items.append(item)
                rows.append({'id':item['id'],'response':{'input':{'category':'normal','decision':'allow'},
                    'output':{'grounded':kind=='control','safe':True,'unsupported':[]}}})
        rows.pop(1)  # Missing code counterfactual must count as wrong in both scopes.
        result = datasets.relational_metrics(items,rows)
        self.assertEqual(result['paired_both_correct'],2/3)
        self.assertEqual(result['invalid'],1)
        self.assertEqual(result['by_group']['code_interpretation']['paired_both_correct'],0)
        self.assertEqual(result['by_group']['course_relation']['paired_both_correct'],1)
        broken = deepcopy(items)
        broken[0]['error_group'] = 'course_relation'
        with self.assertRaises(ValueError): datasets.relational_metrics(broken,rows)
        metadata = {'dataset':'relational','scope':'synthetic','primary_metric':'paired_grounding','interpretation':'dev only'}
        summary = datasets.summarize_result('current',{'runtime':{},'relational_details':result},metadata)
        self.assertEqual(summary['paired_both_correct'],2/3)
        self.assertNotIn('by_group',summary)
        self.assertNotIn('guard',summary)

    def test_relational_pinned_artifacts_rejected_before_any_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = ['inputs.jsonl','proposal.json','manifest.json','review.json']
            for name in paths: (root/name).write_text('synthetic '+name)
            spec = {'files':{name:datasets.digest((root/name).read_bytes()) for name in paths}}
            with patch.dict(datasets.DATASETS,{'relational':spec}),patch.object(relational,'build') as build:
                for name in paths:
                    original = (root/name).read_bytes()
                    (root/name).write_bytes(b'changed')
                    with self.subTest(name=name),self.assertRaises(ValueError):
                        datasets.load_dataset('relational',root)
                    (root/name).write_bytes(original)
                build.assert_not_called()

    def test_counterfactual_regeneration_and_separate_review_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [guard('p/control'),guard('p/counterfactual')]
            base_manifest = {'pairs_provenance':[{'source_id':'p'}],
                             'proposal_sha256':'proposal', 'kb_manifest_sha256':'kb'}
            manifest = {**deepcopy(base_manifest), 'dataset_sha256':datasets.digest(
                ''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows).encode())}
            review = {'scope':'parent_source_review_before_counterfactual_scoring','accepted':True,
                      'new_counterfactual_predictions_observed':False, 'training_allowed':False,
                      'deployment_allowed':False,'independent_source_holdout':False,
                      'accepted_source_ids':['p'], **{key:manifest[key] for key in
                      ('dataset_sha256','proposal_sha256','kb_manifest_sha256')}}
            spec = {'review':'review.json'}
            (root / 'review.json').write_text(json.dumps(review))
            with patch.object(counterfactual,'build_items',side_effect=lambda root: (deepcopy(rows),deepcopy(base_manifest))):
                datasets.validate_counterfactual(root,spec,rows,manifest)
                for key,value in (('accepted',False),('accepted_source_ids',['different']),
                                  ('dataset_sha256','changed'),('proposal_sha256','changed'),
                                  ('kb_manifest_sha256','changed'),('training_allowed',True),
                                  ('deployment_allowed',True),('new_counterfactual_predictions_observed',True)):
                    (root / 'review.json').write_text(json.dumps({**review,key:value}))
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        datasets.validate_counterfactual(root,spec,rows,manifest)
                (root / 'review.json').write_text(json.dumps(review))
                broken = deepcopy(rows)
                broken[0]['candidate_answer'] += ' tampered'
                with self.assertRaises(ValueError):
                    datasets.validate_counterfactual(root,spec,broken,manifest)
                with self.assertRaises(ValueError):
                    datasets.validate_counterfactual(root,spec,rows,{**manifest,'extra':'tampered'})

    def test_counterfactual_review_pin_is_checked_before_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review = root / 'review.json'
            review.write_text('{"accepted":true}')
            spec = {'files':{'review.json':datasets.digest(review.read_bytes())}}
            review.write_text('{"accepted":false}')
            with patch.dict(datasets.DATASETS,{'counterfactual':spec}), \
                 patch.object(counterfactual,'build_items') as rebuild:
                with self.assertRaises(ValueError):
                    datasets.load_dataset('counterfactual',root)
                rebuild.assert_not_called()

    def test_counterfactual_summary_prioritizes_grounding_over_official_aggregate(self):
        result = {'guard':99,'guard_details':{'span_f1':1},'runtime':{},
                  'counterfactual_details':{'count':16,'pairs':8,'invalid':1,
                    'control_accuracy':.875,'counterfactual_detection':.5,'paired_both_correct':.375}}
        metadata = {'dataset':'counterfactual','scope':'manual grounding development',
                    'primary_metric':'paired_grounding','interpretation':'Span is supplementary'}
        summary = datasets.summarize_result('fusion',result,metadata)
        self.assertEqual(summary['paired_both_correct'],.375)
        self.assertEqual(summary['invalid'],1)
        self.assertNotIn('guard',summary)
        self.assertNotIn('guard_details',summary)
        self.assertEqual(result['guard'],99)

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
