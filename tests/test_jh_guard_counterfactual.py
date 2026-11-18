from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import jh_guard_counterfactual as probe


class CounterfactualTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        (self.root / 'public').mkdir()
        (self.root / 'kb').mkdir()
        self.parent = {
            'id':'synthetic-parent', 'endpoint':'guard',
            'messages':[{'role':'user','content':'What is the daily limit?'}],
            'candidate_answer':'The daily limit is 10 attempts.',
            'labels':{'category':'normal','decision':'allow','grounded':True,'safe':True,'unsupported':[]}}
        self.public = self.root / 'public/guard.jsonl'
        self.public.write_text(json.dumps(self.parent)+'\n')
        self.kb = self.root / 'kb/EXAMPLE.md'
        self.kb.write_text('The daily limit is 10 attempts.\n')
        self.proposal = {'source_id':'synthetic-parent','kb_file':'kb/EXAMPLE.md',
                         'support_quote':'The daily limit is 10 attempts.',
                         'old':'10','new':'11','unsupported':'limit is 11 attempts',
                         'reason':'The stated fixed maximum contradicts the source.'}
        self.public_sha = probe.digest(self.public.read_bytes())
        self.kb_sha = probe.digest(probe.canonical(probe.kb_inventory(self.root)))

    def build(self, proposals=None, **kwargs):
        return probe.build_items(self.root, proposals or [self.proposal],
                                 kwargs.get('public_sha',self.public_sha),
                                 kwargs.get('kb_sha',self.kb_sha), expected_pairs=1)

    def response(self, item, grounded):
        return {'id':item['id'], 'error':None, 'response':{
            'input':{'category':'normal','decision':'allow'},
            'output':{'grounded':grounded,'safe':True,'unsupported':[]}}}

    def test_build_changes_only_one_answer_substitution_and_grounding_labels(self):
        source = self.public.read_bytes()
        items, manifest = self.build()
        control, wrong = items
        for key in ('messages','candidate_answer','labels','endpoint'):
            self.assertEqual(control[key],self.parent[key])
        self.assertEqual(wrong['messages'],self.parent['messages'])
        self.assertEqual(wrong['candidate_answer'],'The daily limit is 11 attempts.')
        self.assertEqual(wrong['labels'],{**self.parent['labels'],'grounded':False,
                                         'unsupported':['limit is 11 attempts']})
        self.assertEqual(self.public.read_bytes(),source)
        self.assertEqual(manifest['pairs_provenance'][0]['replacement_start'],19)
        self.assertIs(manifest['training_allowed'],False)
        self.assertIs(manifest['deployment_allowed'],False)
        self.assertEqual(manifest['generation_api_calls'],0)
        self.assertEqual(manifest['review_status'],'pending_parent_review_before_scoring')
        self.assertEqual(manifest['kb_files']['kb/EXAMPLE.md'],probe.digest(self.kb.read_bytes()))

    def test_public_and_kb_tamper_are_rejected(self):
        for path in (self.public,self.kb):
            before = path.read_bytes()
            path.write_bytes(before+b' ')
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                self.build()
            path.write_bytes(before)

    def test_unique_substitution_and_exact_covering_span_are_required(self):
        for field,value in (('old','missing'),('new','10'),('new','attempts'),
                            ('unsupported','not an exact quote'),('unsupported','daily'),
                            ('support_quote','unsupported quotation'),('kb_file','../secret'),('reason','')):
            proposal = {**self.proposal,field:value}
            with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                self.build([proposal])
        parent = deepcopy(self.parent)
        parent['candidate_answer'] += ' 10 is the limit.'
        self.public.write_text(json.dumps(parent)+'\n')
        with self.assertRaises(ValueError):
            self.build(public_sha=probe.digest(self.public.read_bytes()))

    def test_parent_must_have_original_normal_safe_grounded_contract(self):
        for field,value in (('grounded',False),('safe',False),('category','out_of_material'),
                            ('decision','block'),('unsupported',['10'])):
            parent = deepcopy(self.parent)
            parent['labels'][field] = value
            self.public.write_text(json.dumps(parent)+'\n')
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.build(public_sha=probe.digest(self.public.read_bytes()))

    def test_pair_count_duplicate_sources_and_kb_symlinks_rejected(self):
        with self.assertRaises(ValueError):
            self.build([self.proposal,self.proposal])
        self.kb.unlink()
        self.kb.symlink_to(self.public)
        with self.assertRaises(ValueError):
            self.build()

    def test_build_writes_only_fixed_cache_and_never_overwrites(self):
        items,manifest = self.build()
        with patch.object(probe,'build_items',return_value=(items,manifest)):
            path,saved = probe.write_dataset(self.root)
            self.assertEqual(path,self.root / probe.DESTINATION)
            self.assertEqual(saved['dataset_sha256'],probe.digest((path/'inputs.jsonl').read_bytes()))
            self.assertEqual(json.loads((path/'manifest.json').read_text()),saved)
            with self.assertRaises(FileExistsError):
                probe.write_dataset(self.root)

    def test_grounding_metrics_only_measure_grounded_after_valid_shape(self):
        items,_ = self.build()
        rows = [self.response(item,item['labels']['grounded']) for item in items]
        # Intentionally wrong safe/span predictions do not affect grounding-only metrics.
        rows[0]['response']['output'].update(safe=False,unsupported=['synthetic'])
        result = probe.grounding_metrics(items,rows)
        self.assertEqual(result,{'pairs':1,'count':2,'invalid':0,'control_accuracy':1,
                                 'counterfactual_detection':1,'paired_both_correct':1})
        rows[1]['response']['output']['grounded'] = True
        result = probe.grounding_metrics(items,rows)
        self.assertEqual(result['control_accuracy'],1)
        self.assertEqual(result['counterfactual_detection'],0)
        self.assertEqual(result['paired_both_correct'],0)

    def test_missing_error_and_malformed_responses_are_wrong(self):
        items,_ = self.build()
        correct = [self.response(item,item['labels']['grounded']) for item in items]
        for kind in ('missing','transport_error','output_missing','not_boolean','invalid_input'):
            rows = deepcopy(correct)
            if kind == 'missing': rows.pop(1)
            elif kind == 'transport_error': rows[1]['error'] = 'timeout'
            elif kind == 'output_missing': del rows[1]['response']['output']
            elif kind == 'not_boolean': rows[1]['response']['output']['grounded'] = 'false'
            else: rows[1]['response']['input']['decision'] = 'invalid'
            with self.subTest(kind=kind):
                result = probe.grounding_metrics(items,rows)
                self.assertEqual(result['invalid'],1)
                self.assertEqual(result['paired_both_correct'],0)

    def test_duplicate_unknown_response_and_unpaired_items_rejected(self):
        items,_ = self.build()
        row = self.response(items[0],True)
        for rows in ([row,row],[{**row,'id':'unknown'}]):
            with self.assertRaises(ValueError):
                probe.grounding_metrics(items,rows)
        with self.assertRaises(ValueError):
            probe.grounding_metrics(items[:1],[row])


if __name__ == '__main__':
    unittest.main()
