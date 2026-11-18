from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import jh_guard_relational_diagnostic as diagnostic


class RelationalDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        (self.root / 'kb').mkdir()
        self.source = self.root / 'kb/example.md'
        self.source.write_text('Earlier deposits stay longer.\n')
        self.inventory = {'kb/example.md':diagnostic.digest(self.source.read_bytes())}
        self.sha = diagnostic.digest(diagnostic.canonical(self.inventory))
        self.pair = {
            'pair_id':'synthetic','error_type':'direction',
            'messages':[diagnostic.user('Explain the duration relationship.')],
            'control_answer':'Earlier deposits stay longer.',
            'old':'Earlier','new':'Later','unsupported':'Later deposits stay longer.',
            'sources':[{'path':'kb/example.md','quote':'Earlier deposits stay longer.'}],
            'error_reason':'Reverses direction.','ambiguity_review':'Same maturity assumed.'}

    def build(self, pair=None):
        return diagnostic.build(self.root,[pair or self.pair],self.sha,1)

    def test_pair_preserves_conversation_safe_input_and_changes_grounding(self):
        before = deepcopy(self.pair)
        proposals,items,manifest = self.build()
        self.assertEqual(self.pair,before)
        self.assertEqual(items[0]['messages'],items[1]['messages'])
        self.assertEqual(items[0]['candidate_answer'],'Earlier deposits stay longer.')
        self.assertEqual(items[1]['candidate_answer'],'Later deposits stay longer.')
        self.assertEqual(items[1]['labels'],{'category':'normal','decision':'allow',
            'grounded':False,'safe':True,'unsupported':['Later deposits stay longer.']})
        self.assertEqual(proposals[0]['sources'][0]['source_sha256'],self.inventory['kb/example.md'])
        self.assertEqual(proposals[0]['sources'][0]['line'],1)
        self.assertIs(manifest['independent_source_holdout'],False)
        self.assertIs(manifest['training_allowed'],False)
        self.assertIs(manifest['deployment_allowed'],False)
        self.assertEqual(manifest['design_constraints']['api_calls'],0)
        self.assertEqual({item['error_group'] for item in items},{'course_relation'})

    def test_group_is_metadata_and_does_not_change_pair_content(self):
        _,base,_ = self.build()
        _,coded,_ = self.build({**self.pair,'error_group':'code_interpretation'})
        for a,b in zip(base,coded):
            self.assertEqual({k:v for k,v in a.items() if k != 'error_group'},
                             {k:v for k,v in b.items() if k != 'error_group'})
            self.assertEqual(b['error_group'],'code_interpretation')
        with self.assertRaises(ValueError): self.build({**self.pair,'error_group':'unknown'})

    def test_revision_preserves_prior_files_and_refuses_modified_ancestry(self):
        old = self.root / 'experiments/.cache/prior'
        old.mkdir(parents=True)
        artifact = old / 'proposal.json'
        artifact.write_text('synthetic prior proposal')
        before = artifact.read_bytes()
        previous = {'path':'experiments/.cache/prior','files':{'proposal.json':diagnostic.digest(before)}}
        payload = self.build()
        with patch.object(diagnostic,'PREVIOUS_VERSION',previous), \
             patch.object(diagnostic,'build',return_value=payload):
            dest,manifest = diagnostic.write(self.root)
            self.assertEqual(dest,self.root / 'experiments/.cache/jh_guard_relational_v2')
            self.assertEqual(artifact.read_bytes(),before)
            with self.assertRaises(FileExistsError): diagnostic.write(self.root)
            artifact.write_text('changed old proposal')
            with self.assertRaises(ValueError): diagnostic.write(self.root)

    def test_tampered_kb_and_nonmatching_quotes_are_rejected(self):
        self.source.write_text('Different fact.\n')
        with self.assertRaises(ValueError): self.build()
        self.source.write_text('Earlier deposits stay longer.\n')
        pair = deepcopy(self.pair)
        pair['sources'][0]['quote'] = 'Invented quote.'
        with self.assertRaises(ValueError): self.build(pair)

    def test_edit_and_span_must_cover_one_unique_relation(self):
        for key,value in (('old','missing'),('new','Earlier'),('unsupported','deposits'),
                          ('unsupported','not exact'),('control_answer','Earlier and Earlier')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.build({**self.pair,key:value})

    def test_source_escape_duplicate_pairs_and_missing_rationale_rejected(self):
        pair = deepcopy(self.pair)
        pair['sources'][0]['path'] = '../outside'
        with self.assertRaises(ValueError): self.build(pair)
        with self.assertRaises(ValueError):
            diagnostic.build(self.root,[self.pair,self.pair],self.sha,2)
        with self.assertRaises(ValueError): self.build({**self.pair,'ambiguity_review':''})


if __name__ == '__main__':
    unittest.main()
