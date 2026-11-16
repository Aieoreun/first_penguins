import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from jh_tutor_generation_probe import reuse_journal


class GenerationJournalTests(unittest.TestCase):
    def test_only_completed_matching_sources_can_supply_judgments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); cache=root/'experiments/.cache';reference=cache/'prior';dest=cache/'new'
            reference.mkdir(parents=True);dest.mkdir()
            metadata={'scope':'component','source':'own','source_response_sha256':'abc','evaluator_hashes':{'judge':'same'},'source_manifest':{'model':'provided'}}
            (reference/'manifest.json').write_text(json.dumps(metadata))
            (reference/'judge.jsonl').write_text('{"key":"one","result":{"ok":true}}\n')
            with patch('jh_tutor_generation_probe.ROOT',root):
                with self.assertRaisesRegex(ValueError,'incomplete'):
                    reuse_journal(reference,dict(metadata),dest)
                (reference/'result.json').write_text('{}')
                for key,value in [('source_response_sha256','different'),('evaluator_hashes',{}),('source_manifest',{'model':'other'})]:
                    changed={**metadata,key:value}
                    with self.assertRaisesRegex(ValueError,'differs'):
                        reuse_journal(reference,changed,dest)
                    self.assertFalse((dest/'judge.jsonl').exists())
                reused=dict(metadata);reuse_journal(reference,reused,dest)
                self.assertEqual((dest/'judge.jsonl').read_bytes(),(reference/'judge.jsonl').read_bytes())
                self.assertEqual(reused['judge_reused_from'],'experiments/.cache/prior')
                self.assertEqual(len(reused['judge_reused_sha256']),64)
