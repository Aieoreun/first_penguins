import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.jh_paired_scores import compare


class PairedScoreTests(unittest.TestCase):
    def test_paired_delta_and_shared_source_clusters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            rows=[{'id':'a','source_id':'parent'},{'id':'b','source_id':'parent'},{'id':'c','source_id':'other'}]
            runs=[]
            for name,scores in [('base',[0,50,100]),('new',[25,75,100])]:
                run=root/'experiments/.cache'/name/'run';run.mkdir(parents=True);runs.append(run)
                (run.parent/'derived_inputs.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
                (run/'derived_result.json').write_text(json.dumps({'tutor':round(sum(scores)/3,2),'judge_errors':0,'items':[{'id':r['id'],'points':s,'endpoint':'tutor'} for r,s in zip(rows,scores)]}))
            with patch('experiments.jh_paired_scores.ROOT',root):
                result=compare(*runs,repetitions=100)
                self.assertEqual(result['clusters'],2)
                self.assertEqual((result['wins'],result['losses'],result['ties']),(2,0,1))
                self.assertAlmostEqual(result['mean_paired_delta'],50/3)
                p=runs[1]/'derived_result.json';d=json.loads(p.read_text());d['judge_errors']=1;p.write_text(json.dumps(d))
                with self.assertRaisesRegex(ValueError,'Judge errors'):
                    compare(*runs,repetitions=100)
