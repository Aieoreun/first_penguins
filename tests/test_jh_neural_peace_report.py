import unittest
from copy import deepcopy
from pathlib import Path
from experiments.jh_neural_peace_safety_data import load
from experiments.jh_neural_peace_report import safety_metrics


class SafetyDiagnosticTests(unittest.TestCase):
    def test_cross_pairs_preserve_public_inputs_and_balance_safety(self):
        items,meta=load(Path(__file__).resolve().parents[1])
        self.assertEqual(len(items),16)
        self.assertFalse(meta['training_allowed'])
        self.assertFalse(meta['deployment_allowed'])
        self.assertEqual(meta['primary_metric'],'safe_accuracy')
        for control,unsafe in zip(items[::2],items[1::2]):
            self.assertEqual(control['messages'],unsafe['messages'])
            self.assertEqual(control['labels']['decision'],'allow')
            self.assertEqual(unsafe['labels']['category'],'normal')
            self.assertTrue(control['labels']['safe'])
            self.assertFalse(unsafe['labels']['safe'])
            self.assertNotEqual(control['candidate_answer'],unsafe['candidate_answer'])

    def test_misses_false_alarms_and_invalid_are_separate(self):
        items,_=load(Path(__file__).resolve().parents[1])
        response={'input':{'decision':'allow','category':'normal'},
                  'output':{'grounded':True,'unsupported':[],'safe':True}}
        rows=[{'id':x['id'],'response':deepcopy(response)} for x in items]
        rows[0]['response']['output']['safe']=False
        rows[2]['response']=None
        counts=safety_metrics(items,rows)['groups']['gold_normal']
        self.assertEqual(counts['gold_unsafe'],8)
        self.assertEqual(counts['miss_unsafe_as_safe'],8)
        self.assertEqual(counts['false_alarm_safe_as_unsafe'],1)
        self.assertEqual(counts['invalid'],1)
        self.assertEqual(counts['correct'],6)


if __name__=='__main__': unittest.main()
