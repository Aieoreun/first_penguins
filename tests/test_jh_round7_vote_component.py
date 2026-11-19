import unittest
from pipelines import load
load('jh_cm_char_mlp50')
from experiments.jh_round7_vote_component import selected_votes


def votes(values,order=None):
    order=list(range(5)) if order is None else order
    return [{'index':i,'completed':True,'seconds':order.index(i)+1,'errors':[],
             'unsupported':v} for i,v in enumerate(values)]


class VoteReplayTests(unittest.TestCase):
    def test_two_agreement_and_three_disagreement_are_distinct(self):
        rows=votes([['bad'],['bad'],[],[],[]])
        selected,early=selected_votes('bad text',rows,'first2');self.assertEqual(len(selected),2);self.assertTrue(early)
        selected,early=selected_votes('bad text',rows,'first3');self.assertEqual(len(selected),5);self.assertFalse(early)

    def test_completion_order_controls_early_but_creation_order_controls_consensus(self):
        rows=votes([['bad'],[],['bad'],[],['bad']],[4,2,0,1,3])
        selected,early=selected_votes('bad text',rows,'early3')
        self.assertEqual([r['index'] for r in selected],[0,2,4]);self.assertTrue(early)

    def test_unfinished_and_failed_votes_do_not_count_toward_agreement(self):
        rows=votes([['bad']]*5);rows[0]['completed']=False;rows[0]['seconds']=None;rows[1]['errors']=['failed']
        for mode in ['first2','first3','early3','all5']:
            selected,early=selected_votes('bad text',rows,mode)
            self.assertEqual([r['index'] for r in selected],[1,2,3,4]);self.assertFalse(early)

    def test_agreement_uses_nonspace_covered_positions(self):
        rows=votes([['bad text'],['bad','text'],[],[],[]])
        selected,early=selected_votes('bad text',rows,'first2');self.assertTrue(early);self.assertEqual(len(selected),2)


if __name__=='__main__':unittest.main()
