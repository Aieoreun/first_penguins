import unittest
import numpy as np
from experiments.jh_round7_binary_cv import reweight

class BinaryReweightTests(unittest.TestCase):
    def test_probability_mass_moves_between_groups_without_reordering_within(self):
        classes=np.array(['normal','off_topic','jailbreak_injection','out_of_material'])
        p=np.array([[.4,.1,.2,.3]])
        out=reweight(p,classes,np.array([.8]),1)
        self.assertAlmostEqual(out[0,1]+out[0,2],.8)
        self.assertAlmostEqual(out.sum(),1)
        self.assertAlmostEqual(out[0,1]/out[0,2],.5)
        self.assertAlmostEqual(out[0,0]/out[0,3],4/3)
        np.testing.assert_allclose(reweight(p,classes,np.array([.8]),0),p)

    def test_bad_probabilities_rejected(self):
        with self.assertRaises(ValueError):reweight([[.4,.6]],['normal','off_topic'],[float('nan')],1)
        with self.assertRaises(ValueError):reweight([[.4,.6]],['normal','pii_disclosure'],[.5],1)

if __name__=='__main__':unittest.main()
