import unittest
import numpy as np
from core.schema import Label
from experiments.jh_round7_gate_replay import gate_label


class ReplayTests(unittest.TestCase):
    def test_gate_uses_probability_cutoff_and_harmful_subtype(self):
        classes=np.array(['normal','harmful_content'])
        text='교재의 조건을 설명해 주세요'
        self.assertEqual(gate_label(text,classes,[.9,.1],.85),Label.NORMAL)
        self.assertIsNone(gate_label(text,classes,[.8,.2],.85))
        self.assertIsNone(gate_label(text,classes,[.01,.99],.85,('refuse',.94)))
        self.assertEqual(gate_label(text,classes,[.01,.99],.85,('refuse',.99)),Label.HARMFUL_REQUEST)
        self.assertEqual(gate_label(text,classes,[.01,.99],.85,('redirect',.99)),Label.HARMFUL_SPEECH)
        self.assertEqual(gate_label('ㅋㅋㅋ',classes,[.01,.99],.85),Label.LOW_SIGNAL_INPUT)

    def test_invalid_probabilities_cannot_be_counted_as_a_confident_gate(self):
        for p in ([np.nan,0],[-.1,1.1],[1,1],[1]):
            with self.assertRaises(ValueError):gate_label('교재 설명',np.array(['normal','off_topic']),p,.8)


if __name__=='__main__':unittest.main()
