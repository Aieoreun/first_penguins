import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from jh_tutor_generation_probe import section_changes


class SourceSelectionProbeTests(unittest.TestCase):
    def test_selection_uses_own_action_citation_and_predictions_not_diagnostic_labels(self):
        class ForbiddenLabels(dict):
            def __getitem__(self, key):raise AssertionError('Diagnostic gold was read')
            def get(self, key, default=None):raise AssertionError('Diagnostic gold was read')
        items=[{'id':'a','messages':[{'role':'user','content':'학습 질문'}], 'labels':ForbiddenLabels()}]
        prior={'a':{'error':None,'response':{'action':'answer','citations':['old']}}}
        vec=SimpleNamespace(transform=Mock(return_value='own query vector'))
        model=SimpleNamespace(classes_=np.array(['old','new']),predict_proba=Mock(return_value=np.array([[.08,.92]])))
        self.assertEqual(section_changes(items,prior,vec,model,{'old','new'}),
                         {'a':{'old':'old','new':'new','confidence':.92}})
        vec.transform.assert_called_once_with(['학습 질문'])
        prior['a']['response']['action']='abstain'
        self.assertEqual(section_changes(items,prior,vec,model,{'old','new'}),{})
