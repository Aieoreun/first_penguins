import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from pipelines import load
load('jh_cm_char_mlp50')
from pipelines.kimjaehak.classifier_matrix import CharMLP50
from pipelines.kimjaehak.round7_binary_guard import BinaryGuard
from pipelines.kimjaehak.round7_binary_model import BinaryAdjustedModel,BLOCKED
from core.schema import POLICY

class BinaryGuardTests(unittest.TestCase):
    def test_decision_head_adjusts_mass_and_keeps_class_identity(self):
        category=SimpleNamespace(classes=np.array(['normal','off_topic']),probabilities=Mock(return_value=np.array([[.8,.2]])))
        decision=SimpleNamespace(classes=np.array(['allow','block']),probabilities=Mock(return_value=np.array([[.1,.9]])))
        model=BinaryAdjustedModel(category,decision)
        np.testing.assert_allclose(model.probabilities(['question']),[[.1,.9]])
        self.assertIs(model.classes,category.classes)
        decision.probabilities.assert_called_once_with(['question'])

    def test_policy_block_mapping_and_unmodified_methods(self):
        self.assertEqual(BLOCKED,{p.category.value for p in POLICY.values() if p.decision.value=='block'})
        self.assertIs(load('jh_r7_binary_guard'),BinaryGuard)
        for method in ['tutor','guard','_classify','_generate','_fusion_ground','_fusion_safety','_search']:
            self.assertIs(getattr(BinaryGuard,method),getattr(CharMLP50,method))

    def test_degenerate_group_rejected_for_original_fallback(self):
        category=SimpleNamespace(classes=np.array(['normal','off_topic']),probabilities=Mock(return_value=np.array([[1.,0.]])))
        decision=SimpleNamespace(classes=np.array(['allow','block']),probabilities=Mock(return_value=np.array([[.1,.9]])))
        with self.assertRaises(ValueError):BinaryAdjustedModel(category,decision).probabilities(['x'])

if __name__=='__main__':unittest.main()
