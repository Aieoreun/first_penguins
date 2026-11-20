import unittest
from unittest.mock import Mock,patch
from pipelines import load
load('jh_r7_binary_guard')
from pipelines.kimjaehak.round7_binary_guard import BinaryGuard
from pipelines.kimjaehak.round8_category import BinaryBoundary,BinaryBoundaryBudgets,BinaryDense
from pipelines.kimjaehak.neural_peace_guard import InputBudgets

class CategoryTests(unittest.TestCase):
    def test_boundary_retains_exact_binary_model(self):
        decision=Mock();category=Mock()
        def initialize(obj,*args):obj.matrix_model=Mock(decision_model=decision)
        with patch.object(BinaryGuard,'__init__',initialize),patch('pipelines.kimjaehak.round8_category.public_data',return_value=([{'labels':{'category':'normal'}}],['text'],None,None)),patch('pipelines.kimjaehak.round8_category.BoundaryModel') as factory:
            factory.return_value.fit.return_value=category
            decision.classes=['allow','block']
            p=BinaryBoundary(None,None)
            self.assertIs(p.matrix_model.decision_model,decision)
            self.assertIs(p.matrix_model.category_model,category)

    def test_paths_and_registration(self):
        for name,cls in [('jh_r8_binary_boundary',BinaryBoundary),('jh_r8_binary_boundary_budgets',BinaryBoundaryBudgets),('jh_r8_binary_dense',BinaryDense)]:
            self.assertIs(load(name),cls)
            for method in ['tutor','_generate','_classify','_search','_fusion_safety']:
                self.assertIs(getattr(cls,method),getattr(BinaryGuard if method!='_fusion_safety' or cls is not BinaryBoundaryBudgets else InputBudgets,method))
        self.assertIs(BinaryBoundary.guard,BinaryGuard.guard)
        self.assertIs(BinaryBoundaryBudgets.guard,InputBudgets.guard)
        self.assertEqual(BinaryDense.classifier_kind,'dense_mlp')

if __name__=='__main__':unittest.main()
