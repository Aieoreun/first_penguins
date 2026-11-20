import unittest
from pipelines import load
load('jh_r7_binary_guard')
from pipelines.kimjaehak.round7_binary_guard import BinaryGuard
from pipelines.kimjaehak.round8_guard import BinaryParallelSafety,BinaryCheckBudgets,BinaryInputBudgets
from pipelines.kimjaehak.neural_peace_guard import ParallelSafety,CheckBudgets,InputBudgets

class GuardCompositionTests(unittest.TestCase):
    def test_exact_guard_paths_and_unchanged_tutor(self):
        for name,cls,parent in [('jh_r8_binary_parallel',BinaryParallelSafety,ParallelSafety),('jh_r8_binary_timeouts',BinaryCheckBudgets,CheckBudgets),('jh_r8_binary_inputbudgets',BinaryInputBudgets,InputBudgets)]:
            self.assertIs(load(name),cls);self.assertIs(cls.__init__,BinaryGuard.__init__)
            self.assertIs(cls._classify,BinaryGuard._classify)
            self.assertIs(cls._fusion_safety,parent._fusion_safety)
            self.assertIs(cls.guard,parent.guard)
            for method in ['tutor','_generate','_search']:
                self.assertIs(getattr(cls,method),getattr(BinaryGuard,method))
            self.assertEqual(cls.classifier_threshold,BinaryGuard.classifier_threshold)
            self.assertEqual(cls.guard_deadline,28)
            self.assertTrue(cls.guard_cancel_ground_after_block)
        self.assertFalse(BinaryParallelSafety.guard_safety_after_input)
        self.assertTrue(BinaryParallelSafety.guard_safety_launch_early)
        self.assertTrue(BinaryInputBudgets.guard_safety_after_input)
        self.assertIs(BinaryCheckBudgets._vote_group,CheckBudgets._vote_group)

if __name__=='__main__':unittest.main()
