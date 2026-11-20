import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock,patch
from pipelines import load
load('jh_r7_binary_guard')
from pipelines.kimjaehak.round7_binary_guard import BinaryGuard
from pipelines.kimjaehak.round7_adaptive_ground import AdaptiveGround
from pipelines.kimjaehak.round8_efficient_guard import BinaryGroundThree,BinaryAdaptiveGround,BinaryOtherMinimal


class EfficientTests(unittest.IsolatedAsyncioTestCase):
    def test_existing_schedules_keep_B_input_and_tutor(self):
        for name,cls in [('jh_r8_binary_ground3',BinaryGroundThree),('jh_r8_binary_adaptive',BinaryAdaptiveGround),('jh_r8_other_minimal',BinaryOtherMinimal)]:
            self.assertIs(load(name),cls)
            self.assertIs(cls._classify,BinaryGuard._classify)
            self.assertIs(cls.tutor,BinaryGuard.tutor)
            self.assertEqual(cls.classifier_threshold,BinaryGuard.classifier_threshold)
            self.assertEqual(cls.guard_deadline,28)
        self.assertEqual(BinaryGroundThree.output_votes,3)
        self.assertIs(BinaryAdaptiveGround._fusion_ground,AdaptiveGround._fusion_ground)
        self.assertEqual(BinaryAdaptiveGround.guard_adaptive_initial_votes,2)

    async def test_minimal_only_changes_guard_other_call(self):
        p=BinaryOtherMinimal.__new__(BinaryOtherMinimal)
        for guard,name,expected in [(True,'other','minimal'),(True,'block','low'),(True,'material','low'),(False,'other','low')]:
            s=SimpleNamespace(_material_budget_guard=guard)
            with patch.object(BinaryGuard,'_ask',new_callable=AsyncMock,return_value={}) as ask:
                await p._ask(s,name,'system','user','low',10)
                self.assertEqual(ask.call_args.args,(s,name,'system','user',expected,10))


if __name__=='__main__':unittest.main()
