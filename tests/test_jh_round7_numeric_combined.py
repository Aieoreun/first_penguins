import unittest
from unittest.mock import AsyncMock,patch
from pipelines import load
load('jh_cm_char_mlp50')
from pipelines.kimjaehak.classifier_matrix import CharMLP50
from pipelines.kimjaehak.round7_numeric_combined import NumericMaterialVerdict,NumericVerdictCompact,NumericVerdictDirect
from pipelines.kimjaehak.round7_compact_calculator import CompactCalculator
from pipelines.kimjaehak.round7_material_scope import ScopedConcrete
from pipelines.kimjaehak.round7_material_verdict import VERDICT_RULE
from tests import test_jh_reasoning_ablation as reasoning


class NumericCombinedTests(unittest.IsolatedAsyncioTestCase):
    async def test_nonnumeric_material_and_every_guard_prompt_remain_exact(self):
        for cls in [NumericMaterialVerdict,NumericVerdictCompact,NumericVerdictDirect]:
            for guard in [True,False]:
                for text,numeric in [('이 개념을 설명해줘',False),('180000원의 이자를 계산해줘',True)]:
                    p=reasoning.ReasoningTests().pipeline(cls);s=reasoning.ReasoningTests().state(p,guard)
                    s.messages=[{'role':'user','content':text}]
                    with patch.object(CharMLP50,'_ask',new_callable=AsyncMock,return_value={'quote':'본문','material_status':'sufficient'}) as parent:
                        await p._ask(s,'material','original','payload','low',10)
                    system=parent.await_args.args[2]
                    if numeric and not guard:self.assertIn(VERDICT_RULE,system)
                    else:self.assertEqual(system,'original')
                    self.assertEqual(parent.await_args.args[3:],('payload','low',10))

    def test_numeric_composition_preserves_output_choice_and_guard(self):
        for name,cls in [('jh_r7_numeric_verdict',NumericMaterialVerdict),('jh_r7_numeric_compact',NumericVerdictCompact),('jh_r7_numeric_direct',NumericVerdictDirect)]:
            self.assertIs(load(name),cls)
            for method in ['guard','_fusion_ground','_fusion_safety','_search']:
                self.assertIs(getattr(cls,method),getattr(CharMLP50,method))
        self.assertIs(NumericVerdictCompact._attach_explanation,CompactCalculator._attach_explanation)
        self.assertIs(NumericVerdictDirect._attach_explanation,ScopedConcrete._attach_explanation)
        self.assertIs(NumericMaterialVerdict._generate,CharMLP50._generate)


if __name__=='__main__':unittest.main()
