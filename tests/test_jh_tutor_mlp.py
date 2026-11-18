import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from core.schema import Label
from pipelines.kimjaehak.classifier_matrix import CharMLP50
from pipelines.kimjaehak.tutor_mlp import Selective, Replace
from pipelines.kimjaehak.voted import Pipeline as Voted
from pipelines.kimjaehak.learned_sections import Pipeline as Sections
from pipelines.kimjaehak.tutor_section_gate import Pipeline as TutorSections


class TutorMLPTests(unittest.IsolatedAsyncioTestCase):
    def make(self, cls=Selective, category='answer_seeking', confidence=.99, guard=False, multi=False):
        p=cls.__new__(cls)
        p.matrix_model=SimpleNamespace(classes=np.array([category,'normal']),
            probabilities=Mock(return_value=np.array([[confidence,1-confidence]])))
        messages=[{'role':'user','content':'조건을 설명해주세요'}]
        if multi: messages=[{'role':'user','content':'앞 질문'},{'role':'assistant','content':'앞 답변'}]+messages
        state=p._new_state({'messages':messages},guard)
        p._apply_learned_section=Mock()
        p._apply_tutor_section=Mock()
        return p,state

    async def test_guard_delegates_without_tutor_inference(self):
        for cls in (Selective,Replace):
            p,s=self.make(cls,guard=True)
            with patch.object(CharMLP50,'_classify',new_callable=AsyncMock) as original:
                await p._classify(s);original.assert_awaited_once_with(s)
            p.matrix_model.probabilities.assert_not_called()
            self.assertEqual(p.classifier_threshold,CharMLP50.classifier_threshold)

    async def test_selected_tutor_runs_both_source_stages_once_in_order(self):
        for cls in (Selective,Replace):
            p,s=self.make(cls)
            calls=[]
            p._apply_learned_section.side_effect=lambda s:calls.append('learned')
            p._apply_tutor_section.side_effect=lambda s:calls.append('tutor')
            with patch.object(CharMLP50,'_classify',new_callable=AsyncMock) as original:
                await p._classify(s);original.assert_not_awaited()
            self.assertEqual(s.label,Label.ANSWER_SEEKING)
            self.assertEqual(calls,['learned','tutor'])
            self.assertTrue(s.debug['tutor_mlp_gate'])

    async def test_uncertain_paths_are_distinct(self):
        for cls in (Selective,Replace):
            p,s=self.make(cls,confidence=.6)
            with patch.object(CharMLP50,'_classify',new_callable=AsyncMock) as original, patch.object(Voted,'_classify',new_callable=AsyncMock) as llm:
                await p._classify(s)
                if cls is Selective:
                    original.assert_awaited_once_with(s);llm.assert_not_awaited()
                    p._apply_tutor_section.assert_not_called()
                else:
                    original.assert_not_awaited();llm.assert_awaited_once_with(p,s)
                    p._apply_tutor_section.assert_called_once_with(s)

    async def test_material_harmful_and_low_signal_preserve_original(self):
        for cls in (Selective,Replace):
            for category in ('normal','out_of_material','harmful_content','low_signal_input'):
                p,s=self.make(cls,category=category)
                with patch.object(CharMLP50,'_classify',new_callable=AsyncMock) as original:
                    await p._classify(s);original.assert_awaited_once_with(s)
                p._apply_tutor_section.assert_not_called()

    async def test_context_scope_and_invalid_probability(self):
        p,s=self.make(multi=True)
        with patch.object(CharMLP50,'_classify',new_callable=AsyncMock) as original:
            await p._classify(s);original.assert_awaited_once_with(s)
        p.matrix_model.probabilities.assert_not_called()
        p,s=self.make(Replace,multi=True)
        await p._classify(s);self.assertTrue(s.debug['tutor_mlp_gate'])
        p,s=self.make();p.matrix_model.probabilities.return_value=np.array([[float('nan'),0]])
        with patch.object(CharMLP50,'_classify',new_callable=AsyncMock) as original:
            await p._classify(s);original.assert_awaited_once_with(s)
        self.assertEqual(s.errors,['tutor_mlp:ValueError'])

    async def test_concurrent_endpoint_delegation_does_not_change_guard_model(self):
        p,t=self.make(Replace)
        g=p._new_state({'messages':[{'role':'user','content':'가드 질문'}]},True)
        model=p.matrix_model;threshold=p.classifier_threshold
        with patch.object(CharMLP50,'_classify',new_callable=AsyncMock) as original:
            await asyncio.gather(p._classify(t),p._classify(g))
            original.assert_awaited_once_with(g)
        self.assertIs(p.matrix_model,model);self.assertEqual(p.classifier_threshold,threshold)

    def test_real_source_helpers_and_generation_are_inherited(self):
        self.assertIs(Replace._apply_learned_section,Sections._apply_learned_section)
        self.assertIs(Replace._apply_tutor_section,TutorSections._apply_tutor_section)
        for method in ('guard','tutor','_generate','_search','_fusion_ground','_fusion_safety'):
            self.assertIs(getattr(Replace,method),getattr(CharMLP50,method))
