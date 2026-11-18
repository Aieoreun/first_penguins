import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from pipelines.kimjaehak.round6_normal_gate import Pipeline
from pipelines.kimjaehak.material_budget import Pipeline as Base


class GateTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_normal_in_new_band_changes_threshold_and_preserves_postprocessing(self):
        p=Pipeline.__new__(Pipeline)
        p.vectorizer=SimpleNamespace(transform=lambda text:text)
        classes=np.array(['normal','emotional_distress'])
        for category,prob in [('normal',.9399),('normal',.94),('normal',.9499),('normal',.95),
                              ('emotional_distress',.945)]:
            values=[prob,1-prob] if category=='normal' else [1-prob,prob]
            p.classifier=SimpleNamespace(classes_=classes,predict_proba=lambda text:np.array([values]))
            seen=[]
            async def classify(view,state):
                seen.append(view.learned_gate_threshold)
                state.debug={'learned_gate':True,'tutor_section_gate':.91}
                state.section='selected by existing postprocessing'
            with patch.object(Base,'_classify',new=classify):
                states=[p._new_state({'messages':[{'role':'user','content':'question'}]},guard)
                        for guard in (False,True)]
                await asyncio.gather(*(p._classify(state) for state in states))
            changed=category=='normal' and .94<=prob<.95
            self.assertEqual(seen,[.94 if changed else .95,.95])
            self.assertEqual(p.learned_gate_threshold,.95)
            for state in states:
                self.assertEqual(state.section,'selected by existing postprocessing')
                self.assertEqual(state.debug['tutor_section_gate'],.91)
            self.assertEqual('tutor_normal_gate_threshold' in states[0].debug,changed)
            self.assertNotIn('tutor_normal_gate_threshold',states[1].debug)

    async def test_guard_does_not_add_model_prediction(self):
        p=Pipeline.__new__(Pipeline)
        state=p._new_state({'messages':[]},True)
        async def classify(view,state): state.debug={}
        with patch.object(Base,'_classify',new=classify): await p._classify(state)
        self.assertFalse(hasattr(p,'classifier'))
        self.assertIs(Pipeline.guard,Base.guard)
        self.assertIs(Pipeline._check_output,Base._check_output)


if __name__=='__main__': unittest.main()
