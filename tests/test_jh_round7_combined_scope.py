import unittest
from unittest.mock import patch
from core.schema import Label
from pipelines import load
load('jh_cm_char_mlp50')
from pipelines.kimjaehak.classifier_matrix import CharMLP50
from pipelines.kimjaehak.round7_adaptive_ground import AdaptiveGround
from pipelines.kimjaehak.round7_batch import BatchedBlend
from pipelines.kimjaehak.round7_combined import BlendedAdaptive
from pipelines.kimjaehak.round7_material_scope import MaterialScope,ScopedConcrete,MATERIAL_SCOPE_RULE
from pipelines.kimjaehak.plain_abstain import ABSTAIN
from tests import test_jh_reasoning_ablation as reasoning
from tests import test_jh_round7_batch as batch
from tests import test_jh_round7_answer as answer


class CombinedTests(unittest.IsolatedAsyncioTestCase):
    pipeline=reasoning.ReasoningTests.pipeline
    state=reasoning.ReasoningTests.state

    async def test_combined_actual_mro_keeps_dense_classifier_batch_and_adaptive_votes(self):
        fixture=batch.BatchTests();base=fixture.pipeline()
        p=BlendedAdaptive.__new__(BlendedAdaptive);p.__dict__.update(base.__dict__)
        s=fixture.state(p)
        self.assertEqual(p.classifier_kind,'dense_mlp')
        self.assertEqual(p.classifier_threshold,BatchedBlend.classifier_threshold)
        await p._search(s,'검색용 후보 답변')
        await p._classify(s)
        self.assertEqual(p.llm.embed.await_count,1)
        self.assertTrue(s.debug['r7_embedding_reused'])
        self.assertTrue(s.debug['matrix_gate'])
        self.assertEqual(len(fixture.classifier_vectors),1)
        self.assertIs(BlendedAdaptive._fusion_ground,AdaptiveGround._fusion_ground)
        self.assertIs(BlendedAdaptive.tutor,CharMLP50.tutor)
        self.assertIs(BlendedAdaptive._generate,CharMLP50._generate)
        self.assertIs(load('jh_r7_blend_adaptive'),BlendedAdaptive)

    async def test_only_tutor_material_prompt_changes(self):
        for guard in (True,False):
            for name in ('material','block','other'):
                p,q=self.pipeline(MaterialScope),self.pipeline(CharMLP50)
                await p._ask(self.state(p,guard),name,'original','payload','low',7)
                await q._ask(self.state(q,guard),name,'original','payload','low',7)
                c,b=p.llm.chat.await_args,q.llm.chat.await_args
                self.assertEqual(c.args[1:],b.args[1:])
                self.assertEqual(c.args[0],b.args[0]+(MATERIAL_SCOPE_RULE if not guard and name=='material' else ''))
                self.assertAlmostEqual(c.kwargs['timeout'],b.kwargs['timeout'],delta=.03)
                self.assertEqual({k:v for k,v in c.kwargs.items() if k!='timeout'},
                                 {k:v for k,v in b.kwargs.items() if k!='timeout'})

    async def test_scope_telemetry_survives_parent_debug_replacement(self):
        p=self.pipeline(MaterialScope);s=self.state(p,False)
        async def classify(instance,state):
            state._r7_material_scope_used=True;state.debug={'parent':True}
        with patch.object(CharMLP50,'_classify',new=classify):await p._classify(s)
        self.assertEqual(s.debug,{'parent':True,'r7_material_scope_used':True})

    def test_only_explicit_missing_material_can_change_tutor_action(self):
        fixture=answer.AnswerTests()
        for reason in ('missing_material','missing_question_values','unsafe',''):
            p,s=fixture.ready(ScopedConcrete);old=s.answer
            p._attach_explanation(s,{'supported':False,'reason':reason,'answer':''})
            if reason=='missing_material':
                self.assertIs(s.label,Label.OUT_OF_MATERIAL)
                self.assertEqual(s.answer,ABSTAIN)
                self.assertEqual(s.citations,[])
                self.assertTrue(s.debug['r7_apply_abstained'])
            else:
                self.assertIs(s.label,Label.NORMAL)
                self.assertEqual(s.answer,old)
                self.assertEqual(s.citations,['A'])
        p,s=fixture.ready(ScopedConcrete)
        with self.assertRaises(ValueError):p._attach_explanation(s,{'supported':False,'reason':'missing_material','answer':None})
        self.assertIs(s.label,Label.NORMAL)


if __name__=='__main__':unittest.main()
