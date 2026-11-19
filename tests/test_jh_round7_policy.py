import unittest
from core.schema import Label
from pipelines import load
load('jh_cm_char_mlp50')
from pipelines.kimjaehak.classifier_matrix import CharMLP50
from pipelines.kimjaehak.round7_policy import ProductContext, OtherBoundary, ScaffoldGuide
from tests import test_jh_reasoning_ablation as reasoning


class PolicyTests(unittest.IsolatedAsyncioTestCase):
    pipeline = reasoning.ReasoningTests.pipeline
    state = reasoning.ReasoningTests.state

    async def test_input_prompts_only_change_the_target_tutor_check(self):
        for cls, target in ((ProductContext, 'block'), (OtherBoundary, 'other')):
            for guard in (True, False):
                for name in ('block', 'other', 'material'):
                    p, q = self.pipeline(cls), self.pipeline(CharMLP50)
                    await p._ask(self.state(p,guard),name,'original','payload','low',7)
                    await q._ask(self.state(q,guard),name,'original','payload','low',7)
                    c, b = p.llm.chat.await_args, q.llm.chat.await_args
                    self.assertEqual(c.args[1:],b.args[1:])
                    if not guard and name == target:
                        self.assertTrue(c.args[0].startswith(b.args[0]))
                        self.assertNotEqual(c.args[0],b.args[0])
                    else:
                        self.assertEqual(c.args,b.args)
                    self.assertEqual({k:v for k,v in c.kwargs.items() if k!='timeout'},
                                     {k:v for k,v in b.kwargs.items() if k!='timeout'})
                    self.assertAlmostEqual(c.kwargs['timeout'],b.kwargs['timeout'],delta=.03)

    async def test_scaffold_prompt_reaches_actual_generation_only_for_seeking(self):
        original = dict(CharMLP50.answer_systems)
        for label in (Label.ANSWER_SEEKING, Label.ACADEMIC_DISHONESTY):
            p,q=self.pipeline(ScaffoldGuide),self.pipeline(CharMLP50)
            for obj in (p,q):
                state=self.state(obj,False);state.label=label
                obj.llm.chat.return_value='자료의 조건을 확인하고 적용해 보세요.'
                await obj._generate(state)
            c,b=p.llm.chat.await_args,q.llm.chat.await_args
            self.assertEqual(c.kwargs,b.kwargs)
            self.assertEqual(c.args[1:],b.args[1:])
            self.assertEqual(c.args[0]==b.args[0],label is not Label.ANSWER_SEEKING)
        self.assertEqual(CharMLP50.answer_systems,original)

    def test_existing_candidates_and_guard_structure_preserved(self):
        self.assertIs(load('jh_cm_char_mlp50'),CharMLP50)
        for name,cls in (('jh_r7_product',ProductContext),('jh_r7_other',OtherBoundary),('jh_r7_scaffold',ScaffoldGuide)):
            self.assertIs(load(name),cls)
            for method in ('guard','_classify','_fusion_safety','_fusion_ground','_search'):
                self.assertIs(getattr(cls,method),getattr(CharMLP50,method))
            for attr in ('classifier_threshold','guard_deadline','deadline','answer_effort','answer_max_tokens'):
                self.assertEqual(getattr(cls,attr),getattr(CharMLP50,attr))


if __name__=='__main__':unittest.main()
