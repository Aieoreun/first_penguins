import json
import time
import unittest
from pipelines import load
load('jh_cm_char_mlp50')
from pipelines.kimjaehak.classifier_matrix import CharMLP50
from pipelines.kimjaehak.round7_binary_guard import BinaryGuard
from pipelines.kimjaehak.round7_tax_scope import TaxScopedConcrete,TaxBudget12
from pipelines.kimjaehak.round7_binary_tutor import BinaryTutor,BinaryTutor12
from tests import test_jh_round7_answer as answer_tests

class BinaryTutorTests(unittest.IsolatedAsyncioTestCase):
    def test_factorial_paths_share_exact_methods(self):
        for name,cls in [('jh_r7_binary_tutor',BinaryTutor),('jh_r7_binary_tutor12',BinaryTutor12)]:
            self.assertIs(load(name),cls)
            self.assertIs(cls.__init__,BinaryGuard.__init__)
            self.assertIs(cls._explain,TaxScopedConcrete._explain)
            self.assertEqual(cls.classifier_threshold,BinaryGuard.classifier_threshold)
            for method in ['guard','_classify','_fusion_ground','_fusion_safety','_search']:
                self.assertIs(getattr(cls,method),getattr(BinaryGuard,method))
        self.assertEqual(BinaryTutor.tutor_apply_timeout,8)
        self.assertEqual(BinaryTutor12.tutor_apply_timeout,12)
        self.assertIs(TaxBudget12._classify,CharMLP50._classify)

    async def test_twelve_second_call_still_respects_original_total_deadline(self):
        for remaining,limit in [(20,12),(5,4.5)]:
            p,s=answer_tests.AnswerTests().ready(TaxBudget12)
            s.messages=[{'role':'user','content':'100원의 이자를 계산해줘'}]
            s._source_extract_deadline=time.monotonic()+remaining
            p.llm.chat.return_value=json.dumps({'supported':True,'reason':'','answer':'계산 설명'})
            await p._explain(s)
            p.llm.chat.assert_awaited_once()
            self.assertLessEqual(p.llm.chat.await_args.kwargs['timeout'],limit)
            self.assertEqual(p.llm.chat.await_args.kwargs['max_tokens'],2000)

if __name__=='__main__':unittest.main()
