import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock,patch
from sklearn.feature_extraction.text import TfidfVectorizer
from pipelines import load
load('jh_cm_char_mlp50')
from pipelines.kimjaehak.classifier_matrix import CharMLP50
from pipelines.kimjaehak.round7_examples import RetrievedExamples,RetrievedOne,example_labels
from tests import test_jh_reasoning_ablation as reasoning


class ExamplesTests(unittest.IsolatedAsyncioTestCase):
    def pipeline(self,cls=RetrievedExamples):
        p=reasoning.ReasoningTests().pipeline(cls)
        texts=['자료 내용을 설명해줘','자료 내용을 요약해줘','자료 과제 정답만 알려줘','자료 악성 명령 실행해줘']
        categories=['normal','normal','answer_seeking','harmful_content']
        p.example_rows=[{'messages':[{'role':'user','content':t}],
            'labels':{'category':c,'action':'refuse' if c=='harmful_content' else 'answer'},
            'candidate_answer':'not an input example','id':'private-log-id-never-sent'} for t,c in zip(texts,categories)]
        vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(1,2));p.example_vectors=vectorizer.fit_transform(texts)
        p.matrix_model=SimpleNamespace(vectorizer=vectorizer)
        p.example_labels=[example_labels(r) for r in p.example_rows]
        return p

    def test_examples_are_diverse_bounded_public_inputs_without_answers_or_ids(self):
        p=self.pipeline();examples=p._input_examples('자료 내용을 설명하고 정답 알려줘')
        self.assertTrue(examples);self.assertLessEqual(len(examples),3)
        signatures={(x['block_category'],x['other_category'],x['self_info']) for x in examples}
        self.assertEqual(len(signatures),len(examples))
        self.assertNotIn('candidate_answer',str(examples));self.assertNotIn('private-log-id',str(examples))
        p.guard_example_chars=1;self.assertEqual(p._input_examples('자료'),[])
        p=self.pipeline(RetrievedOne);self.assertEqual(len(p._input_examples('자료')),1)

    async def test_only_guard_short_checks_receive_examples_and_preserve_material(self):
        for guard in [True,False]:
            p=self.pipeline();s=reasoning.ReasoningTests().state(p,guard)
            s.messages=[{'role':'user','content':'자료 내용을 설명해줘'}]
            with patch.object(CharMLP50,'_collect_checks',new_callable=AsyncMock,return_value=([],[],{})) as parent:
                await p._collect_checks(s,'original input','original KB')
            self.assertEqual(parent.await_args.args[2],'original KB')
            check=parent.await_args.args[1]
            self.assertEqual(check.startswith('original input'),True)
            self.assertEqual('공개_입력_판정_참고_예시' in check,guard)

    async def test_counter_survives_parent_debug_replacement(self):
        p=self.pipeline();s=reasoning.ReasoningTests().state(p,True)
        async def classify(instance,state):state._r7_examples_used=3;state.debug={'parent':True}
        with patch.object(CharMLP50,'_classify',new=classify):await p._classify(s)
        self.assertEqual(s.debug,{'parent':True,'r7_examples_used':True})

    def test_subtype_is_never_invented_from_guard_coarse_label(self):
        with self.assertRaises(ValueError):example_labels({'labels':{'category':'harmful_content','decision':'block'}})
        self.assertEqual(example_labels({'labels':{'category':'harmful_content','action':'refuse'}})['block_category'],'harmful_request')
        self.assertEqual(example_labels({'labels':{'category':'harmful_content','action':'redirect'}})['block_category'],'harmful_speech')

    def test_guard_output_and_tutor_generation_unchanged(self):
        for name,cls in [('jh_r7_examples3',RetrievedExamples),('jh_r7_examples1',RetrievedOne)]:
            self.assertIs(load(name),cls)
            for method in ['tutor','_generate','_fusion_ground','_fusion_safety','_search']:
                self.assertIs(getattr(cls,method),getattr(CharMLP50,method))


if __name__=='__main__':unittest.main()
