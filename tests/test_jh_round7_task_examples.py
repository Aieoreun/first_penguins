import json
import unittest
from unittest.mock import patch,AsyncMock
from pipelines import load
load('jh_cm_char_mlp50')
from pipelines.kimjaehak.classifier_matrix import CharMLP50
from pipelines.kimjaehak.round7_task_examples import TaskExamples,TaskExampleOne
from tests import test_jh_round7_examples as examples
from tests import test_jh_reasoning_ablation as reasoning


class TaskExampleTests(unittest.IsolatedAsyncioTestCase):
    async def test_specific_expected_schema_before_target_guard_only(self):
        for guard in [True,False]:
            for name in ['block','other','material']:
                p=examples.ExamplesTests().pipeline(TaskExamples)
                s=reasoning.ReasoningTests().state(p,guard)
                s.messages=[{'role':'user','content':'자료 내용을 설명해줘'}]
                with patch.object(CharMLP50,'_ask',new_callable=AsyncMock,return_value={}) as parent:
                    await p._ask(s,name,'system','actual target','low',10)
                user=parent.await_args.args[3]
                if guard and name in {'block','other'}:
                    self.assertTrue(user.endswith('actual target'))
                    payload=json.loads(user.split('\n',1)[1].split('\n[현재 판정 대상:',1)[0])
                    for item in payload:
                        self.assertEqual(set(item),{'conversation','decision'})
                        self.assertEqual(set(item['decision']),{'category'} if name=='block' else {'category','self_info'})
                else:self.assertEqual(user,'actual target')
                self.assertEqual(parent.await_args.args[2],'system')

    async def test_collect_does_not_append_second_set_of_examples(self):
        p=examples.ExamplesTests().pipeline(TaskExamples);s=reasoning.ReasoningTests().state(p,True)
        with patch.object(CharMLP50,'_collect_checks',new_callable=AsyncMock,return_value=([],[],{})) as parent:
            await p._collect_checks(s,'original','material')
        # Explicit unbound superclass invocation also receives self.
        self.assertEqual(parent.await_args.args,(p,s,'original','material'))
        self.assertIs(load('jh_r7_task_examples3'),TaskExamples)
        self.assertIs(load('jh_r7_task_examples1'),TaskExampleOne)


if __name__=='__main__':unittest.main()
