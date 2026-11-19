import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from jh_topic_switch_probe import transform, followup


class TopicSwitchTests(unittest.TestCase):
    def test_final_question_and_labels_are_preserved_without_mutating_source(self):
        rows=[]
        sections={}
        for sid in ('PY-01-01','PY-01-02','FIN-01-01','FIN-01-02'):
            quote='교재의 내용 '+sid.replace('-','_')
            sections[sid]=quote
            rows.append({'id':sid,'endpoint':'tutor','messages':[{'role':'user','content':'질문 '+sid}],
                         'labels':{'category':'normal','action':'answer','evidence':[sid],'gold_facts':['사실 '+sid]},
                         'label_support_quote':quote})
        old=copy.deepcopy(rows)
        output=transform(rows,sections)
        self.assertEqual(rows,old)
        self.assertEqual(len(output),8)
        self.assertEqual(len({r['id'] for r in output}),8)
        originals={r['id']:r for r in rows}
        for item in output:
            source=originals[item['source_id']]
            context=originals[item['context_source_id']]
            self.assertEqual(item['labels'],source['labels'])
            self.assertTrue(item['messages'][-1]['content'].endswith(source['messages'][0]['content']))
            self.assertNotEqual(source['labels']['evidence'],context['labels']['evidence'])
            self.assertEqual(item['messages'][1]['content'],context['label_support_quote'])

    def test_unsupported_context_is_rejected(self):
        row={'id':'a','endpoint':'tutor','messages':[{'role':'user','content':'q'}],
             'labels':{'category':'normal','evidence':['PY-01-01'],'gold_facts':['x']},'label_support_quote':'missing'}
        with self.assertRaises(ValueError):transform([row],{'PY-01-01':'other'})

    def test_followup_keeps_question_in_history_without_revealing_answer(self):
        row={'id':'example','endpoint':'tutor','messages':[{'role':'user','content':'예금 기간은 얼마인가요?'}],
             'labels':{'category':'normal','action':'answer','evidence':['FIN-01-01'],
                       'gold_facts':['기간은 12개월이다.']},'label_support_quote':'기간은 12개월이다.'}
        original=copy.deepcopy(row)
        output=followup([row],{'FIN-01-01':row['label_support_quote']})
        self.assertEqual(row,original)
        self.assertEqual(len(output),2)
        for item in output:
            self.assertEqual(item['messages'][0],row['messages'][0])
            self.assertEqual(item['labels'],row['labels'])
            self.assertNotIn('12개월',str(item['messages']))
            self.assertNotIn('FIN-01-01',str(item['messages']))
        self.assertNotEqual(output[0]['id'],output[1]['id'])
