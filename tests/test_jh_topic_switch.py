import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from jh_topic_switch_probe import transform


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
