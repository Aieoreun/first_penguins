import asyncio
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from experiments import jh_generate_grounded_tasks as g

MATERIAL = '# AA-01-01 조건 알아보기\n\n## 두 방식\n조건이 참이면 문을 열어요. 반면 거짓이면 문을 닫아요.\n'
QUESTION = '문 열기 연습에서 조건의 참과 거짓에 따라 동작은 어떻게 달라지나요?'
FACT = '문 열기 연습에서는 조건이 참이면 문을 열고 거짓이면 문을 닫아요.'


class SupportTests(unittest.TestCase):
    def test_complete_contiguous_paragraph_and_deterministic_choice(self):
        text = '# Title\n\n앞 문장이에요. 소개 문장이에요.\n\n'+MATERIAL.split('## 두 방식\n')[1]+'\n다른 문장이에요. 다른 설명이에요.\n'
        unit = g.support_unit(text)
        self.assertEqual(unit['quote'], '조건이 참이면 문을 열어요. 반면 거짓이면 문을 닫아요.')
        self.assertEqual(text[unit['start']:unit['end']], unit['quote'])
        self.assertEqual(unit, g.support_unit(text))
        self.assertTrue(8 <= len(unit['quote']) <= 350)

    def test_fences_tables_and_overlong_paragraphs_are_not_cut(self):
        for text in ('# Title\n\n```python\n조건이면 출력해요. 반면 아니면 멈춰요.\n```\n',
                     '# Title\n\n~~~text\n조건이면 출력해요. 반면 아니면 멈춰요.\n~~~\n',
                     '# Title\n\n## 표\n조건,동작\n참이면,문 열기\n거짓이면,문 닫기\n',
                     '# Title\n\n|조건|동작|\n|--|--|\n|참이면|문 열기|\n',
                     '# Title\n\n조건,동작\n참이면,문 열기\n거짓이면,문 닫기\n',
                     '# Title\n\n'+'조건이면 문을 열어요. '*40):
            self.assertIsNone(g.support_unit(text),text[:60])
        broken = '# Title\n\n```python\nunfinished\n\n조건이면 문을 열어요. 반면 아니면 닫아요.'
        self.assertIsNone(g.support_unit(broken))
        self.assertIsNone(g.support_unit('# Title\n\n짧은 소개예요.\n'))

    def test_prose_commas_and_tables_after_prose(self):
        text = MATERIAL+'\n## 표\n조건,결과\n참이면,출력\n'
        self.assertEqual(g.support_unit(text)['quote'],g.support_unit(MATERIAL)['quote'])
        self.assertIsNotNone(g.support_unit('# Title\n\n조건이 참이면, 문을 열어요. 반면 거짓이면 닫아요.\n'))

    def test_fixed_quote_and_error_categories(self):
        unit = g.support_unit(MATERIAL)
        raw = json.dumps({'question':QUESTION,'answer_fact':FACT},ensure_ascii=False)
        examples,error = g.parse_example(raw,'AA-01-01',MATERIAL,unit)
        self.assertIsNone(error)
        self.assertEqual(examples,[{'question':QUESTION,'answer_fact':FACT,'quote':unit['quote']}])
        self.assertEqual(g.valid_examples(examples,'AA-01-01',MATERIAL),examples)
        cases=[('', 'empty'),('bad JSON','parse'),('[]','validation'),
               ('{"question":"", "answer_fact":""}','empty'),
               (json.dumps({'question':QUESTION,'answer_fact':FACT,'quote':'model rewrite'}),'validation'),
               (json.dumps({'question':QUESTION,'answer_fact':[]}), 'validation'),
               (json.dumps({'question':'위 문장의 차이는 무엇인가요?','answer_fact':FACT}), 'validation')]
        for raw,expected in cases:
            self.assertEqual(g.parse_example(raw,'AA-01-01',MATERIAL,unit),([],expected))

    def test_manifest_resume_rejects_model_style_prompt_and_support_changes(self):
        sections={'AA-01-01':MATERIAL};units=g.plan_support(sections)
        meta=g.make_manifest(sections,g.digest(sections),units,'openai/gpt-5-nano')
        with self.assertRaises(ValueError):g.make_manifest(sections,'kb',units,'another-model')
        with TemporaryDirectory() as temporary:
            dest=Path(temporary)/'new'
            g.initialize_cache(dest,meta,units)
            g.initialize_cache(dest,meta,units,True)
            with self.assertRaises(ValueError):g.initialize_cache(dest,meta,units)
            for key in ('model','question_style','prompt_sha256','kb_sha256','support_units_sha256','generator_sha256','validator_sha256'):
                with self.assertRaises(ValueError):g.initialize_cache(dest,{**meta,key:'different'},units,True)
            with self.assertRaises(ValueError):g.initialize_cache(dest,meta,{},True)


class OfflineCollectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_mock_generation_payload_budget_workers_and_resume_without_retries(self):
        sections={f'AA-01-{i:02}':MATERIAL for i in range(7)}
        sections['AA-02-00']='# Short\n\n소개예요.'
        units=g.plan_support(sections)
        seen=[];active=0;maximum=0
        async def chat(system,user,**kwargs):
            nonlocal active,maximum
            active+=1;maximum=max(maximum,active)
            seen.append((system,json.loads(user),kwargs))
            await asyncio.sleep(0)
            active-=1
            return json.dumps({'question':QUESTION,'answer_fact':FACT},ensure_ascii=False)
        llm=SimpleNamespace(chat=AsyncMock(side_effect=chat))
        with TemporaryDirectory() as temporary:
            dest=Path(temporary)
            await g.generate(sections,units,dest,set(),llm)
            rows=[json.loads(line) for line in (dest/'facts.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),8)
            self.assertEqual(sum(len(r['examples']) for r in rows),7)
            self.assertEqual(maximum,4)
            self.assertEqual(llm.chat.await_count,7)
            saved=(dest/'facts.jsonl').read_bytes()
            completed=g.read_completed(dest,sections,units)
            await g.generate(sections,units,dest,completed,llm)
            self.assertEqual(saved,(dest/'facts.jsonl').read_bytes())
            self.assertEqual(llm.chat.await_count,7)
        for system,payload,kwargs in seen:
            self.assertEqual(system,g.SYSTEM)
            self.assertEqual(set(payload),{'section_title','heading','support_quote'})
            self.assertNotIn('AA-01',payload['section_title'])
            self.assertNotIn('gold_facts',payload)
            self.assertNotIn('question',payload)
            self.assertEqual(kwargs,{'effort':'low','max_tokens':2000,'timeout':25,'json_mode':True})

    async def test_all_failures_are_recorded_and_not_retried(self):
        for reply,error in [('', 'empty'),('bad', 'parse'),('{}','validation'),(TimeoutError(),'api')]:
            sections={'AA-01-01':MATERIAL};units=g.plan_support(sections)
            llm=SimpleNamespace(chat=AsyncMock())
            if isinstance(reply,Exception):llm.chat.side_effect=reply
            else:llm.chat.return_value=reply
            with TemporaryDirectory() as temporary:
                dest=Path(temporary)
                await g.generate(sections,units,dest,set(),llm)
                row=json.loads((dest/'facts.jsonl').read_text())
                self.assertEqual(row['error'],error)
                self.assertEqual(row['examples'],[])
                completed=g.read_completed(dest,sections,units)
                await g.generate(sections,units,dest,completed,llm)
                self.assertEqual(llm.chat.await_count,1)

    async def test_cancel_propagates_without_recording_fake_completion(self):
        sections={'AA-01-01':MATERIAL};units=g.plan_support(sections)
        llm=SimpleNamespace(chat=AsyncMock(side_effect=asyncio.CancelledError()))
        with TemporaryDirectory() as temporary:
            dest=Path(temporary)
            with self.assertRaises(asyncio.CancelledError):await g.generate(sections,units,dest,set(),llm)
            self.assertFalse((dest/'facts.jsonl').exists())
            self.assertEqual(llm.chat.await_count,1)

    async def test_existing_probe_builder_and_review_accept_row_shape(self):
        import jh_kb_fact_probe as probe
        import jh_review_kb_facts as review
        sections={'AA-01-01':MATERIAL,'AA-01-02':'# Title\n\n소개예요.'};units=g.plan_support(sections)
        meta=g.make_manifest(sections,g.digest(sections),units,'openai/gpt-5-nano')
        llm=SimpleNamespace(chat=AsyncMock(return_value=json.dumps({'question':QUESTION,'answer_fact':FACT})))
        with TemporaryDirectory() as temporary:
            dest=Path(temporary)/'facts';g.initialize_cache(dest,meta,units)
            await g.generate(sections,units,dest,set(),llm)
            with patch.object(probe,'sections_and_hash',return_value=(sections,g.digest(sections))),patch.object(probe,'load_public',return_value=[]):
                items,returned=probe.build_items(dest/'facts.jsonl')
            self.assertEqual(len(items),1)
            self.assertEqual(returned['question_style'],'grounded_relation')
            self.assertEqual(items[0]['label_support_quote'],units['AA-01-01']['quote'])
            review_dir=Path(temporary)/'review';review_dir.mkdir()
            fake_review=SimpleNamespace(chat=AsyncMock(return_value='{"valid":true,"reason":"valid"}'),
                                        _chat=SimpleNamespace(close=AsyncMock()))
            with patch('core.llm.LLM',return_value=fake_review):
                await review.review(items,review_dir,4,set())
            payload=json.loads(fake_review.chat.await_args.args[1])
            self.assertEqual(payload,{'question':QUESTION,'answer_fact':FACT,'support_quote':units['AA-01-01']['quote']})
            self.assertTrue(json.loads((review_dir/'reviews.jsonl').read_text())['valid'])


if __name__=='__main__':
    unittest.main()
