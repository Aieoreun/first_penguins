import asyncio
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from core.schema import Label
from core.state import Chunk
from pipelines.kimjaehak.focused_excerpt import Selected
from pipelines.kimjaehak.normalized_tutor import Pipeline, normalize_category_query


class NormalizationTests(unittest.TestCase):
    def test_only_standalone_markers_and_prose_spaces_change_without_nfkc(self):
        self.assertEqual(normalize_category_query('ㅋㅋㅋ 금리   4.2%는  아니야 ㅠㅠ'), '금리 4.2%는 아니야')
        self.assertEqual(normalize_category_query('ㅎㅎ 질문 ㅜㅜ'), '질문')
        self.assertEqual(normalize_category_query('  ㅋㅋㅋ 질문  ㅠㅠ  '), '질문')
        for text in ('금리ㅋㅋ비교', '힘들어요ㅠㅠ', 'abc_ㅋㅋㅋ', 'ㅋ 질문',
                     'ㅋㅋㅋ, 질문', 'ＡＢＣ １２３', 'ㅋㅋㅋ"문자"', '`code`ㅠㅠ'):
            self.assertEqual(normalize_category_query(text), text)

    def test_empty_or_marker_only_keeps_original_low_signal(self):
        for text in ('', '  ', 'ㅋㅋㅋ', 'ㅠㅠ', ' ㅋㅋㅋ  ㅎㅎ ㅠㅠ ㅜㅜ '):
            self.assertEqual(normalize_category_query(text), text)

    def test_literals_urls_and_identifiers_are_exactly_preserved(self):
        protected = ['```python\nx = "ㅋㅋㅋ"\nif x != 0:\n    print(x)\n```',
                     '~~~\nㅋㅋㅋ  ㅠㅠ\n~~~', '`x = "ㅠㅠ"`', '"ㅋㅋㅋ   ㅠㅠ"',
                     "'ㅎㅎ  ㅜㅜ'", '“ㅋㅋㅋ  ㅠㅠ”', '‘ㅎㅎ  ㅜㅜ’',
                     'https://example.org/ㅋㅋㅋ?q=ㅠㅠ', 'www.example.org/ㅎㅎ',
                     'ㅋㅋㅋ@example.org', '아이디: ㅋㅋㅋ', 'ID=ㅠㅠ',
                     '전화번호: 010-1234-5678', '+82 (10)  1234-5678',
                     '학번: 202601', '계좌번호: 123-456-789', 'abc_ㅋㅋㅋ']
        for value in protected:
            with self.subTest(value=value):
                self.assertEqual(normalize_category_query('질문 ㅋㅋㅋ ' + value + ' ㅠㅠ'), '질문 ' + value)

    def test_code_lines_indentation_and_unclosed_literals_are_preserved(self):
        for text in ('x = "ㅋㅋㅋ"  # ㅠㅠ', 'if x != 0:  # ㅎㅎ', '    x = "ㅋㅋㅋ"', '  x = "ㅋㅋㅋ"',
                     'def f():\n    return "ㅠㅠ"', '```python\nㅋㅋㅋ  ㅠㅠ',
                     '"ㅋㅋㅋ  ㅠㅠ', '`ㅋㅋㅋ  ㅠㅠ', '"""ㅋㅋㅋ\n  ㅠㅠ'):
            with self.subTest(text=text):
                self.assertEqual(normalize_category_query(text), text)


class NormalizedTutorTests(unittest.IsolatedAsyncioTestCase):
    def make_case(self, guard=False, confidence=.99):
        p = Pipeline.__new__(Pipeline)
        p.vectorizer = Mock(transform=Mock(side_effect=lambda texts: texts))
        p.classifier = SimpleNamespace(classes_=np.array(['normal', 'off_topic']),
                                      predict_proba=Mock(return_value=np.array([[confidence, 1-confidence]])))
        p.section_vectorizer = Mock(transform=Mock(side_effect=lambda texts: texts))
        p.section_classifier = SimpleNamespace(classes_=np.array(['S', 'T']),
                                              predict_proba=Mock(return_value=np.array([[.6, .4]])))
        p.harmful_vectorizer = Mock(transform=Mock(side_effect=lambda texts: texts))
        p.harmful_classifier = SimpleNamespace(classes_=np.array(['refuse', 'redirect']),
                                              predict_proba=Mock(return_value=np.array([[.99, .01]])))
        p.guard_vectorizer = Mock(transform=Mock(side_effect=lambda texts: texts))
        p.guard_classifier = SimpleNamespace(classes_=np.array(['normal', 'off_topic']),
                                            predict_proba=Mock(return_value=np.array([[.6, .4]])))
        p._known_sections = {'S', 'T'}
        p.llm = SimpleNamespace(chat=AsyncMock(return_value='{"category":"none","self_info":false,"quote":"material"}'))
        body = {'messages': [{'role': 'user', 'content': '앞 문맥 ㅎㅎ'},
                             {'role': 'assistant', 'content': '이전 답'},
                             {'role': 'user', 'content': 'ㅋㅋㅋ 금리  4.2% 질문 ㅠㅠ'}]}
        s = p._new_state(body, guard)
        s.chunks = [Chunk('S', 'material')]
        return p, s

    async def test_real_category_chain_normalizes_only_category_and_retains_state(self):
        p, s = self.make_case()
        before = deepcopy(s.messages)
        original_vectorizer = p.vectorizer
        await p._classify(s)
        self.assertEqual(s.label, Label.NORMAL)
        for args in p.vectorizer.transform.call_args_list:
            self.assertEqual(args.args[0], ['금리 4.2% 질문'])
        for args in p.section_vectorizer.transform.call_args_list:
            self.assertEqual(args.args[0], [before[-1]['content']])
        self.assertEqual(s.messages, before)
        self.assertIs(p.vectorizer, original_vectorizer)
        self.assertFalse(s._normalized_tutor_guard)
        p.harmful_vectorizer.transform.assert_not_called()
        p.llm.chat.assert_not_awaited()

    async def test_section_gates_keep_original_query_and_thresholds(self):
        for confidence, marker in [(.99, 'learned_section'), (.91, 'tutor_section_gate')]:
            p, s = self.make_case()
            p.section_classifier.predict_proba.return_value = np.array([[confidence, 1-confidence]])
            await p._classify(s)
            self.assertTrue(s.debug[marker])
            self.assertEqual(s.section, 'S')
            for args in p.section_vectorizer.transform.call_args_list:
                self.assertEqual(args.args[0], [s.last_user])

    async def test_harmful_subtype_model_receives_original_text(self):
        p, s = self.make_case()
        p.classifier.classes_ = np.array(['harmful_content', 'normal'])
        await p._classify(s)
        self.assertEqual(s.label, Label.HARMFUL_REQUEST)
        p.harmful_vectorizer.transform.assert_called_once_with([s.last_user])
        p.vectorizer.transform.assert_called_once_with(['금리 4.2% 질문'])

    async def test_guard_marker_survives_debug_replacement_and_fallback_is_raw(self):
        for clear_debug in (False, True):
            p, s = self.make_case(guard=True)
            if clear_debug:
                s.debug = {}
            await p._classify(s)
            self.assertTrue(s._normalized_tutor_guard)
            self.assertTrue(s._section_gate_guard)
            for args in p.vectorizer.transform.call_args_list:
                self.assertEqual(args.args[0], [s.last_user])
            if not clear_debug:
                p.guard_vectorizer.transform.assert_called_once_with([s.last_user])

    async def test_llm_fallback_payload_and_search_queries_use_original_messages(self):
        p, s = self.make_case(confidence=.6)
        before = deepcopy(s.messages)
        await p._classify(s)
        self.assertGreater(p.llm.chat.await_count, 0)
        for args in p.llm.chat.await_args_list:
            self.assertIn(s.last_user, args.args[1])
            self.assertIn(before[0]['content'], args.args[1])
        self.assertEqual(s.messages, before)
        # Exercise the unchanged search method against an equivalent baseline.
        p.searcher = SimpleNamespace(search=Mock(return_value=s.chunks), has_embeddings=False)
        baseline = Selected.__new__(Selected)
        baseline.searcher = SimpleNamespace(search=Mock(return_value=s.chunks), has_embeddings=False)
        await p._search(s)
        await baseline._search(s)
        self.assertEqual(p.searcher.search.call_args, baseline.searcher.search.call_args)
        self.assertIn(s.last_user, p.searcher.search.call_args.args[0])

    async def test_concurrent_tutor_and_guard_do_not_swap_shared_models(self):
        p, tutor = self.make_case()
        guard = p._new_state({'messages': tutor.messages}, True)
        original = p.vectorizer
        entered = asyncio.Event()
        release = asyncio.Event()
        views = []

        async def parent(instance, state):
            views.append((state._normalized_tutor_guard, instance, instance.vectorizer))
            instance.vectorizer.transform([state.last_user])
            if len(views) == 2:
                entered.set()
            await release.wait()
            self.assertIs(p.vectorizer, original)
            instance.vectorizer.transform([state.last_user])

        with patch.object(Selected, '_classify', new=parent):
            tasks = [asyncio.create_task(p._classify(s)) for s in (tutor, guard)]
            try:
                await asyncio.wait_for(entered.wait(), 1)
                self.assertIs(p.vectorizer, original)
                release.set()
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        for is_guard, instance, view in views:
            self.assertIs(instance.classifier, p.classifier)
            self.assertIs(instance.section_vectorizer, p.section_vectorizer)
            self.assertIs(instance.harmful_vectorizer, p.harmful_vectorizer)
            self.assertEqual(instance is p, is_guard)
            self.assertEqual(view is original, is_guard)
        observed = [args.args[0][0] for args in original.transform.call_args_list]
        self.assertEqual(observed.count('금리 4.2% 질문'), 2)
        self.assertEqual(observed.count(tutor.last_user), 2)

    def test_unrelated_methods_and_settings_remain_inherited(self):
        for method in ('guard', '_search', '_generate', '_check_output', '_prepare_within', '__init__'):
            self.assertIs(getattr(Pipeline, method), getattr(Selected, method))
        for setting in ('learned_gate_c', 'learned_gate_threshold', 'deadline', 'prepare_limit'):
            self.assertEqual(getattr(Pipeline, setting), getattr(Selected, setting))


if __name__ == '__main__':
    unittest.main()
