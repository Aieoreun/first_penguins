import asyncio
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from core.kb import KBChunk
from core.llm import LLM
from core.search import Searcher
from experiments import jh_ground_budget_probe as probe
from pipelines.kimjaehak.peace_span_variants import CurrentSpan80


def sdk_result(raw='{"unsupported":[]}', finish='stop'):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,
                           message=SimpleNamespace(content=raw))],
                           usage=SimpleNamespace(prompt_tokens=100, completion_tokens=30,
                               total_tokens=130,
                               completion_tokens_details=SimpleNamespace(reasoning_tokens=20)))


def fake_llm():
    llm = LLM.__new__(LLM)
    llm.model = 'provided-test-model'
    llm.embedding_model = 'provided-test-embedding'
    llm._chat = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(return_value=sdk_result()))), close=AsyncMock())
    llm._embed = SimpleNamespace(close=AsyncMock())
    llm.embed = AsyncMock(return_value=[[1., 0.]])
    return llm


class GroundBudgetTests(unittest.IsolatedAsyncioTestCase):
    def test_frozen_selection_is_balanced_normal_allow_and_order_independent(self):
        rows = probe.public_items()
        selected = probe.select_items(rows)
        self.assertEqual(selected, probe.select_items(list(reversed(rows))))
        self.assertEqual(len({r['id'] for r in selected}), 6)
        self.assertEqual(sum(r['labels']['grounded'] for r in selected), 3)
        for row in selected:
            self.assertEqual(row['labels']['category'], 'normal')
            self.assertEqual(row['labels']['decision'], 'allow')
            body = probe.request_body(row)
            self.assertEqual(set(body), {'spec_version', 'session_id', 'messages', 'candidate_answer'})
        broken = deepcopy(rows)
        broken.append(deepcopy(selected[0]))
        with self.assertRaises(ValueError):
            probe.select_items(broken)

    def test_source_hash_and_prepared_provenance_are_enforced_without_clients(self):
        rows = probe.public_items()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'public').mkdir()
            (root / 'public/guard.jsonl').write_text('changed')
            with patch.object(probe, 'ROOT', root):
                with self.assertRaises(ValueError):
                    probe.public_items()
            frozen = {'prompt_sha256': 'original', 'selection': [r['id'] for r in probe.select_items(rows)]}
            with patch.object(probe, 'ROOT', root), patch.object(probe, 'public_items', return_value=rows), \
                    patch.object(probe, 'provenance', return_value=frozen), \
                    patch.object(probe, 'execute', side_effect=AssertionError('must not execute')), \
                    patch.object(probe, 'LLM', side_effect=AssertionError('must not initialize')):
                with redirect_stdout(io.StringIO()) as stdout:
                    self.assertEqual(probe.main(['--name', 'prepared', '--prepare-only']), 0)
                self.assertEqual(json.loads(stdout.getvalue())['api_calls'], 0)
                dest, _, manifest = probe.prepare('prepared')
                self.assertEqual(manifest['status'], 'prepared')
                manifest['status'] = 'interrupted'
                probe.write_json(dest / 'manifest.json', manifest)
                with self.assertRaises(ValueError):
                    probe.prepare('prepared')
                with self.assertRaises(ValueError):
                    probe.prepare('../escape')
                manifest['status'] = 'prepared'
                manifest['provenance']['prompt_sha256'] = 'changed'
                probe.write_json(dest / 'manifest.json', manifest)
                with self.assertRaises(ValueError):
                    probe.prepare('prepared')

    def test_strict_diagnostic_parser_and_original_parser_compatibility(self):
        for raw in ('{"unsupported":[]}', '{"unsupported":["wrong"]}',
                    '```json\n{"checks":["ignored"],"unsupported":["wrong"]}\n```'):
            self.assertTrue(probe.validate_response(raw, 'a wrong fact')['unsupported_valid'])
        for raw in ('', '{"unsupported":[', '{}', '[]', '{"unsupported":"wrong"}',
                    '{"unsupported":[1]}', '{"unsupported":[""]}', '{"unsupported":["absent"]}'):
            self.assertFalse(probe.validate_response(raw, 'a wrong fact')['unsupported_valid'])

    async def test_sdk_kwargs_equal_core_and_finish_usage_preserved(self):
        llm = fake_llm()
        await llm.chat('system', 'user', effort='medium', max_tokens=2000, timeout=20, json_mode=True)
        expected = llm._chat.chat.completions.create.call_args
        llm._chat.chat.completions.create.return_value = sdk_result('{"unsupported":[', 'length')
        row = await probe.completion(llm, 'system', 'user', 'candidate', 'medium', 2000)
        self.assertEqual(llm._chat.chat.completions.create.call_args, expected)
        self.assertEqual(row['finish_reason'], 'length')
        self.assertEqual(row['usage']['reasoning_tokens'], 20)
        self.assertFalse(row['unsupported_valid'])
        self.assertEqual(llm._chat.chat.completions.create.await_count, 2)

    async def test_search_top_one_frozen_for_both_budgets_actual_mro(self):
        llm = fake_llm()
        chunks = [KBChunk('S1', 'python', 't', 'h', '검색어 파이썬'),
                  KBChunk('GOLD_SENTINEL', 'python', 'other', 'h', '다른 원문')]
        # Real init fits only the existing public models; no API is used by init.
        pipeline = CurrentSpan80(Searcher(chunks, [[1., 0.], [0., 1.]]), llm)
        items = probe.select_items(probe.public_items())
        items = deepcopy(items)
        for item in items:
            item['messages'] = [{'role': 'user', 'content': '검색어 파이썬'}]
            item['candidate_answer'] = '공개 후보'
            item['labels']['evidence'] = ['GOLD_SENTINEL']
        before = deepcopy(items)
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()) as stdout:
            dest = Path(temp)
            manifest = {'status': 'prepared'}
            await probe.collect(items, pipeline, dest, manifest)
            self.assertEqual(llm.embed.await_count, 6)
            calls = llm._chat.chat.completions.create.call_args_list
            self.assertEqual(len(calls), 12)
            for i in range(0, 12, 2):
                users = [calls[i+j].kwargs['messages'][1]['content'] for j in range(2)]
                self.assertEqual(len(set(users)), 1)
                payload = json.loads(users[0])
                self.assertEqual([m['section'] for m in payload['교재']], ['S1'])
                self.assertNotIn('GOLD_SENTINEL', users[0])
                self.assertEqual([calls[i+j].kwargs['max_completion_tokens'] for j in range(2)],
                                 [1500, 2000])
            self.assertEqual(manifest['api_calls'], 18)
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(len(manifest['retrievals']), 6)
            self.assertTrue(manifest['retrievals'][0]['source_chunks'][0]['chunk_id'].startswith('S1:'))
            self.assertNotIn('공개 후보', stdout.getvalue())
            self.assertNotIn('source_id', stdout.getvalue())
            self.assertEqual(len((dest / 'responses.jsonl').read_text().splitlines()), 12)
        self.assertEqual(items, before)

    async def test_error_redaction_no_retry_and_cancel_propagation(self):
        llm = fake_llm()
        llm._chat.chat.completions.create.side_effect = ValueError('SECRET URL BODY')
        row = await probe.completion(llm, 's', 'u', 'c', 'low', 1500)
        self.assertEqual(row['error'], 'ValueError')
        self.assertNotIn('SECRET', json.dumps(row))
        self.assertEqual(llm._chat.chat.completions.create.await_count, 1)
        llm._chat.chat.completions.create.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await probe.completion(llm, 's', 'u', 'c', 'low', 1500)

    async def test_timeout_cancels_sdk_without_retry(self):
        llm = fake_llm()
        drained = asyncio.Event()
        async def delayed(**kwargs):
            try:
                await asyncio.Future()
            finally:
                drained.set()
        llm._chat.chat.completions.create.side_effect = delayed
        with patch.object(probe, 'TIMEOUT', .01):
            row = await probe.completion(llm, 's', 'u', 'c', 'medium', 2000)
        self.assertEqual(row['error'], 'TimeoutError')
        self.assertTrue(drained.is_set())
        self.assertEqual(llm._chat.chat.completions.create.await_count, 1)

    async def test_invalid_budgets_rejected_before_prepare_or_sdk(self):
        llm = fake_llm()
        for budget in (2001, 4000, 6000, 0, -1, True, 1500.0, '2000'):
            with self.subTest(budget=budget):
                with self.assertRaises(ValueError):
                    await probe.completion(llm, 's', 'u', 'c', 'medium', budget)
                with patch.object(probe, 'CONFIGS', (('medium', budget),)), \
                        patch.object(probe, 'public_items') as source:
                    with self.assertRaises(ValueError):
                        probe.prepare('must_not_create')
                    source.assert_not_called()
        llm._chat.chat.completions.create.assert_not_called()

    def test_planned_calls_follow_selected_items_and_configs(self):
        items = probe.select_items(probe.public_items())
        self.assertEqual(probe.CONFIGS, (('low', 1500), ('medium', 2000)))
        manifest = probe.provenance(items)
        self.assertEqual(manifest['planned_chat_calls'], 12)
        self.assertEqual(manifest['planned_embedding_calls'], 6)
        with patch.object(probe, 'CONFIGS', (('low', 1500),)):
            self.assertEqual(probe.provenance(items)['planned_chat_calls'], 6)

    async def test_cancelled_search_preserves_attempt_manifest_and_propagates(self):
        pipeline = CurrentSpan80.__new__(CurrentSpan80)
        pipeline._search = AsyncMock(side_effect=asyncio.CancelledError())
        with tempfile.TemporaryDirectory() as temp:
            manifest = {'status': 'prepared'}
            with self.assertRaises(asyncio.CancelledError):
                await probe.collect(probe.select_items(probe.public_items()), pipeline, Path(temp), manifest)
            saved = json.loads((Path(temp) / 'manifest.json').read_text())
        self.assertEqual(saved['status'], 'interrupted')
        self.assertEqual(saved['error'], 'CancelledError')
        self.assertEqual(saved['embedding_attempts'], 1)
        self.assertEqual(saved['chat_attempts'], 0)


if __name__ == '__main__':
    unittest.main()
