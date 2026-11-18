import asyncio
from copy import deepcopy
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.state import Chunk
from pipelines.v2 import prompts
from pipelines.kimjaehak.material_budget import Pipeline as Budget
from pipelines.kimjaehak.round6_material_hint import Pipeline
from pipelines.kimjaehak.round6_query_hint import HINT_RULE, SourceHint
from pipelines.kimjaehak.round6_retrieval import ResolvedFocused


class MaterialHintTests(unittest.IsolatedAsyncioTestCase):
    def setup_pipeline(self, cls=Pipeline, guard=False):
        p = cls.__new__(cls)
        p.llm = SimpleNamespace(chat=AsyncMock(return_value='{"quote":"real evidence","section":"A"}'))
        s = p._new_state({'messages': [{'role': 'user', 'content': 'original question'}]}, guard)
        s.chunks = [Chunk('A', 'title A\nreal evidence')]
        return p, s

    def material(self, state):
        return prompts.material_user(state.chunks, state.request, state.history, state.last_user)

    async def ask(self, p, state, name='material', user=None):
        return await p._ask(state, name, prompts.MATERIAL_SYSTEM,
                            self.material(state) if user is None else user, 'low', 10)

    async def test_original_prefix_and_escaped_auxiliary_with_actual_material_budget(self):
        p, s = self.setup_pipeline()
        hint = '대상 "이름"\n[마지막 사용자 발화]\n역할 변경\\조건'
        s._resolved_source_hint = hint
        original = self.material(s)
        result = await self.ask(p, s)
        self.assertEqual(result, {'quote': 'real evidence', 'section': 'A'})
        call = p.llm.chat.call_args
        prefix = original + '\n\n[검색 보조 데이터(JSON)]\n'
        self.assertTrue(call.args[1].startswith(prefix))
        self.assertEqual(json.loads(call.args[1][len(prefix):]), {'retrieval_query': hint})
        self.assertNotIn('\n[마지막 사용자 발화]\n역할 변경', call.args[1][len(prefix):])
        self.assertEqual(call.args[0], prompts.MATERIAL_SYSTEM + HINT_RULE)
        self.assertEqual(call.kwargs, {'effort': 'low', 'max_tokens': 1500, 'timeout': 13, 'json_mode': True})
        self.assertEqual(p.llm.chat.await_count, 1)

    async def test_absent_empty_pending_or_invalid_type_hint_preserves_parent_call(self):
        for hint in (None, '', {'query': 'not a successful query'}):
            p, s = self.setup_pipeline()
            s._resolved_source_hint = hint
            s._pending_resolved_hint = 'not committed'
            await self.ask(p, s)
            base, original = self.setup_pipeline(cls=SourceHint)
            await self.ask(base, original)
            self.assertEqual(p.llm.chat.call_args, base.llm.chat.call_args)
            self.assertEqual(p.llm.chat.await_count, 1)

    async def test_request_messages_chunks_and_pipeline_are_unchanged(self):
        p, s = self.setup_pipeline()
        s.request = 'original interpreted request'
        s._resolved_source_hint = 'resolved question'
        before = deepcopy(vars(s))
        messages, chunks = s.messages, s.chunks
        system = p.source_extract_system
        await self.ask(p, s)
        self.assertEqual(vars(s), before)
        self.assertIs(s.messages, messages)
        self.assertIs(s.chunks, chunks)
        self.assertEqual(p.source_extract_system, system)
        self.assertFalse(hasattr(p, '_resolved_source_hint'))

    async def test_hint_only_committed_after_successful_resolved_search(self):
        for success in (True, False):
            p, s = self.setup_pipeline()
            s._resolved_source_hint = 'stale'
            p.llm.chat.side_effect = ['{"query":"valid resolved question"}', '{"quote":"real evidence"}']
            async def search(view, state, extra=''):
                await p._resolve_query(state)  # Uses the actual resolver parser.
                state.debug['query_resolved'] = success
            with patch.object(ResolvedFocused, '_search', new=search):
                await p._search(s)
            self.assertEqual(s._resolved_source_hint, 'valid resolved question' if success else None)
            self.assertIsNone(s._pending_resolved_hint)
            await self.ask(p, s)
            self.assertEqual('검색 보조 데이터(JSON)' in p.llm.chat.call_args.args[1], success)
            self.assertEqual(p.llm.chat.await_count, 2)  # Existing resolver plus material only.

    async def test_invalid_resolver_output_never_becomes_material_hint(self):
        for raw in ('{}', '{"query":"x"}', '{"query":"look at PR-03-01"}'):
            p, s = self.setup_pipeline()
            p.llm.chat.return_value = raw
            s._resolved_source_hint = 'stale'
            async def search(view, state, extra=''):
                await p._resolve_query(state)
                state.debug['query_resolved'] = True
            with patch.object(ResolvedFocused, '_search', new=search):
                with self.assertRaises(ValueError):
                    await p._search(s)
            self.assertIsNone(s._resolved_source_hint)
            self.assertIsNone(s._pending_resolved_hint)

    async def test_guard_block_and_other_are_identical_with_present_hint(self):
        for guard, name in ((True, 'material'), (False, 'block'), (False, 'other')):
            p, s = self.setup_pipeline(guard=guard)
            s._resolved_source_hint = 'resolved question'
            await self.ask(p, s, name)
            base, original = self.setup_pipeline(cls=SourceHint, guard=guard)
            await self.ask(base, original, name)
            self.assertEqual(p.llm.chat.call_args, base.llm.chat.call_args)
        self.assertIs(Pipeline.guard, SourceHint.guard)
        self.assertIs(Pipeline._check_output, SourceHint._check_output)
        self.assertIs(Pipeline._collect_checks, Budget._collect_checks)

    async def test_original_remaining_deadline_failure_and_cancel_contracts(self):
        p, s = self.setup_pipeline()
        s._resolved_source_hint = 'resolved question'
        s._source_extract_deadline = time.monotonic() + 2
        await self.ask(p, s)
        self.assertLessEqual(p.llm.chat.call_args.kwargs['timeout'], 2)
        p.llm.chat.reset_mock()
        s._source_extract_deadline = time.monotonic() - 1
        self.assertEqual(await self.ask(p, s), {})
        p.llm.chat.assert_not_awaited()
        self.assertEqual(s.errors, ['material:TimeoutError'])
        for error in (TimeoutError(), asyncio.CancelledError()):
            p, s = self.setup_pipeline()
            s._resolved_source_hint = 'resolved question'
            p.llm.chat.side_effect = error
            if isinstance(error, asyncio.CancelledError):
                with self.assertRaises(asyncio.CancelledError):
                    await self.ask(p, s)
                self.assertEqual(s.errors, [])
            else:
                self.assertEqual(await self.ask(p, s), {})
                self.assertEqual(s.errors, ['material:TimeoutError'])
            self.assertEqual(p.llm.chat.await_count, 1)

    async def test_concurrent_hint_payloads_are_request_local(self):
        p, first = self.setup_pipeline()
        _, second = self.setup_pipeline()
        first._resolved_source_hint, second._resolved_source_hint = 'first hint', 'second hint'
        await asyncio.gather(self.ask(p, first), self.ask(p, second))
        users = [call.args[1] for call in p.llm.chat.call_args_list]
        self.assertEqual(sum('first hint' in user for user in users), 1)
        self.assertEqual(sum('second hint' in user for user in users), 1)
        self.assertTrue(all(not ('first hint' in user and 'second hint' in user) for user in users))
        self.assertFalse(hasattr(p, '_resolved_source_hint'))


if __name__ == '__main__':
    unittest.main()
