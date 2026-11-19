import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from core.schema import Label
from pipelines.kimjaehak.calibrated import BLOCK_SYSTEM, Pipeline as Calibrated
from pipelines.kimjaehak.focused_excerpt import Selected as SelectedFocused
from pipelines.kimjaehak.structured_block import Pipeline, STRUCTURED_BLOCK_SYSTEM
from pipelines.kimjaehak.voted import Pipeline as Voted


class StructuredBlockTests(unittest.IsolatedAsyncioTestCase):
    def make_pipeline(self, cls=Pipeline, reply=None):
        p = cls.__new__(cls)
        p.llm = SimpleNamespace(chat=AsyncMock(return_value=json.dumps(
            reply if reply is not None else {'category': 'none', 'confidence': 90})))
        return p

    def state(self, p, guard=False):
        s = p._new_state({'messages': [{'role': 'user', 'content': 'synthetic request'}]}, guard)
        s.debug = {}  # Learned classification may replace debug entirely.
        return s

    async def test_real_ask_chain_uses_tutor_hook_and_preserves_budget(self):
        p = self.make_pipeline()
        s = self.state(p)
        result = await p._ask(s, 'block', 'upstream placeholder', 'payload', 'low', 9)
        p.llm.chat.assert_awaited_once_with(STRUCTURED_BLOCK_SYSTEM, 'payload',
            effort='low', max_tokens=1500, timeout=9, json_mode=True)
        self.assertEqual(result, {'category': 'none', 'confidence': 90})
        self.assertEqual(s.errors, [])

    async def test_guard_and_baseline_use_original_system_and_serialization(self):
        for guard in (False, True):
            base = self.make_pipeline(SelectedFocused)
            state = self.state(base, guard)
            await base._ask(state, 'block', 'ignored', 'payload', 'low', 7)
            self.assertEqual(base.llm.chat.await_args.args[0], BLOCK_SYSTEM)
        p = self.make_pipeline()
        s = self.state(p, True)
        await p._ask(s, 'block', 'ignored', 'payload', 'low', 7)
        self.assertEqual(p.llm.chat.await_args, base.llm.chat.await_args)
        self.assertEqual(s.to_guard_response(), state.to_guard_response())
        self.assertEqual(s.to_tutor_response(), state.to_tutor_response())
        self.assertTrue(s._structured_block_guard)

    async def test_other_material_checks_pass_through(self):
        for guard in (False, True):
            p = self.make_pipeline()
            s = self.state(p, guard)
            for name in ('other', 'material', 'rewrite'):
                await p._ask(s, name, 'unchanged system', 'payload', 'low', 6)
                self.assertEqual(p.llm.chat.await_args.args[0], 'unchanged system')

    async def test_auxiliary_fields_do_not_override_category(self):
        for category, act, expected in [('none', 'apply', Label.NORMAL),
                                        ('jailbreak_injection', 'discuss', Label.JAILBREAK_INJECTION)]:
            raw = {'category': category, 'confidence': 90, 'speech_act': act,
                   'target': 'example', 'request_span': 'untrusted auxiliary content'}
            p = self.make_pipeline(reply=raw)
            s = self.state(p)
            self.assertEqual(await p._ask(s, 'block', '', 'payload', 'low', 8), raw)
            # Exercise the actual ballot/category consumer, with only model output mocked.
            p._collect_checks = AsyncMock(return_value=(
                [raw, raw], [{'category': 'none', 'self_info': False}], {'quote': 'present'}))
            await Voted._classify(p, s)
            self.assertEqual(s.label, expected)

    async def test_invalid_output_and_timeout_keep_existing_fallback(self):
        for guard in (False, True):
            for behavior in ('invalid', 'timeout'):
                p = self.make_pipeline()
                s = self.state(p, guard)
                if behavior == 'invalid':
                    p.llm.chat.return_value = 'not JSON'
                else:
                    p.llm.chat.side_effect = TimeoutError()
                self.assertEqual(await p._ask(s, 'block', '', 'payload', 'low', 8), {})
                self.assertEqual(s.errors, ['block:parse' if behavior == 'invalid' else 'block:TimeoutError'])
                self.assertEqual(p.llm.chat.await_count, 1)

    async def test_cancellation_propagates_without_retry(self):
        p = self.make_pipeline()
        p.llm.chat.side_effect = asyncio.CancelledError()
        s = self.state(p)
        with self.assertRaises(asyncio.CancelledError):
            await p._ask(s, 'block', '', 'payload', 'low', 8)
        self.assertEqual(p.llm.chat.await_count, 1)
        self.assertEqual(s.errors, [])

    async def test_parallel_guard_and_tutor_do_not_share_identity(self):
        p = self.make_pipeline()
        seen = []
        async def chat(system, user, **kwargs):
            await asyncio.sleep(0)
            seen.append((user, system, kwargs))
            return '{"category":"none","confidence":90}'
        p.llm.chat = AsyncMock(side_effect=chat)
        states = [(str(i), self.state(p, bool(i % 2))) for i in range(24)]
        await asyncio.gather(*(p._ask(s, 'block', 'ignored', key, 'low', 8) for key, s in states))
        self.assertEqual(len(seen), 24)
        for key, system, kwargs in seen:
            self.assertEqual(system, BLOCK_SYSTEM if int(key) % 2 else STRUCTURED_BLOCK_SYSTEM)
            self.assertEqual(kwargs, {'effort':'low', 'max_tokens':1500, 'timeout':8, 'json_mode':True})
        self.assertNotIn('_structured_block_guard', vars(p))
        self.assertTrue(all(not s.errors for _, s in states))

    def test_prompt_and_all_other_components_are_preserved(self):
        self.assertTrue(STRUCTURED_BLOCK_SYSTEM.startswith(BLOCK_SYSTEM))
        schema = STRUCTURED_BLOCK_SYSTEM.rsplit('# 출력 형식 보완', 1)[1]
        for key in ('speech_act', 'target', 'request_span'):
            self.assertLess(schema.index('"'+key+'"'), schema.index('"category"'))
        p = self.make_pipeline()
        self.assertEqual(p._block_system(SimpleNamespace()), BLOCK_SYSTEM)
        for name in ('guard', '_classify', '_ask', '_generate', '_collect_checks', '_vote_group',
                     '_search', '_check_output', '_source_candidates', '_validate_source'):
            self.assertIs(getattr(Pipeline, name), getattr(SelectedFocused, name))
        for name in ('deadline', 'prepare_limit', 'learned_gate_threshold', 'short_votes',
                     'block_min_votes', 'request_slots', 'source_extract_timeout'):
            self.assertEqual(getattr(Pipeline, name), getattr(SelectedFocused, name))
        self.assertEqual(Calibrated.__dict__['_block_system'](p, SimpleNamespace()), BLOCK_SYSTEM)


if __name__ == '__main__':
    unittest.main()
