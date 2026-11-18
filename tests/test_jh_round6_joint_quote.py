import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.schema import Label
from core.search import Searcher
from core.state import Chunk
from pipelines.kimjaehak.material_budget import Pipeline as Budget
from pipelines.kimjaehak.round6_joint_evidence import Pipeline as Joint
from pipelines.kimjaehak.round6_joint_quote import Pipeline
from pipelines.kimjaehak.voted import Pipeline as Voted


class JointQuoteTests(unittest.IsolatedAsyncioTestCase):
    def setup_pipeline(self, result=None, cls=Pipeline):
        p = cls.__new__(cls)
        p.searcher = Searcher([Chunk('A', 'title A\nalpha'), Chunk('B', 'title B\nbeta evidence')])
        if result is None:
            result = {'supported': True, 'section': 'B', 'quote': 'beta evidence'}
        p.llm = SimpleNamespace(chat=AsyncMock(return_value=json.dumps(result)))
        s = p._new_state({'messages': [{'role': 'user', 'content': 'beta question'}]}, False)
        s.chunks = p.searcher.chunks
        return p, s

    async def ask(self, p, s):
        return await p._ask(s, 'material', 'old system', 'old user', 'low', 10)

    async def test_original_adapter_behavior_unchanged(self):
        p, s = self.setup_pipeline({'supported': True, 'section': 'B'}, cls=Joint)
        self.assertEqual(await self.ask(p, s), {'quote': 'title B\nbeta evidence', 'section': 'B'})
        self.assertEqual(s._joint_evidence_source, 'B')
        self.assertEqual(s.errors, [])
        self.assertEqual(p._adapt_joint_result({'supported': False, 'section': 'A'}, ['A']),
                         ({'quote': '', 'section': 'A'}, None, None))
        with self.assertRaises(ValueError):
            p._adapt_joint_result({'supported': 'true', 'section': 'A'}, ['A'])

    async def test_valid_quote_normalizes_whitespace_single_nonstream_call(self):
        p, s = self.setup_pipeline({'supported': True, 'section': 'B', 'quote': 'beta\n evidence'})
        self.assertEqual(await self.ask(p, s), {'quote': 'beta\n evidence', 'section': 'B'})
        self.assertEqual(s._joint_evidence_source, 'B')
        self.assertIsNone(s.section)
        self.assertEqual(p.llm.chat.await_count, 1)
        self.assertEqual(p.llm.chat.call_args.kwargs,
                         {'effort': 'low', 'max_tokens': 2000, 'timeout': 13, 'json_mode': True})
        self.assertIn('asked_fact', p.llm.chat.call_args.args[0])

    async def test_missing_empty_wrong_section_or_invented_quote_abstains_no_retry(self):
        for quote in (None, '', '  \n', 123, 'alpha', 'invented evidence'):
            result = {'supported': True, 'section': 'B'}
            if quote is not None:
                result['quote'] = quote
            p, s = self.setup_pipeline(result)
            adapted = await self.ask(p, s)
            self.assertEqual(adapted, {'quote': '', 'section': 'B'})
            self.assertIsNone(s._joint_evidence_source)
            self.assertEqual(s.errors, ['material_joint:InvalidQuote'])
            self.assertEqual(p.llm.chat.await_count, 1)
            p._collect_checks = AsyncMock(return_value=([{'category': 'none'}],
                [{'category': 'none', 'self_info': False}], adapted))
            await Voted._classify(p, s)
            self.assertEqual(s.label, Label.OUT_OF_MATERIAL)

    async def test_false_needs_no_quote_and_invalid_decision_keeps_old_fallback(self):
        p, s = self.setup_pipeline({'supported': False, 'section': 'B'})
        self.assertEqual(await self.ask(p, s), {'quote': '', 'section': 'B'})
        self.assertEqual(s.errors, [])
        for result in ({'supported': 1, 'section': 'B', 'quote': 'beta'},
                       {'supported': True, 'section': 'missing', 'quote': 'beta'}):
            p, s = self.setup_pipeline(result)
            self.assertEqual(await self.ask(p, s), {})
            self.assertEqual(s.errors, ['material_joint:ValueError'])
            self.assertIsNone(s._joint_evidence_source)

    async def test_lexical_validation_is_not_semantic_validation(self):
        p, s = self.setup_pipeline({'supported': True, 'section': 'B', 'quote': 'title B',
            'asked_fact': {'entity': 'unrelated entity', 'property_or_condition': 'missing property'}})
        # A real substring passes even when the requested fact is unsupported.
        # The prompt must address this; this validator makes no semantic claim.
        self.assertEqual((await self.ask(p, s))['quote'], 'title B')
        self.assertEqual(s._joint_evidence_source, 'B')

    async def test_existing_block_priority_beats_empty_quote(self):
        p, s = self.setup_pipeline()
        p._collect_checks = AsyncMock(return_value=([{'category': 'harmful_request'}] * 3,
            [{'category': 'none', 'self_info': False}], {'quote': '', 'section': 'B'}))
        await Voted._classify(p, s)
        self.assertEqual(s.label, Label.HARMFUL_REQUEST)

    async def test_gates_and_completed_classification_control_reuse(self):
        for label, gate in ((Label.NORMAL, None), (Label.PII_DISCLOSURE, None),
                            (Label.NORMAL, 'learned_section'), (Label.NORMAL, 'tutor_section_gate'),
                            (Label.HARMFUL_REQUEST, None)):
            p, s = self.setup_pipeline()
            async def classify(view, state):
                await self.ask(view, state)
                state.section = 'A'
                state.label = label
                state.debug = {gate: True} if gate else {}
            with patch.object(Budget, '_classify', new=classify):
                await p._classify(s)
            reuse = label in {Label.NORMAL, Label.PII_DISCLOSURE} and gate is None
            self.assertEqual(s._reuse_joint_evidence, reuse)
            self.assertEqual(s.section, 'B' if reuse else 'A')

    async def test_cancel_propagates_and_never_enables_reuse(self):
        p, s = self.setup_pipeline()
        p.llm.chat.side_effect = asyncio.CancelledError
        with self.assertRaises(asyncio.CancelledError):
            await self.ask(p, s)
        self.assertEqual(s.errors, [])
        p, s = self.setup_pipeline()
        async def incomplete(view, state):
            await self.ask(view, state)
            raise asyncio.CancelledError
        with patch.object(Budget, '_classify', new=incomplete):
            with self.assertRaises(asyncio.CancelledError):
                await p._classify(s)
        self.assertFalse(s._reuse_joint_evidence)

    async def test_guard_delegates_and_success_skips_source_only(self):
        p, s = self.setup_pipeline()
        s._material_budget_guard = True
        with patch.object(Budget, '_ask', new=AsyncMock(return_value={'guard': True})) as original:
            self.assertEqual(await self.ask(p, s), {'guard': True})
            original.assert_awaited_once_with(s, 'material', 'old system', 'old user', 'low', 10)
        p.llm.chat.assert_not_awaited()
        self.assertIs(Pipeline.guard, Joint.guard)
        self.assertIs(Pipeline._check_output, Joint._check_output)
        p, s = self.setup_pipeline()
        await self.ask(p, s)
        s.section = 'B'
        s.label = Label.NORMAL
        s._reuse_joint_evidence = True
        p._source_response = AsyncMock()
        await p._generate(s)
        self.assertEqual(s.citations, ['B'])
        self.assertIn('title B\nbeta evidence', s.answer)
        p._source_response.assert_not_awaited()
        self.assertEqual(p.llm.chat.await_count, 1)


if __name__ == '__main__':
    unittest.main()
