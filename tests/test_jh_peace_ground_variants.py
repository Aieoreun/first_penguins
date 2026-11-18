"""Offline checks of prompt selection and the unchanged grounding machinery."""
import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from core.search import Searcher
from core.state import Chunk
from pipelines.kimjaehak.peace_conditioned_safety import ConditionedSafety
from pipelines.kimjaehak.peace_ground_variants import (
    PROVENANCE_RULE, RELATION_RULE, ProvenanceGrounding, RelationGrounding,
)


class PeaceGroundVariantsTests(unittest.IsolatedAsyncioTestCase):
    variants = ((RelationGrounding, RELATION_RULE), (ProvenanceGrounding, PROVENANCE_RULE))

    def setup_pipeline(self, cls, response):
        p = cls.__new__(cls)
        p.searcher = Searcher([Chunk('A', 'Example textbook evidence.')])
        p.llm = SimpleNamespace(chat=AsyncMock(return_value=json.dumps(response)))
        p.span_tagger = SimpleNamespace(predict=lambda text: ([], 0.0))
        state = p._new_state({'messages': [{'role': 'user', 'content': 'Explain the example.'}],
                             'candidate_answer': 'The count is seven. Please verify.'}, True)
        state.chunks, state.section = p.searcher.chunks, 'A'
        return p, state

    async def test_actual_five_ground_calls_select_extended_prompt_only(self):
        response = {'checks': ['Short comparison.'], 'unsupported': ['count is seven']}
        for cls, rule in self.variants:
            p, state = self.setup_pipeline(cls, response)
            base, original = self.setup_pipeline(ConditionedSafety, response)
            await p._fusion_ground(state)
            await base._fusion_ground(original)
            self.assertEqual(p.llm.chat.await_count, 5)
            self.assertEqual(state.unsupported, original.unsupported)
            self.assertEqual(state.grounded, original.grounded)
            self.assertEqual(state.safe, original.safe)
            for call, parent in zip(p.llm.chat.call_args_list, base.llm.chat.call_args_list):
                self.assertEqual(call.args[0], ConditionedSafety.unsupported_system + rule)
                self.assertEqual(call.args[1:], parent.args[1:])
                self.assertEqual(call.kwargs, parent.kwargs)
                self.assertEqual(call.kwargs['effort'], 'low')
                self.assertEqual(call.kwargs['max_tokens'], 1500)
                self.assertEqual(call.kwargs['timeout'], 13)

    async def test_checks_are_ignored_even_when_they_claim_an_error(self):
        for cls, _ in self.variants:
            for checks in (['Everything is wrong.'], {'unsupported': ['count is seven']}, None):
                p, state = self.setup_pipeline(cls, {'checks': checks, 'unsupported': []})
                await p._fusion_ground(state)
                self.assertTrue(state.grounded)
                self.assertEqual(state.unsupported, [])
                self.assertEqual(state.errors, [])

    async def test_only_candidate_substrings_leave_existing_locator(self):
        for cls, _ in self.variants:
            p, state = self.setup_pipeline(cls, {
                'checks': ['invented explanation outside the candidate'],
                'unsupported': ['COUNT IS SEVEN', 'zzzzzzzzzzzzzzzzzzzzzzzz', 17]})
            await p._fusion_ground(state)
            self.assertEqual(state.unsupported, ['count is seven'])
            self.assertFalse(state.grounded)
            self.assertTrue(all(part in state.candidate_answer for part in state.unsupported))

    async def test_missing_unsupported_keeps_original_parse_failure_fallback(self):
        for cls, _ in self.variants:
            p, state = self.setup_pipeline(cls, {'checks': ['An error exists.']})
            await p._fusion_ground(state)
            self.assertTrue(state.grounded)
            self.assertEqual(state.unsupported, [])
            self.assertEqual(len(state.errors), 5)
            self.assertTrue(all('ValueError' in error for error in state.errors))

    async def test_span_refinement_remains_once_after_consensus(self):
        for cls, _ in self.variants:
            p, state = self.setup_pipeline(cls, {'checks': [], 'unsupported': ['count is seven']})
            seen = []
            def predict(text):
                seen.append(text)
                return ['The count is seven'], .99
            p.span_tagger.predict = predict
            await p._fusion_ground(state)
            self.assertEqual(seen, [state.candidate_answer])
            self.assertEqual(state.unsupported, ['The count is seven'])
            self.assertTrue(state.debug['learned_span'])
            self.assertFalse(state.grounded)

    async def test_concurrent_variants_do_not_mutate_parent_prompt(self):
        baseline = ConditionedSafety.unsupported_system
        pairs = [self.setup_pipeline(cls, {'checks': [], 'unsupported': []}) for cls, _ in self.variants]
        await asyncio.gather(*(p._fusion_ground(s) for p, s in pairs))
        self.assertIs(ConditionedSafety.unsupported_system, baseline)
        for (p, _), (cls, _) in zip(pairs, self.variants):
            self.assertTrue(all(call.args[0] == cls.unsupported_system for call in p.llm.chat.call_args_list))

    def test_all_methods_and_tutor_configuration_are_inherited(self):
        for cls, rule in self.variants:
            self.assertEqual(cls.unsupported_system, ConditionedSafety.unsupported_system + rule)
            for name in ('__init__', 'tutor', 'guard', '_classify', '_generate', '_ask',
                         '_fusion_ground', '_fusion_safety', '_check_output'):
                self.assertIs(getattr(cls, name), getattr(ConditionedSafety, name))
            for name in ('answer_systems', 'policy_system', 'source_extract_system',
                         'guard_safety_system', 'output_votes', 'learned_span_threshold',
                         'deadline', 'guard_deadline', 'guard_ground_timeout'):
                self.assertEqual(getattr(cls, name), getattr(ConditionedSafety, name))
            self.assertEqual(cls.output_votes, 5)
            self.assertEqual(cls.guard_ground_timeout, 13)


if __name__ == '__main__':
    unittest.main()
