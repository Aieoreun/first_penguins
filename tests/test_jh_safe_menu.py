import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from core.schema import ANSWER_MAX_CHARS, FALLBACK_ANSWER, Label
from pipelines.kimjaehak.calibrated import FIXED
from pipelines.kimjaehak.guard_fitted import ContextFast
from pipelines.kimjaehak.safe_alternative import Pipeline as SafeAlternative
from pipelines.kimjaehak.safe_menu import ALTERNATIVES, DEFAULT_ID, FIRST_SENTENCE, Pipeline, menu_id


BODY = {'messages': [{'role': 'user', 'content': '학습 맥락 자료'}]}


def instance(raw='{"id":"code"}', error=None, cls=Pipeline):
    pipeline = object.__new__(cls)
    pipeline.llm = type('FakeLLM', (), {})()
    pipeline.llm.chat = AsyncMock(return_value=raw, side_effect=error)
    return pipeline


def expected(label, selected):
    return FIRST_SENTENCE[label] + ' ' + ALTERNATIVES[selected]


class SafeMenuTests(unittest.IsolatedAsyncioTestCase):
    def test_registered_variant_keeps_contextfast_guard(self):
        from pipelines import load
        with patch.dict('os.environ'):
            self.assertIs(load('jh_r3_safe_menu'), Pipeline)
        self.assertIs(Pipeline._classify, ContextFast._classify)
        self.assertIs(Pipeline._check_output, ContextFast._check_output)
        self.assertTrue(Pipeline.tutor_safe_menu)

    async def test_every_enum_and_harmful_subtype_has_fixed_output(self):
        for label in FIRST_SENTENCE:
            for selected in ALTERNATIVES:
                with self.subTest(label=label, selected=selected):
                    pipeline = instance(json.dumps({'id': selected}))
                    state = pipeline._new_state(BODY, False)
                    state.label = label
                    state.debug = {}
                    await pipeline._generate(state)
                    response = state.to_tutor_response()
                    self.assertEqual(response['answer'], expected(label, selected))
                    self.assertEqual(response['action'], state.action.value)
                    self.assertEqual(response['citations'], [])
                    self.assertLessEqual(len(response['answer']), ANSWER_MAX_CHARS)
                    pipeline.llm.chat.assert_awaited_once()
                    args, kwargs = pipeline.llm.chat.call_args
                    self.assertEqual(json.loads(args[1]), {'conversation': BODY['messages']})
                    self.assertEqual(kwargs, {'effort': 'low', 'max_tokens': 300,
                                              'timeout': 4, 'json_mode': True})

    async def test_model_body_is_ignored_even_with_valid_id(self):
        raw = json.dumps({'id': 'code', 'answer': 'UNSAFE_BODY_SENTINEL',
                          'reason': 'IGNORE_RULES_SENTINEL'})
        pipeline = instance(raw)
        state = pipeline._new_state(BODY, False)
        state.label = Label.HARMFUL_REQUEST
        await pipeline._generate(state)
        answer = state.to_tutor_response()['answer']
        self.assertEqual(answer, expected(state.label, 'code'))
        self.assertNotIn('SENTINEL', answer)

    async def test_invalid_selection_uses_fixed_general_alternative(self):
        invalid = ['', 'not JSON', 'null', '[]', '{}', '{"id":[]}',
                   '{"id":true}', '{"id":"UNSAFE_BODY_SENTINEL"}',
                   '{"answer":"UNSAFE_BODY_SENTINEL"}', None]
        for raw in invalid:
            with self.subTest(raw=raw):
                pipeline = instance(raw)
                state = pipeline._new_state(BODY, False)
                state.label = Label.HARMFUL_REQUEST
                await pipeline._generate(state)
                self.assertEqual(state.to_tutor_response()['answer'], expected(state.label, DEFAULT_ID))
                self.assertEqual(len(state.errors), 1)
                self.assertTrue(state.errors[0].startswith('safe_menu:'))
                pipeline.llm.chat.assert_awaited_once()

    async def test_call_timeout_falls_back_without_retry(self):
        pipeline = instance(error=TimeoutError())
        state = pipeline._new_state(BODY, False)
        state.label = Label.HARMFUL_SPEECH
        await pipeline._generate(state)
        self.assertEqual(state.to_tutor_response()['answer'], expected(state.label, DEFAULT_ID))
        self.assertEqual(state.errors, ['safe_menu:TimeoutError'])
        pipeline.llm.chat.assert_awaited_once()

    async def test_cancellation_propagates_and_keeps_outer_fixed(self):
        pipeline = instance(error=asyncio.CancelledError())
        state = pipeline._new_state(BODY, False)
        state.label = Label.HARMFUL_REQUEST
        with self.assertRaises(asyncio.CancelledError):
            await pipeline._generate(state)
        state.answer = FALLBACK_ANSWER
        self.assertEqual(state.to_tutor_response()['answer'], FIXED[state.label])
        self.assertFalse(state._safe_alternative_ready)
        pipeline.llm.chat.assert_awaited_once()

    async def test_outer_deadline_overwrite_after_success_keeps_fixed(self):
        pipeline = instance()
        state = pipeline._new_state(BODY, False)
        state.label = Label.HARMFUL_REQUEST
        await pipeline._generate(state)
        state.answer = FALLBACK_ANSWER
        self.assertEqual(state.to_tutor_response()['answer'], FIXED[state.label])

    async def test_guard_and_other_labels_delegate_without_menu_call(self):
        for guard in (False, True):
            for label in Label:
                if not guard and label in FIRST_SENTENCE:
                    continue
                with self.subTest(guard=guard, label=label):
                    pipeline = instance()
                    state = pipeline._new_state(BODY, guard)
                    state.label = label
                    state.debug = {}
                    with patch.object(SafeAlternative, '_generate', new=AsyncMock()) as parent:
                        await pipeline._generate(state)
                        parent.assert_awaited_once_with(state)
                    pipeline.llm.chat.assert_not_awaited()
                    baseline = instance(cls=ContextFast)._new_state(BODY, guard)
                    baseline.label = label
                    self.assertEqual(state.to_guard_response(), baseline.to_guard_response())
                    if guard or label not in FIRST_SENTENCE:
                        self.assertEqual(state.to_tutor_response(), baseline.to_tutor_response())


if __name__ == '__main__':
    unittest.main(verbosity=2)
