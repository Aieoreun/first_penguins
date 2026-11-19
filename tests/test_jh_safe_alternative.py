import asyncio
import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

from core.schema import ANSWER_MAX_CHARS, FALLBACK_ANSWER, Label
from pipelines.kimjaehak.calibrated import FIXED
from pipelines.kimjaehak.guard_fitted import ContextFast
from pipelines.kimjaehak.safe_alternative import (
    ALTERNATIVE_SYSTEM, FIRST_SENTENCE, Pipeline, compose_answer,
)


BODY = {'messages': [{'role': 'user', 'content': '실습 자료로 공부하고 싶어요.'}]}
ALTERNATIVE = '가상의 자료를 사용해 조건문으로 값을 구분하는 연습을 함께 해 봐요.'


def instance(cls=Pipeline, answer=ALTERNATIVE, error=None):
    pipeline = object.__new__(cls)  # No fitted models, credentials, client or API.
    pipeline.llm = type('FakeLLM', (), {})()
    pipeline.llm.chat = AsyncMock(return_value=answer, side_effect=error)
    pipeline._slots = asyncio.Semaphore(1)
    return pipeline


class SafeAlternativeTests(unittest.IsolatedAsyncioTestCase):
    async def test_harmful_subtypes_use_one_call_with_original_budget(self):
        for label in FIRST_SENTENCE:
            with self.subTest(label=label):
                pipeline = instance()
                state = pipeline._new_state(BODY, False)
                state.label = label
                state.debug = {}  # Classification can replace this dictionary.
                await pipeline._generate(state)
                response = state.to_tutor_response()
                self.assertEqual(response['answer'], FIRST_SENTENCE[label] + ' ' + ALTERNATIVE)
                self.assertEqual(response['action'], state.action.value)
                self.assertEqual(response['citations'], [])
                pipeline.llm.chat.assert_awaited_once()
                args, kwargs = pipeline.llm.chat.call_args
                self.assertEqual(kwargs, {'effort': 'low', 'max_tokens': 1500, 'timeout': 8})
                self.assertEqual(args[0], ALTERNATIVE_SYSTEM)
                self.assertEqual(json.loads(args[1]), {'conversation': BODY['messages']})

    async def test_other_labels_delegate_and_serialize_unchanged(self):
        for label in Label:
            if label in FIRST_SENTENCE:
                continue
            with self.subTest(label=label):
                pipeline = instance()
                state = pipeline._new_state(BODY, False)
                state.label = label
                with patch.object(ContextFast, '_generate', new=AsyncMock()) as parent:
                    await pipeline._generate(state)
                    parent.assert_awaited_once_with(state)
                pipeline.llm.chat.assert_not_awaited()
                baseline = instance(ContextFast)._new_state(BODY, False)
                baseline.label = label
                for item in (state, baseline):
                    item.answer = '기존 답변'
                    item.citations = ['PY-01-01']
                self.assertEqual(state.to_tutor_response(), baseline.to_tutor_response())

    async def test_guard_delegates_even_for_harmful_label(self):
        pipeline = instance()
        state = pipeline._new_state(BODY, True)
        state.label = Label.HARMFUL_REQUEST
        state.debug = {}
        with patch.object(ContextFast, '_generate', new=AsyncMock()) as parent:
            await pipeline._generate(state)
            parent.assert_awaited_once_with(state)
        pipeline.llm.chat.assert_not_awaited()

    async def test_all_guard_labels_serialize_identically(self):
        for label in Label:
            with self.subTest(label=label):
                body = {**BODY, 'candidate_answer': '원래 검사 대상 답변'}
                state = instance()._new_state(body, True)
                baseline = instance(ContextFast)._new_state(body, True)
                for item in (state, baseline):
                    item.label = label
                    item.answer = '기존 값'
                    item.citations = ['PY-01-01']
                    item.safe = False
                    item.grounded = False
                    item.unsupported = ['실제 구절']
                self.assertEqual(state.to_guard_response(), baseline.to_guard_response())
                self.assertEqual(state.to_tutor_response(), baseline.to_tutor_response())

    async def test_failure_uses_existing_fixed_without_second_call(self):
        for label in FIRST_SENTENCE:
            for invalid in ['', ' \n ', None, 17, '가' * ANSWER_MAX_CHARS]:
                with self.subTest(label=label, invalid=type(invalid).__name__):
                    pipeline = instance(answer=invalid)
                    state = pipeline._new_state(BODY, False)
                    state.label = label
                    await pipeline._generate(state)
                    self.assertEqual(state.to_tutor_response()['answer'], FIXED[label])
                    pipeline.llm.chat.assert_awaited_once()
            pipeline = instance(error=TimeoutError())
            state = pipeline._new_state(BODY, False)
            state.label = label
            await pipeline._generate(state)
            self.assertEqual(state.to_tutor_response()['answer'], FIXED[label])
            pipeline.llm.chat.assert_awaited_once()

    async def test_punctuation_and_line_breaks_are_not_content_rejections(self):
        alternatives = [
            '가상 자료로 조건문을 함께 연습해 봐요',
            '가상의 자료를 사용해\n조건문을 함께 연습해 봐요.',
            '가상 자료로 연습해요. 개인정보를 보호해요.',
        ]
        for alternative in alternatives:
            with self.subTest(alternative=alternative):
                pipeline = instance(answer=alternative)
                state = pipeline._new_state(BODY, False)
                state.label = Label.HARMFUL_REQUEST
                await pipeline._generate(state)
                self.assertEqual(
                    state.to_tutor_response()['answer'],
                    FIRST_SENTENCE[state.label] + ' ' + alternative,
                )
                self.assertEqual(state.errors, [])

    async def test_cancellation_propagates_and_serializes_fixed(self):
        for label in FIRST_SENTENCE:
            with self.subTest(label=label):
                pipeline = instance(error=asyncio.CancelledError())
                state = pipeline._new_state(BODY, False)
                state.label = label
                with self.assertRaises(asyncio.CancelledError):
                    await pipeline._generate(state)
                state.answer = FALLBACK_ANSWER
                self.assertEqual(state.to_tutor_response()['answer'], FIXED[label])
                pipeline.llm.chat.assert_awaited_once()

    async def test_outer_deadline_and_post_success_overwrite_use_fixed(self):
        pipeline = instance()

        async def prepare(state, extra=''):
            state.label = Label.HARMFUL_REQUEST

        started = asyncio.Event()

        async def pending(*args, **kwargs):
            started.set()
            await asyncio.Future()

        pipeline._prepare_within = prepare
        pipeline.llm.chat = AsyncMock(side_effect=pending)
        # Start the actual tutor path, then cancel it through its outer deadline
        # helper after generation starts; no wall-clock race or network involved.
        original_wait_for = asyncio.wait_for

        async def expire_work(awaitable, timeout):
            if timeout == pipeline.deadline:
                task = asyncio.create_task(awaitable)
                await started.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                raise asyncio.TimeoutError()
            return await original_wait_for(awaitable, timeout)

        with patch('pipelines.kimjaehak.voted.asyncio.wait_for', side_effect=expire_work):
            state = await pipeline.tutor(copy.deepcopy(BODY))
        self.assertEqual(state.to_tutor_response()['answer'], FIXED[state.label])
        self.assertIn('request:Deadline', state.errors)
        pipeline.llm.chat.assert_awaited_once()

        pipeline = instance()
        state = pipeline._new_state(BODY, False)
        state.label = Label.HARMFUL_REQUEST
        await pipeline._generate(state)
        state.answer = FALLBACK_ANSWER
        self.assertEqual(state.to_tutor_response()['answer'], FIXED[state.label])

    async def test_real_tutor_and_guard_routing_with_mock_prepare(self):
        async def prepare(state, extra=''):
            state.label = Label.HARMFUL_REQUEST

        pipeline = instance()
        pipeline._prepare_within = prepare
        state = await pipeline.tutor(copy.deepcopy(BODY))
        self.assertEqual(
            state.to_tutor_response()['answer'],
            FIRST_SENTENCE[state.label] + ' ' + ALTERNATIVE,
        )
        pipeline.llm.chat.assert_awaited_once()

        guard = instance()
        guard._prepare_within = prepare
        guard._check_output = AsyncMock()
        state = await guard.guard({**BODY, 'candidate_answer': '검토할 답변'})
        guard._check_output.assert_awaited_once_with(state)
        guard.llm.chat.assert_not_awaited()

    async def test_exact_length_and_overflow(self):
        label = Label.HARMFUL_REQUEST
        remaining = ANSWER_MAX_CHARS - len(FIRST_SENTENCE[label]) - 1
        self.assertEqual(len(compose_answer(label, '가' * remaining)), ANSWER_MAX_CHARS)
        with self.assertRaises(ValueError):
            compose_answer(label, '가' * (remaining + 1))

    async def test_text_validation_is_not_semantic_safety_validation(self):
        # This neutral sentinel deliberately survives text/length checks.
        # Neither a fixed refusal nor successful assembly verifies model safety.
        text = 'UNSAFE_BODY_SENTINEL'
        self.assertIn(text, compose_answer(Label.HARMFUL_REQUEST, text))


if __name__ == '__main__':
    unittest.main()
