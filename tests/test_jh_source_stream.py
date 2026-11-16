import asyncio
import json
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from pipelines.kimjaehak.source_stream import Pipeline, first_section
from core.schema import Label
from core.state import State, Chunk
from pipelines.kimjaehak.source_extract import SelectedSection
from pipelines.kimjaehak.focused_excerpt import Selected as SelectedFocused
from pipelines.kimjaehak.guard_fitted import ContextFast


def event(content=None, **kwargs):
    return NS(choices=[NS(index=0, finish_reason=kwargs.pop('finish_reason', None),
                          delta=NS(content=content, **kwargs))])


class Stream:
    def __init__(self, parts, close_error=None):
        self.parts = parts
        self.closed = False
        self.read = 0
        self.close_error = close_error
        self.started = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True
        if self.close_error:
            raise self.close_error

    async def __aiter__(self):
        self.started.set()
        for part in self.parts:
            if part is WAIT:
                await asyncio.Event().wait()
            elif isinstance(part, BaseException):
                raise part
            else:
                self.read += 1
                yield event(part) if isinstance(part, str) else part


WAIT = object()


def case(parts, cls=Pipeline, **kwargs):
    stream = Stream(parts, **kwargs)
    create = AsyncMock(return_value=stream)
    p = cls.__new__(cls)
    p.llm = NS(model='provided-model', _chat=NS(chat=NS(completions=NS(create=create))),
               chat=AsyncMock(return_value='{"section":"last"}'))
    p.searcher = NS(search=lambda *a, **kw: [Chunk('last', 'last material')],
                    section_text=lambda sid: sid + ' material')
    s = State(messages=[{'role': 'user', 'content': '자료 질문'}], label=Label.NORMAL,
              section='old', chunks=[Chunk('old', 'old material')])
    s._source_extract_deadline = time.monotonic() + 26
    return p, s, stream, create


class ParserTests(unittest.TestCase):
    def test_every_split_waits_until_complete_value_and_delimiter(self):
        text = ' \r\n{ "sec\\u0074ion" : "PY-\\u0030\\u0031" \t,'
        for n in range(len(text)):
            self.assertIsNone(first_section(text[:n], ['PY-01']))
        self.assertEqual(first_section(text, ['PY-01']), 'PY-01')

    def test_no_nested_partial_nonstring_or_prose_matches(self):
        for text in ['{"reason":"section","section":"last"}',
                     '{"x":{"section":"last"}}', '```{"section":"last"}',
                     'prefix {"section":"last"}', '{"section":"las",',
                     '{"section":null}', '{"section":1}', '{"section":["last"]}',
                     '{"section":"last"x', '{"section":"last"',
                     '{"section":"outside"}']:
            self.assertIsNone(first_section(text, ['last']), text)

    def test_escaped_quote_id_and_duplicate_semantics(self):
        self.assertEqual(first_section('{"section":"a\\\"b"}', ['a"b']), 'a"b')
        # First field policy is explicit, including duplicate in same chunk.
        self.assertEqual(first_section('{"section":"old","section":"last"}',
                                       ['old', 'last']), 'old')


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_early_finish_closes_does_not_read_reason_and_exact_payload(self):
        p, s, stream, create = case([NS(choices=[]), event(), '{"section":', '"last",', WAIT])
        result = await p._source_response(s, ['old', 'last'], .1)
        self.assertEqual(result, '{"section":"last"}')
        self.assertTrue(stream.closed)
        self.assertEqual(stream.read, 4)
        create.assert_awaited_once_with(model=p.llm.model,
            messages=[{'role': 'system', 'content': p.source_extract_system},
                      {'role': 'user', 'content': json.dumps(p._source_payload(s, ['old', 'last']), ensure_ascii=False)}],
            extra_body={'reasoning_effort': 'low'}, max_completion_tokens=2000,
            stream=True, response_format={'type': 'json_object'})
        p.llm.chat.assert_not_awaited()

    async def test_other_first_key_and_incomplete_eof_return_original(self):
        for parts in [[' {"reason":"ok",', '"section":"last"} '], ['{"section":"la']]:
            p, s, stream, create = case(parts)
            self.assertEqual(await p._source_response(s, ['last'], .1), ''.join(parts).strip())
            self.assertTrue(stream.closed)
            self.assertEqual(stream.read, len(parts))
            self.assertEqual(create.await_count, 1)

    async def test_outside_id_does_not_early_accept(self):
        p, s, stream, _ = case(['{"section":"outside",', '"reason":"ok"}'])
        self.assertIn('outside', await p._source_response(s, ['last'], .1))
        self.assertEqual(stream.read, 2)
        self.assertTrue(stream.closed)

    async def test_nontext_refusal_tool_or_filter_closes_and_raises(self):
        for chunk in [event(['text']), event(refusal='no'), event(tool_calls=[{}]),
                      event(function_call={"name":"x"}), event(finish_reason='content_filter'),
                      NS(choices=[NS(index=1, delta=NS(content='x'))]),
                      NS(choices=[NS(), NS()])]:
            p, s, stream, create = case([chunk])
            with self.assertRaises(ValueError):
                await p._source_response(s, ['last'], .1)
            self.assertTrue(stream.closed)
            self.assertEqual(create.await_count, 1)

    async def test_buffer_limit_closes(self):
        p, s, stream, _ = case(['x' * 16001])
        with self.assertRaises(ValueError):
            await p._source_response(s, ['last'], .1)
        self.assertTrue(stream.closed)

    async def test_body_error_closes_without_retry(self):
        p, s, stream, create = case([RuntimeError('mock transport')])
        with self.assertRaises(RuntimeError):
            await p._source_response(s, ['last'], .1)
        self.assertTrue(stream.closed)
        self.assertEqual(create.await_count, 1)

    async def test_timeout_closes(self):
        p, s, stream, create = case([WAIT])
        with self.assertRaises(TimeoutError):
            await p._source_response(s, ['last'], .01)
        self.assertTrue(stream.closed)
        self.assertEqual(create.await_count, 1)

    async def test_external_cancel_closes_and_propagates(self):
        p, s, stream, create = case([WAIT])
        task = asyncio.create_task(p._source_response(s, ['last'], .5))
        await stream.started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(stream.closed)
        self.assertEqual(create.await_count, 1)

    async def test_cancel_still_propagates_when_close_raises(self):
        p, s, stream, create = case([WAIT], close_error=RuntimeError('close'))
        task = asyncio.create_task(p._source_response(s, ['last'], .5))
        await stream.started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(stream.closed)
        self.assertEqual(create.await_count, 1)

    async def test_create_cancellation_propagates_without_acquired_handle(self):
        p, s, stream, create = case([])
        started, cancelled = asyncio.Event(), asyncio.Event()
        async def pending(**kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        create.side_effect = pending
        task = asyncio.create_task(p._source_response(s, ['last'], .5))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(cancelled.is_set())
        self.assertFalse(stream.closed)  # No stream handle was returned.
        self.assertEqual(create.await_count, 1)

    async def test_create_timeout_no_retry(self):
        p, s, _, create = case([])
        async def pending(**kwargs):
            await asyncio.Event().wait()
        create.side_effect = pending
        with self.assertRaises(TimeoutError):
            await p._source_response(s, ['last'], .01)
        self.assertEqual(create.await_count, 1)

    async def test_cleanup_failure_is_not_false_success(self):
        p, s, stream, create = case(['{"section":"last"}'], close_error=RuntimeError('close'))
        with self.assertRaises(RuntimeError):
            await p._source_response(s, ['last'], .1)
        self.assertTrue(stream.closed)
        self.assertEqual(create.await_count, 1)

    async def test_timeout_clamped_and_empty_budget_makes_no_call(self):
        p, s, stream, create = case(['{"section":"last"}'])
        original = asyncio.wait_for
        with patch('pipelines.kimjaehak.source_stream.asyncio.wait_for', wraps=original) as wait:
            await p._source_response(s, ['last'], 26)
            self.assertEqual(wait.call_args.kwargs['timeout'], 8)
        create.reset_mock()
        with self.assertRaises(TimeoutError):
            await p._source_response(s, ['last'], 0)
        create.assert_not_awaited()


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_hook_stays_nonstreaming_and_same_parameters(self):
        p, s, stream, create = case([], cls=SelectedSection)
        await p._generate(s)
        create.assert_not_awaited()
        args = p.llm.chat.call_args
        self.assertEqual(args.args, (p.source_extract_system, json.dumps(p._source_payload(s, ['old','last']), ensure_ascii=False)))
        self.assertEqual(args.kwargs, {'effort':'low', 'max_tokens':2000, 'timeout':8, 'json_mode':True})
        self.assertEqual(s.section, 'last')

    async def test_candidate_selects_real_excerpt_and_keeps_external_json(self):
        p, s, stream, create = case(['{"section":"last",', WAIT])
        with patch.object(Pipeline, '_excerpt', lambda self, state: self.searcher.section_text(state.section)):
            await p._generate(s)
        self.assertEqual(s.section, 'last')
        self.assertEqual(s.citations, ['last'])
        self.assertIn('last material', s.answer)
        self.assertTrue(stream.closed)
        self.assertIsInstance(json.loads(json.dumps(s.to_tutor_response())), dict)
        self.assertEqual(create.await_count, 1)

    async def test_invalid_error_falls_back_and_no_retry(self):
        p, s, stream, create = case(['{"section":"outside"}'])
        with patch.object(Pipeline, '_excerpt', lambda self, state: self.searcher.section_text(state.section)):
            await p._generate(s)
        self.assertEqual(s.section, 'old')
        self.assertEqual(s.errors, ['source_extract:ValueError'])
        self.assertTrue(stream.closed)
        self.assertEqual(create.await_count, 1)

    async def test_guard_policy_and_gates_skip(self):
        for mode in ['guard', 'policy', 'learned_section', 'tutor_section_gate', 'deadline']:
            p, s, stream, create = case([])
            if mode == 'guard': s._extract_guard = True
            elif mode == 'policy': s.label = Label.ANSWER_SEEKING
            elif mode == 'deadline': s._source_extract_deadline = time.monotonic()
            else: s.debug[mode] = True
            with patch.object(ContextFast, '_generate', new_callable=AsyncMock), patch.object(Pipeline, '_excerpt', return_value='old material'):
                await p._generate(s)
            create.assert_not_awaited()
            p.llm.chat.assert_not_awaited()
        self.assertIs(Pipeline.guard, SelectedFocused.guard)
        self.assertIs(Pipeline.tutor, SelectedFocused.tutor)
        self.assertEqual(Pipeline.deadline, 26)

    async def test_remaining_budget_and_cancellation_preserved(self):
        p, s, _, create = case([])
        s._source_extract_deadline = time.monotonic() + 2
        with patch.object(p, '_source_response', new_callable=AsyncMock, side_effect=asyncio.CancelledError) as response:
            with self.assertRaises(asyncio.CancelledError):
                await p._generate(s)
        self.assertGreater(response.call_args.kwargs['timeout'], 1)
        self.assertLessEqual(response.call_args.kwargs['timeout'], 1.5)
        self.assertEqual(s.errors, [])
        create.assert_not_awaited()


if __name__ == '__main__':
    unittest.main(verbosity=2)
