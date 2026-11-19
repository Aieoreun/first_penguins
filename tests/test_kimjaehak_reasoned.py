import asyncio
import unittest
from types import SimpleNamespace

from core.state import State
from pipelines.kimjaehak.reasoned import Pipeline, stream_text
from pipelines.kimjaehak.calibrated import BLOCK_SYSTEM
from tests.test_pipeline_v2 import SEARCHER


class Stream:
    def __init__(self, delay=0, parts=None):
        self.delay = delay
        self.closed = False
        self.parts = parts if parts is not None else ('{"category":', '"none"}')

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def __aiter__(self):
        yield SimpleNamespace(choices=[])
        await asyncio.sleep(self.delay)
        for content in self.parts:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


class StreamingTests(unittest.TestCase):
    def llm(self, stream, calls):
        async def create(**kwargs):
            calls.append(kwargs)
            return stream
        return SimpleNamespace(model="test-model", _chat=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))))

    def test_consumes_stream_and_preserves_medium_settings(self):
        stream, calls = Stream(), []
        state = State(messages=[])
        pipe = Pipeline(SEARCHER, self.llm(stream, calls))
        result = asyncio.run(pipe._ask(state, "block", "original", "request", "low", 10))
        self.assertEqual(result, {"category": "none"})
        self.assertEqual(state.errors, [])
        self.assertTrue(stream.closed)
        self.assertEqual(calls[0]["messages"][0]["content"], BLOCK_SYSTEM)
        self.assertEqual(calls[0]["extra_body"], {"reasoning_effort": "medium"})
        self.assertEqual(calls[0]["max_completion_tokens"], 4000)
        self.assertTrue(calls[0]["stream"])
        self.assertEqual(calls[0]["response_format"], {"type": "json_object"})

    def test_plain_answer_does_not_request_json_format(self):
        stream, calls = Stream(parts=["수업 ", "설명입니다."]), []
        result = asyncio.run(stream_text(self.llm(stream, calls), "system", "user",
                                         effort="medium", max_tokens=4000,
                                         timeout=1, json_mode=False))
        self.assertEqual(result, "수업 설명입니다.")
        self.assertNotIn("response_format", calls[0])
        self.assertTrue(stream.closed)

    def test_timeout_covers_stream_body_and_closes_connection(self):
        stream = Stream(delay=0.1)
        llm = self.llm(stream, [])
        with self.assertRaises(TimeoutError):
            asyncio.run(stream_text(llm, "system", "user", effort="medium", max_tokens=4000, timeout=0.01))
        self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
