import asyncio
import io
import json
import logging
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import guard_classifier as gc
import guard_output as go
import guard_runtime as rt
import guard_supervisor as gs
from guard_schemas import InputVerdict


def fake_client(*outputs):
    replies = [SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content=json.dumps(value)))])
        for value in outputs]
    call = AsyncMock(side_effect=replies)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=call))), call


class RuntimeBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_review_deadline_is_separate_and_restores_base_scope(self):
        with rt.request_scope(time.monotonic() + .1, "same-request"):
            with rt.deadline_scope(time.monotonic() + 5):
                self.assertGreater(rt.remaining_time(), 4)
                self.assertEqual(rt._request.get()["id"], "same-request")
            self.assertLess(rt.remaining_time(), .2)

    async def test_schema_contains_only_candidate_ids(self):
        client, call = fake_client({"unsupported_ids": [], "error_descriptions": []})
        await go.check_grounding(client, "m", [], "문장 하나. 문장 둘.", [])
        schema = call.call_args.kwargs["response_format"]["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["unsupported_ids"]["items"]["enum"], [0, 1])

    async def test_empty_answer_does_not_send_invalid_empty_enum(self):
        client, call = fake_client()
        result = await go.check_grounding(client, "m", [], " \n ", [])
        self.assertTrue(result.grounded)
        call.assert_not_awaited()

    async def test_no_extraction_budget_keeps_confirmed_error_sentence(self):
        client, call = fake_client({"unsupported_ids": [0]})
        with rt.request_scope(time.monotonic() + .5):
            result = await go.check_grounding(client, "m", [], "값은 99입니다.", [])
        self.assertFalse(result.grounded)
        self.assertEqual(result.unsupported, ["값은 99입니다."])
        self.assertEqual(call.await_count, 1)

    async def test_extraction_timeout_respects_parent_deadline_and_retains_selection(self):
        client, _ = fake_client({"unsupported_ids": [0]})
        async def slow(*args, **kwargs):
            await asyncio.sleep(5)
        with rt.request_scope(time.monotonic() + 1.35), \
             patch.object(go, "extract_unsupported", side_effect=slow):
            start = time.monotonic()
            result = await go.check_grounding(client, "m", [], "값은 99입니다.", [])
        self.assertLess(time.monotonic() - start, 1.35)
        self.assertEqual(result.unsupported, ["값은 99입니다."])

    async def test_material_budget_shortage_preserves_input_model_verdict(self):
        expected = InputVerdict(decision="allow", category="normal")
        searcher = SimpleNamespace(search=AsyncMock())
        with rt.request_scope(time.monotonic() + .4), \
             patch.object(gc, "_call_json", AsyncMock(return_value=expected)) as model:
            result = await gc.classify_input(None, "m", [], searcher=searcher)
        self.assertEqual(result, expected)
        self.assertEqual(model.await_count, 1)
        searcher.search.assert_not_awaited()

    async def test_retry_reuses_search_and_does_not_repeat_completed_stages(self):
        searcher = SimpleNamespace(search=AsyncMock(return_value=[]))
        supervisor = gs.GuardSupervisor(None, "m", searcher, review_mode="off")
        with patch.object(gs, "classify_input", AsyncMock(return_value=InputVerdict(decision="allow", category="normal"))) as inp, \
             patch.object(gs, "check_safety", AsyncMock(return_value=go.SafetyVerdict(safe=True))) as safe, \
             patch.object(gs, "check_grounding", AsyncMock(side_effect=[ValueError("bad"), go.GroundingVerdict(grounded=True, unsupported=[])])) as ground:
            result = await supervisor.evaluate([], "答え")
        self.assertTrue(result["output"]["grounded"])
        self.assertEqual((inp.await_count, safe.await_count, ground.await_count), (1, 1, 2))
        searcher.search.assert_awaited_once()  # Empty search is also a cached result.

    async def test_exhausted_tokens_are_not_retried_unchanged(self):
        supervisor = gs.GuardSupervisor(None, "m", None, review_mode="off")
        op = AsyncMock(side_effect=go.ModelResponseError("unexpected_finish:length"))
        with self.assertRaises(ValueError):
            await supervisor._stage(op, time.monotonic() + 10, "grounding")
        op.assert_awaited_once()

    async def test_insufficient_retry_budget_does_not_start_second_call(self):
        supervisor = gs.GuardSupervisor(None, "m", None, review_mode="off")
        op = AsyncMock(side_effect=ValueError("bad"))
        with self.assertRaises(ValueError):
            await supervisor._stage(op, time.monotonic() + 1, "grounding")
        op.assert_awaited_once()

    async def test_failure_never_fabricates_uncomputed_grounding(self):
        supervisor = gs.GuardSupervisor(None, "m", SimpleNamespace(search=AsyncMock(return_value=[])), review_mode="off")
        with patch.object(gs, "classify_input", AsyncMock(return_value=InputVerdict(decision="allow", category="normal"))), \
             patch.object(gs, "check_safety", AsyncMock(return_value=go.SafetyVerdict(safe=True))), \
             patch.object(gs, "check_grounding", AsyncMock(side_effect=ValueError("bad"))):
            with self.assertRaises(ValueError):
                await supervisor.evaluate([], "答え")

    async def test_metadata_is_content_free_and_request_context_isolated(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        old_level = rt.logger.level
        rt.logger.setLevel(logging.INFO)
        rt.logger.addHandler(handler)
        try:
            async def one(i):
                with rt.request_scope(time.monotonic() + 5, str(i)):
                    rt.record("start", "input", prompt="SECRET", verdict="SECRET")
                    await asyncio.sleep(0)
                    try:
                        async with rt.timed_stage("input"):
                            raise ValueError("SECRET")
                    except ValueError:
                        pass
            await asyncio.gather(one(1), one(2))
            self.assertIsNone(rt._request.get())
            data = stream.getvalue()
            self.assertNotIn("SECRET", data)
            rows = [json.loads(line.split("guard_event ")[1]) for line in data.splitlines()]
            self.assertEqual({row["request_id"] for row in rows}, {"1", "2"})
            self.assertEqual(sum(row["event"] == "failure" for row in rows), 2)
        finally:
            rt.logger.removeHandler(handler)
            rt.logger.setLevel(old_level)
