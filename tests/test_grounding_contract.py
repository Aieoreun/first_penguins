"""The extraction stage can narrow spans but cannot overturn grounding."""
import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from guard_output import check_grounding, extract_unsupported


def response(content):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content=json.dumps(content)))])


def fake(*contents):
    call = AsyncMock(side_effect=[
        value if isinstance(value, BaseException) else response(value)
        for value in contents
    ])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=call))), call


class GroundingContract(unittest.IsolatedAsyncioTestCase):
    async def test_grounded_answer_skips_extraction(self):
        client, call = fake({"unsupported_ids": []})
        verdict = await check_grounding(client, "model", [], "안녕하세요.", [])
        self.assertTrue(verdict.grounded)
        self.assertEqual(verdict.unsupported, [])
        self.assertEqual(call.await_count, 1)

    async def test_evidence_passed_and_narrow_span_comes_from_candidate(self):
        client, call = fake(
            {"unsupported_ids": [1]},
            {"errors": [{"sentence_id": 1, "start_word": 2, "end_word": 3}]},
        )
        answer = "일반 안내입니다. 이 수업의 정답은 95점 이라고 합니다."
        evidence = [{"section_id": "SAMPLE", "text": "이 수업의 정답은 90점입니다."}]
        verdict = await check_grounding(client, "model", [], answer, evidence)
        self.assertFalse(verdict.grounded)
        self.assertEqual(verdict.unsupported, ["정답은 95점"])
        self.assertEqual(call.await_count, 2)
        initial_payload = json.loads(call.call_args_list[0].kwargs["messages"][1]["content"])
        self.assertEqual(initial_payload["evidence"], evidence)
        extraction_payload = json.loads(call.call_args_list[1].kwargs["messages"][1]["content"])
        self.assertEqual(set(extraction_payload), {"items"})
        self.assertNotIn("evidence", extraction_payload)
        self.assertNotIn("conversation", extraction_payload)
        self.assertEqual(extraction_payload["items"], [{
            "sentence_id": 1,
            "error_description": "오류로 판정된 문장",
            "candidate_text": "이 수업의 정답은 95점 이라고 합니다.",
            "words": [{"id": i, "text": word} for i, word in enumerate(
                ["이", "수업의", "정답은", "95점", "이라고", "합니다."])],
        }])

    async def test_invalid_initial_sentence_id_rejected(self):
        client, _ = fake({"unsupported_ids": [50]})
        with self.assertRaises(ValueError):
            await check_grounding(client, "model", [], "안녕하세요.", [])

    async def test_initial_model_cannot_supply_redundant_grounded_field(self):
        for result in [
            {"grounded": True, "unsupported_ids": []},
            {"grounded": False, "unsupported_ids": [0]},
        ]:
            with self.subTest(result=result):
                client, _ = fake(result)
                with self.assertRaises(ValueError):
                    await check_grounding(client, "model", [], "안녕하세요.", [])

    async def test_extraction_cannot_add_grounded_field(self):
        client, _ = fake({"grounded": True, "errors": []})
        with self.assertRaises(ValueError):
            await extract_unsupported(client, "model", ["정답은 95점입니다."], [0], [])

    async def test_invalid_extraction_preserves_false_and_original_sentence(self):
        for extraction in [
            {"grounded": True, "errors": []},
            {"errors": []},
            {"errors": [{"sentence_id": 0, "start_word": 10, "end_word": 10}]},
            TimeoutError("test timeout"),
        ]:
            with self.subTest(extraction=extraction):
                client, _ = fake({"unsupported_ids": [0]}, extraction)
                verdict = await check_grounding(client, "model", [], "정답은 95점입니다.", [])
                self.assertFalse(verdict.grounded)
                self.assertEqual(verdict.unsupported, ["정답은 95점입니다."])

    async def test_ranges_must_reference_valid_words_in_selected_sentence(self):
        sentences = ["일반 안내입니다.", "정답은 95점 입니다.", "   "]
        for error in [
            {"sentence_id": 1, "start_word": 2, "end_word": 1},
            {"sentence_id": 1, "start_word": 0, "end_word": 3},
            {"sentence_id": 1, "start_word": -1, "end_word": 1},
            {"sentence_id": 1, "start_word": 0, "end_word": -1},
            {"sentence_id": 1, "start_word": True, "end_word": 1},
            {"sentence_id": True, "start_word": 0, "end_word": 1},
            {"sentence_id": 1, "start_word": 0, "end_word": False},
            {"sentence_id": 0, "start_word": 0, "end_word": 1},
            {"sentence_id": 10, "start_word": 0, "end_word": 1},
        ]:
            with self.subTest(error=error):
                client, _ = fake({"errors": [error]})
                with self.assertRaises(ValueError):
                    await extract_unsupported(client, "model", sentences, [1], [])
        client, _ = fake({"errors": [{"sentence_id": 2, "start_word": 0, "end_word": 0}]})
        with self.assertRaises(ValueError):
            await extract_unsupported(client, "model", sentences, [2], [])

    async def test_word_range_preserves_original_internal_whitespace(self):
        original = "앞말  틀린\t  표현    뒷말입니다."
        client, _ = fake({"errors": [{"sentence_id": 0, "start_word": 1, "end_word": 2}]})
        spans = await extract_unsupported(client, "model", [original], [0], [])
        self.assertEqual(spans, ["틀린\t  표현"])
        self.assertIn(spans[0], original)

    async def test_each_selected_sentence_needs_a_span(self):
        client, _ = fake({"errors": [{"sentence_id": 0, "start_word": 0, "end_word": 0}]})
        with self.assertRaises(ValueError):
            await extract_unsupported(client, "model", ["95점입니다.", "10명입니다."], [0, 1], [])

    async def test_multiple_errors_in_one_sentence(self):
        client, _ = fake({"errors": [
            {"sentence_id": 0, "start_word": 0, "end_word": 0},
            {"sentence_id": 0, "start_word": 2, "end_word": 2},
        ]})
        spans = await extract_unsupported(client, "model", ["95점 학생은 10명 이라고 합니다."], [0], [])
        self.assertEqual(spans, ["95점", "10명"])

    async def test_cancellation_is_not_hidden_by_fallback(self):
        client, _ = fake({"unsupported_ids": [0]}, asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await check_grounding(client, "model", [], "정답은 95점입니다.", [])

    async def test_initial_error_description_reaches_extractor_without_changing_grounded(self):
        reason = "자료의 기준 점수와 후보 답변의 기준 점수가 다릅니다."
        client, call = fake(
            {"unsupported_ids": [1], "error_descriptions": [reason]},
            {"errors": [{"sentence_id": 1, "start_word": 1, "end_word": 1}]},
        )
        verdict = await check_grounding(
            client, "model", [], "안내입니다. 기준은 95점 이라고 합니다.",
            [{"section_id": "SAMPLE", "text": "기준은 90점입니다."}],
        )
        payload = json.loads(call.call_args_list[1].kwargs["messages"][1]["content"])
        self.assertEqual(payload, {"items": [{
            "sentence_id": 1,
            "error_description": reason,
            "candidate_text": "기준은 95점 이라고 합니다.",
            "words": [{"id": i, "text": word} for i, word in enumerate(
                ["기준은", "95점", "이라고", "합니다."])],
        }]})
        self.assertNotIn("evidence", payload)
        self.assertNotIn("conversation", payload)
        self.assertFalse(verdict.grounded)
        self.assertEqual(verdict.unsupported, ["95점"])

    async def test_duplicate_ids_keep_explanations_with_their_original_sentence(self):
        client, call = fake(
            {"unsupported_ids": [0, 0, 1], "error_descriptions": ["원인A", "원인B", "원인C"]},
            {"errors": [{"sentence_id": 0, "start_word": 0, "end_word": 0},
                        {"sentence_id": 1, "start_word": 0, "end_word": 0}]},
        )
        verdict = await check_grounding(client, "model", [], "95점입니다. 10명입니다.", [])
        items = json.loads(call.call_args_list[1].kwargs["messages"][1]["content"])["items"]
        self.assertEqual([(i["sentence_id"], i["error_description"]) for i in items],
                         [(0, "원인A 원인B"), (1, "원인C")])
        self.assertFalse(verdict.grounded)
        self.assertEqual(verdict.unsupported, ["95점입니다.", "10명입니다."])
