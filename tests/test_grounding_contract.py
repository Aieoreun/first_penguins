"""Exact claim quotation extracts source spans without changing grounding."""
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


def claims_response(*quotes):
    return {"claims": [{"claim_id": i, "sentence_id": sentence_id, "error_claim": text}
                       for i, (sentence_id, text) in enumerate(quotes)]}


class GroundingContract(unittest.IsolatedAsyncioTestCase):
    async def test_grounded_answer_skips_extraction(self):
        client, call = fake({"unsupported_ids": []})
        verdict = await check_grounding(client, "model", [], "안녕하세요.", [])
        self.assertTrue(verdict.grounded)
        self.assertEqual(verdict.unsupported, [])
        self.assertEqual(call.await_count, 1)

    async def test_evidence_passed_and_narrow_span_comes_from_candidate(self):
        client, call = fake({"unsupported_ids": [1]}, claims_response((1, "정답은 95점")))
        answer = "일반 안내입니다. 이 수업의 정답은 95점 이라고 합니다."
        evidence = [{"section_id": "SAMPLE", "text": "이 수업의 정답은 90점입니다."}]
        verdict = await check_grounding(client, "model", [], answer, evidence)
        self.assertFalse(verdict.grounded)
        self.assertEqual(verdict.unsupported, ["정답은 95점"])
        self.assertEqual(call.await_count, 2)
        initial_payload = json.loads(call.call_args_list[0].kwargs["messages"][1]["content"])
        self.assertEqual(initial_payload["evidence"], evidence)
        extraction_payload = json.loads(call.call_args_list[1].kwargs["messages"][1]["content"])
        self.assertEqual(extraction_payload, {"items": [{
            "sentence_id": 1,
            "error_description": "오류로 판정된 문장",
            "candidate_text": "이 수업의 정답은 95점 이라고 합니다.",
        }]})
        self.assertNotIn("evidence", extraction_payload)
        self.assertNotIn("conversation", extraction_payload)

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
        client, _ = fake({"grounded": True, "claims": []})
        with self.assertRaises(ValueError):
            await extract_unsupported(client, "model", ["정답은 95점입니다."], [0], [])

    async def test_invalid_extraction_preserves_false_and_original_sentence(self):
        for extraction in [
            {"grounded": True, "claims": []},
            claims_response(),
            claims_response((0, "원문에 없는 표현")),
            TimeoutError("test timeout"),
        ]:
            with self.subTest(extraction=extraction):
                client, _ = fake({"unsupported_ids": [0]}, extraction)
                verdict = await check_grounding(client, "model", [], "정답은 95점입니다.", [])
                self.assertFalse(verdict.grounded)
                self.assertEqual(verdict.unsupported, ["정답은 95점입니다."])

    async def test_model_cannot_supply_word_or_character_coordinates(self):
        for patch in [{"start_word": 0}, {"end_word": 0}, {"start_char": 0}, {"end_char": 3}]:
            with self.subTest(patch=patch):
                result = claims_response((0, "95점"))
                result["claims"][0].update(patch)
                client, _ = fake(result)
                with self.assertRaises(ValueError):
                    await extract_unsupported(client, "model", ["정답은 95점 입니다."], [0], [])

    async def test_quote_must_reference_selected_sentence_and_exact_source(self):
        sentences = ["일반 안내입니다.", "정답은 95점 입니다.", "   "]
        for quote in [(0, "일반"), (10, "95점"), (1, "90점"), (True, "95점"), (1, "")]:
            with self.subTest(quote=quote):
                client, _ = fake(claims_response(quote))
                with self.assertRaises(ValueError):
                    await extract_unsupported(client, "model", sentences, [1], [])
        client, _ = fake(claims_response((2, " ")))
        with self.assertRaises(ValueError):
            await extract_unsupported(client, "model", sentences, [2], [])

    async def test_source_lookup_preserves_original_internal_whitespace(self):
        original = "앞말  틀린\t  표현    뒷말입니다."
        client, _ = fake(claims_response((0, "틀린\t  표현")))
        spans = await extract_unsupported(client, "model", [original], [0], [])
        self.assertEqual(spans, ["틀린\t  표현"])
        self.assertIn(spans[0], original)

    async def test_each_selected_sentence_needs_a_claim(self):
        client, _ = fake(claims_response((0, "95점입니다.")))
        with self.assertRaisesRegex(ValueError, "span_missing_claim_sentence"):
            await extract_unsupported(client, "model", ["95점입니다.", "10명입니다."], [0, 1], [])

    async def test_multiple_errors_in_one_sentence(self):
        client, _ = fake(claims_response((0, "95점"), (0, "10명")))
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
            claims_response((1, "95점")),
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
        }]})
        self.assertNotIn("evidence", payload)
        self.assertNotIn("conversation", payload)
        self.assertFalse(verdict.grounded)
        self.assertEqual(verdict.unsupported, ["95점"])

    async def test_duplicate_ids_keep_explanations_with_their_original_sentence(self):
        client, call = fake(
            {"unsupported_ids": [0, 0, 1], "error_descriptions": ["원인A", "원인B", "원인C"]},
            claims_response((0, "95점입니다."), (1, "10명입니다.")),
        )
        verdict = await check_grounding(client, "model", [], "95점입니다. 10명입니다.", [])
        items = json.loads(call.call_args_list[1].kwargs["messages"][1]["content"])["items"]
        self.assertEqual([(i["sentence_id"], i["error_description"]) for i in items],
                         [(0, "원인A 원인B"), (1, "원인C")])
        self.assertFalse(verdict.grounded)
        self.assertEqual(verdict.unsupported, ["95점입니다.", "10명입니다."])
