"""Claims-only source mapping, structural coverage and immutable grounding."""
import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from guard_output import check_grounding, extract_unsupported


def claim(claim_id=0, sentence_id=0, text="95점"):
    return {"claim_id": claim_id, "sentence_id": sentence_id, "error_claim": text}


def selection(*claims):
    return {"claims": list(claims)}


def fake(*results):
    responses = [x if isinstance(x, BaseException) else SimpleNamespace(choices=[
        SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
            content=json.dumps(x, ensure_ascii=False)))]) for x in results]
    call = AsyncMock(side_effect=responses)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=call))), call


class ClaimSpanContract(unittest.IsolatedAsyncioTestCase):
    async def test_every_listed_claim_gets_a_source_range(self):
        client, call = fake(selection(claim(0, text="95점"), claim(1, text="10명")))
        spans = await extract_unsupported(client, "test", ["95점 학생은 10명 입니다."], [0], [])
        self.assertEqual(spans, ["95점", "10명"])
        self.assertEqual(call.await_count, 1)

    async def test_claim_order_does_not_change_source_sentence_mapping(self):
        client, _ = fake(selection(claim(9, 1, "10명"), claim(3, 0, "95점")))
        spans = await extract_unsupported(client, "test", ["95점입니다.", "10명입니다."], [0, 1], [])
        self.assertEqual(spans, ["10명", "95점"])

    async def test_missing_selected_sentence_rejected(self):
        client, _ = fake(selection(claim()))
        with self.assertRaisesRegex(ValueError, "span_missing_claim_sentence"):
            await extract_unsupported(client, "test", ["95점입니다.", "10명입니다."], [0, 1], [])

    async def test_empty_inventory_rejected_for_selected_sentence(self):
        client, _ = fake(selection())
        with self.assertRaisesRegex(ValueError, "span_missing_claim_sentence"):
            await extract_unsupported(client, "test", ["95점입니다."], [0], [])

    async def test_duplicate_claim_id_rejected(self):
        client, _ = fake(selection(claim(0, text="95점"), claim(0, text="10명")))
        with self.assertRaisesRegex(ValueError, "span_duplicate_claim"):
            await extract_unsupported(client, "test", ["95점 학생은 10명 입니다."], [0], [])

    async def test_unselected_sentence_rejected(self):
        client, _ = fake(selection(claim(0, 1, "정상")))
        with self.assertRaisesRegex(ValueError, "span_unknown_sentence"):
            await extract_unsupported(client, "test", ["95점입니다.", "정상 문장입니다."], [0], [])

    async def test_claim_must_exist_in_its_own_sentence(self):
        client, _ = fake(selection(claim(0, 0, "10명"), claim(1, 1, "95점")))
        with self.assertRaisesRegex(ValueError, "span_claim_not_in_source"):
            await extract_unsupported(client, "test", ["95점입니다.", "10명입니다."], [0, 1], [])

    async def test_strict_claim_field_types(self):
        for patch in [{"claim_id": -1}, {"claim_id": True}, {"claim_id": "0"},
                      {"sentence_id": -1}, {"sentence_id": True},
                      {"error_claim": ""}, {"error_claim": 95}]:
            with self.subTest(patch=patch):
                value = claim()
                value.update(patch)
                client, _ = fake(selection(value))
                with self.assertRaises(ValueError):
                    await extract_unsupported(client, "test", ["95점입니다."], [0], [])

    async def test_whitespace_only_quote_rejected(self):
        client, _ = fake(selection(claim(text="  ")))
        with self.assertRaisesRegex(ValueError, "span_empty_claim"):
            await extract_unsupported(client, "test", ["95점  입니다."], [0], [])

    async def test_condition_negation_and_internal_whitespace_preserved(self):
        source = "안내: 점수가\t80점 이상이  아니면 탈락합니다. 출처"
        quote = "점수가\t80점 이상이  아니면 탈락합니다."
        client, _ = fake(selection(claim(text=quote)))
        spans = await extract_unsupported(client, "test", [source], [0], [])
        self.assertEqual(spans, [quote])
        self.assertIn("아니면", spans[0])
        self.assertIn(spans[0], source)

    async def test_source_lookup_accepts_subword_character_ranges(self):
        client, _ = fake(selection(claim(text="85점 이상이면 A")))
        source = "이 수업에서는 85점 이상이면 A입니다."
        spans = await extract_unsupported(client, "test", [source], [0], [])
        self.assertEqual(spans, ["85점 이상이면 A"])
        self.assertNotIn("입니다", spans[0])

    async def test_source_lookup_does_not_normalize_or_paraphrase_quote(self):
        for quote in ["90점", "95점 입니다", "95점입니다!"]:
            with self.subTest(quote=quote):
                client, _ = fake(selection(claim(text=quote)))
                with self.assertRaisesRegex(ValueError, "span_claim_not_in_source"):
                    await extract_unsupported(client, "test", ["95점입니다."], [0], [])

    async def test_models_cannot_supply_coordinate_or_grounded_fields(self):
        for extra in [{"grounded": True}, {"grounded": False}, {"errors": []}]:
            with self.subTest(extra=extra):
                result = {**selection(claim()), **extra}
                client, _ = fake(result)
                with self.assertRaises(ValueError):
                    await extract_unsupported(client, "test", ["95점입니다."], [0], [])
        for extra in [{"start_word": 0}, {"end_word": 1}, {"start_char": 0}, {"end_char": 3}]:
            with self.subTest(extra=extra):
                client, _ = fake(selection({**claim(), **extra}))
                with self.assertRaises(ValueError):
                    await extract_unsupported(client, "test", ["95점입니다."], [0], [])

    async def test_duplicate_text_with_distinct_claim_ids_returns_one_span(self):
        client, _ = fake(selection(claim(0), claim(1)))
        spans = await extract_unsupported(client, "test", ["95점은 95점입니다."], [0], [])
        self.assertEqual(spans, ["95점"])

    async def test_grounded_true_skips_extractor(self):
        client, call = fake({"unsupported_ids": [], "error_descriptions": []})
        result = await check_grounding(client, "test", [], "안녕하세요.", [])
        self.assertTrue(result.grounded)
        self.assertEqual(result.unsupported, [])
        self.assertEqual(call.await_count, 1)

    async def test_successful_mapping_cannot_change_grounded_false(self):
        client, call = fake({"unsupported_ids": [0], "error_descriptions": ["점수 오류"]},
                            selection(claim()))
        result = await check_grounding(client, "test", [], "정답은 95점 입니다.", [])
        self.assertFalse(result.grounded)
        self.assertEqual(result.unsupported, ["95점"])
        self.assertEqual(call.await_count, 2)
        payload = json.loads(call.call_args_list[1].kwargs["messages"][1]["content"])
        self.assertEqual(set(payload), {"items"})
        self.assertNotIn("words", payload["items"][0])

    async def test_missing_sentence_and_bad_source_fall_back_without_reclassification(self):
        first = {"unsupported_ids": [0, 1], "error_descriptions": ["점수 오류", "인원 오류"]}
        for result in [selection(claim()), selection(claim(text="90점"), claim(1, 1, "10명"))]:
            with self.subTest(result=result):
                client, _ = fake(first, result)
                verdict = await check_grounding(client, "test", [], "95점입니다. 10명입니다.", [])
                self.assertFalse(verdict.grounded)
                self.assertEqual(verdict.unsupported, ["95점입니다.", "10명입니다."])

    async def test_timeout_falls_back_without_reclassification(self):
        client, _ = fake({"unsupported_ids": [0]}, TimeoutError("synthetic timeout"))
        result = await check_grounding(client, "test", [], "95점입니다.", [])
        self.assertFalse(result.grounded)
        self.assertEqual(result.unsupported, ["95점입니다."])

    async def test_cancellation_propagates(self):
        client, _ = fake({"unsupported_ids": [0]}, asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await check_grounding(client, "test", [], "95점입니다.", [])
