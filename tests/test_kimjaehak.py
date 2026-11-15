"""Regression tests for Peace module adapters; no API keys or network needed."""
import asyncio
import json
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

from core.schema import Decision, Label
from core.state import State
from pipelines.kimjaehak.pipeline import PeaceBoth, PeaceInput, PeaceOutput, PeaceState
from pipelines.kimjaehak.peace.guard_classifier import INPUT_POLICY
from pipelines.kimjaehak.peace.guard_output import GROUNDING_POLICY, SAFETY_POLICY
from pipelines.v1 import prompts as v1
from pipelines.v2 import prompts as v2
from tests.test_pipeline import FakeLLM, QUESTION
from tests.test_pipeline_v2 import SEARCHER, TOP1, NONE_BLOCK, NONE_OTHER, FOUND


class PeaceLLM(FakeLLM):
    model = "test-model"

    def __init__(self, replies):
        super().__init__(replies)
        self._chat = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))

    async def create(self, **kw):
        system, user = [m["content"] for m in kw["messages"]]
        text = await self.chat(system, user, effort="low", max_tokens=kw["max_completion_tokens"], timeout=10)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=text))])


class AdapterTests(unittest.TestCase):
    def test_baseline_is_original_v2(self):
        from pipelines import load
        from pipelines.v2.pipeline import Pipeline
        self.assertIs(load("jh_baseline"), Pipeline)

    def test_peace_input_preserves_independent_decision(self):
        llm = PeaceLLM({INPUT_POLICY: '{"decision":"block","category":"normal"}'})
        state = asyncio.run(PeaceInput(SEARCHER, llm).guard({"messages": QUESTION}))
        self.assertEqual(state.to_guard_response(), {"input": {"decision": "block", "category": "normal"}})
        self.assertEqual(state.section, TOP1)
        self.assertEqual(len(llm.calls), 1)

    def test_harmful_category_uses_existing_refusal_alias(self):
        llm = PeaceLLM({INPUT_POLICY: '{"decision":"block","category":"harmful_content"}'})
        state = asyncio.run(PeaceInput(SEARCHER, llm).tutor({"messages": QUESTION}))
        self.assertEqual(state.label, Label.HARMFUL_REQUEST)
        self.assertEqual(state.to_tutor_response()["action"], "refuse")

    def test_safe_refusal_to_blocked_input_is_safe(self):
        candidate = "도와드릴 수 없어요. 안전하게 배워요."
        llm = PeaceLLM({INPUT_POLICY: '{"decision":"block","category":"jailbreak_injection"}',
                       GROUNDING_POLICY: '{"grounded":true,"unsupported_ids":[]}',
                       SAFETY_POLICY: '{"safe":true}'})
        state = asyncio.run(PeaceBoth(SEARCHER, llm).guard({"messages": QUESTION, "candidate_answer": candidate}))
        self.assertTrue(state.safe)
        self.assertTrue(state.grounded)
        self.assertEqual(len(llm.calls), 3)

    def test_unsafe_answer_to_allowed_input_is_unsafe(self):
        llm = PeaceLLM({INPUT_POLICY: '{"decision":"allow","category":"normal"}',
                       GROUNDING_POLICY: '{"grounded":true,"unsupported_ids":[]}',
                       SAFETY_POLICY: '{"safe":false}'})
        state = asyncio.run(PeaceBoth(SEARCHER, llm).guard({"messages": QUESTION, "candidate_answer": "후보"}))
        self.assertFalse(state.safe)

    def test_grounding_returns_exact_sentence(self):
        llm = PeaceLLM({GROUNDING_POLICY: '{"grounded":false,"unsupported_ids":[1]}',
                       SAFETY_POLICY: '{"safe":true}'})
        state = PeaceState(messages=QUESTION, candidate_answer="맞는 문장. 틀린 문장!", section=TOP1)
        asyncio.run(PeaceBoth(SEARCHER, llm)._check_output(state))
        self.assertEqual(state.unsupported, ["틀린 문장!"])
        self.assertFalse(state.grounded)

    def test_invalid_grounding_keeps_successful_safety(self):
        llm = PeaceLLM({GROUNDING_POLICY: '{"grounded":false,"unsupported_ids":[99]}',
                       SAFETY_POLICY: '{"safe":false}'})
        state = PeaceState(messages=QUESTION, candidate_answer="후보", section=TOP1)
        asyncio.run(PeaceBoth(SEARCHER, llm)._check_output(state))
        self.assertTrue(state.grounded)
        self.assertFalse(state.safe)
        self.assertEqual(state.errors, ["peace_grounding:ValueError"])

    def test_input_only_keeps_v2_output_path(self):
        llm = PeaceLLM({INPUT_POLICY: '{"decision":"allow","category":"normal"}',
                       v1.UNSUPPORTED_SYSTEM: '{"unsupported":[]}'})
        state = asyncio.run(PeaceInput(SEARCHER, llm).guard({"messages": QUESTION, "candidate_answer": "후보"}))
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(state.safe)

    def test_output_only_keeps_v2_input_path(self):
        llm = PeaceLLM({v2.BLOCK_SYSTEM: NONE_BLOCK, v2.OTHER_SYSTEM: NONE_OTHER,
                       v2.MATERIAL_SYSTEM: FOUND, GROUNDING_POLICY: '{"grounded":true,"unsupported_ids":[]}',
                       SAFETY_POLICY: '{"safe":true}'})
        state = asyncio.run(PeaceOutput(SEARCHER, llm).guard({"messages": QUESTION, "candidate_answer": "후보"}))
        self.assertEqual(len(llm.calls), 5)
        self.assertEqual(state.label, Label.NORMAL)

    def test_timeout_preserves_other_result(self):
        llm = PeaceLLM({GROUNDING_POLICY: TimeoutError(), SAFETY_POLICY: '{"safe":true}'})
        state = PeaceState(messages=QUESTION, candidate_answer="후보", peace_decision=Decision.BLOCK)
        asyncio.run(PeaceBoth(SEARCHER, llm)._check_output(state))
        self.assertTrue(state.safe)
        self.assertEqual(state.errors, ["peace_grounding:TimeoutError"])

    def test_public_replay_keys_are_unique(self):
        from experiments.jh_compare import load_public, request_key
        items = load_public()
        self.assertEqual(len(items), len({request_key(p['endpoint'], p) for p in items}))

    def test_public_judge_checkpoint_reuses_only_identical_calls(self):
        from experiments.jh_compare import journal_judge
        calls = []
        def judge(kind, payload):
            calls.append((kind, payload))
            return {"ok": True}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "judge.jsonl"
            first = journal_judge(judge, path)
            first("faithful", {"answer": "one"})
            resumed = journal_judge(judge, path)
            self.assertEqual(resumed("faithful", {"answer": "one"}), {"ok": True})
            resumed("faithful", {"answer": "two"})
            resumed("complied", {"answer": "one"})
        self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
