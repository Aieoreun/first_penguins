import asyncio
import json
import unittest

from core.schema import Decision, Label
from pipelines.kimjaehak.pipeline import PeaceState
from pipelines.kimjaehak.guided import (
    GuidedInput, GuidedOutput, GuidedBoth, FocusInput, INPUT_SYSTEM, GROUND_SYSTEM, SAFE_SYSTEM,
)
from tests.test_pipeline import FakeLLM, QUESTION
from tests.test_pipeline_v2 import SEARCHER, TOP1


class GuidedTests(unittest.TestCase):
    def test_precise_output_uses_category_policy_and_keeps_safe_refusal(self):
        from pipelines.kimjaehak.precise_output import Pipeline, POLICY_SYSTEM, POLICY_HINTS
        system = POLICY_SYSTEM + "\n" + POLICY_HINTS["off_topic"]
        llm = FakeLLM({system: '{"safe":true}'})
        state = PeaceState(messages=QUESTION, candidate_answer="수업 내용으로 돌아가요.", label=Label.OFF_TOPIC)
        asyncio.run(Pipeline(SEARCHER, llm)._check_output(state))
        self.assertTrue(state.safe)
        self.assertEqual(llm.calls, [system])

    def test_precise_grounding_includes_conversation(self):
        from pipelines.kimjaehak.precise_output import Pipeline
        state = PeaceState(messages=QUESTION, candidate_answer="후보", section=TOP1)
        payload = Pipeline(SEARCHER, FakeLLM({}))._ground_payload(state)
        self.assertEqual(payload["conversation"], QUESTION)
        self.assertEqual(payload["candidate_answer"], "후보")

    def test_v4_output_keeps_deadlines_generation_and_independent_safety(self):
        from pipelines.kimjaehak.v4_output import Pipeline
        from pipelines.v4.pipeline import Pipeline as V4
        from pipelines.v2 import prompts as v2
        self.assertIs(Pipeline.tutor, V4.tutor)
        self.assertIs(Pipeline.guard, V4.guard)
        self.assertIs(Pipeline.answer_systems, v2.ANSWER_SYSTEMS)
        self.assertEqual(Pipeline.answer_effort, "low")
        llm = FakeLLM({SAFE_SYSTEM: '{"safe":true}'})
        state = PeaceState(messages=QUESTION, candidate_answer="거절합니다.", label=Label.HARMFUL_REQUEST)
        asyncio.run(Pipeline(SEARCHER, llm)._check_output(state))
        self.assertTrue(state.safe)
        self.assertEqual(llm.calls, [SAFE_SYSTEM])

    def test_focus_places_actual_last_request_after_evidence(self):
        def reply(payload):
            obj = json.loads(payload)
            self.assertEqual(list(obj)[-1], "classify_this_last_user_message")
            self.assertEqual(obj["classify_this_last_user_message"], QUESTION[-1]["content"])
            return '{"category":"normal","section":"","quote":""}'
        state = asyncio.run(FocusInput(SEARCHER, FakeLLM({INPUT_SYSTEM: reply})).guard({"messages": QUESTION}))
        self.assertEqual(state.errors, [])

    def test_nonstreaming_proxy_token_ceiling(self):
        class BoundedLLM(FakeLLM):
            async def chat(inner, system, user, **kwargs):
                self.assertLessEqual(kwargs["max_tokens"], 2000)
                return await super().chat(system, user, **kwargs)
        llm = BoundedLLM({INPUT_SYSTEM: '{"category":"normal","section":"","quote":""}'})
        state = asyncio.run(GuidedInput(SEARCHER, llm).guard({"messages": QUESTION}))
        self.assertEqual(state.errors, [])

    def test_harmful_speech_keeps_redirect_subtype(self):
        llm = FakeLLM({INPUT_SYSTEM: '{"category":"harmful_speech","section":"","quote":""}'})
        state = asyncio.run(GuidedInput(SEARCHER, llm).tutor({"messages": QUESTION}))
        self.assertEqual(state.to_tutor_response()["action"], "redirect")
        self.assertEqual(state.decision, Decision.BLOCK)

    def test_input_has_material_and_rejects_untrusted_section(self):
        def reply(payload):
            self.assertIn('material', payload)
            self.assertIn(TOP1, payload)
            return '{"category":"out_of_material","section":{},"quote":""}'
        llm = FakeLLM({INPUT_SYSTEM: reply})
        state = asyncio.run(GuidedInput(SEARCHER, llm).guard({"messages": QUESTION}))
        self.assertEqual(state.label, Label.OUT_OF_MATERIAL)
        self.assertEqual(state.section, TOP1)

    def test_grounding_returns_only_original_short_phrases(self):
        llm = FakeLLM({GROUND_SYSTEM: '{"unsupported":["다섯 권", "정답은 세 권"]}',
                       SAFE_SYSTEM: '{"safe":true}'})
        state = PeaceState(messages=QUESTION, candidate_answer="다섯 권을 빌릴 수 있어요.", section=TOP1)
        asyncio.run(GuidedOutput(SEARCHER, llm)._check_output(state))
        self.assertEqual(state.unsupported, ["다섯 권"])
        self.assertFalse(state.grounded)
        self.assertTrue(state.safe)

    def test_blocked_refusal_is_safe_and_skips_grounding(self):
        llm = FakeLLM({SAFE_SYSTEM: '{"safe":true}'})
        state = PeaceState(messages=QUESTION, candidate_answer="도울 수 없어요.",
                           label=Label.HARMFUL_REQUEST)
        asyncio.run(GuidedBoth(SEARCHER, llm)._check_output(state))
        self.assertTrue(state.safe)
        self.assertTrue(state.grounded)
        self.assertEqual(len(llm.calls), 1)

    def test_allowed_input_does_not_force_candidate_safe(self):
        llm = FakeLLM({GROUND_SYSTEM: '{"unsupported":[]}', SAFE_SYSTEM: '{"safe":false}'})
        state = PeaceState(messages=QUESTION, candidate_answer="후보", section=TOP1)
        asyncio.run(GuidedBoth(SEARCHER, llm)._check_output(state))
        self.assertFalse(state.safe)

    def test_one_stage_failure_preserves_successful_safety(self):
        llm = FakeLLM({GROUND_SYSTEM: TimeoutError(), SAFE_SYSTEM: '{"safe":true}'})
        state = PeaceState(messages=QUESTION, candidate_answer="후보", section=TOP1)
        asyncio.run(GuidedBoth(SEARCHER, llm)._check_output(state))
        self.assertTrue(state.safe)
        self.assertEqual(state.errors, ["guided_grounding:TimeoutError"])


if __name__ == "__main__":
    unittest.main()
