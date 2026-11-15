"""v2 흐름 검사: 세 확인의 답을 우선순위대로 합치는지, 실패했을 때 무엇으로 보는지. 모델은 가짜를 쓴다."""
import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.kb import load_chunks  # noqa: E402
from core.schema import FALLBACK_ANSWER  # noqa: E402
from core.search import Searcher  # noqa: E402
from grader.scoring import parse_guard, parse_tutor  # noqa: E402
from pipelines.v1 import prompts as v1_prompts  # noqa: E402
from pipelines.v2 import prompts  # noqa: E402
from pipelines.v2.pipeline import Pipeline  # noqa: E402
from tests.test_pipeline import QUESTION, FakeLLM  # noqa: E402

SEARCHER = Searcher(load_chunks(ROOT / "kb"))
TOP1 = SEARCHER.search(QUESTION[0]["content"])[0].section_id
NONE_BLOCK = '{"category": "none"}'
NONE_OTHER = '{"category": "none", "self_info": false}'
FOUND = '{"asked": "한도", "quote": "하루 이체 한도는 5만 원", "section": "%s"}' % TOP1
MISSING = '{"asked": "한도", "quote": "", "section": "%s"}' % TOP1


def classify(block=NONE_BLOCK, other=NONE_OTHER, material=FOUND, **more):
    replies = {prompts.BLOCK_SYSTEM: block, prompts.OTHER_SYSTEM: other, prompts.MATERIAL_SYSTEM: material}
    replies.update(more)
    llm = FakeLLM(replies)
    state = asyncio.run(_prepare(Pipeline(SEARCHER, llm)))
    return state, llm


async def _prepare(pipe):
    from core.state import State
    state = State.from_request({"messages": QUESTION})
    await pipe.prepare(state)
    return state


class MergeTest(unittest.TestCase):
    def label(self, **kw):
        return classify(**kw)[0].label.value

    def test_nothing_flagged_is_normal(self):
        state, llm = classify()
        self.assertEqual(state.label.value, "normal")
        self.assertEqual(state.section, TOP1)
        self.assertEqual(len(llm.calls), 3)  # 세 확인을 한 번씩 부른다

    def test_block_beats_everything(self):
        self.assertEqual(self.label(block='{"category": "jailbreak_injection"}',
                                    other='{"category": "emotional_distress", "self_info": true}',
                                    material=MISSING), "jailbreak_injection")

    def test_distress_beats_academic(self):
        self.assertEqual(self.label(block='{"category": "academic_dishonesty"}',
                                    other='{"category": "emotional_distress", "self_info": false}'),
                         "emotional_distress")

    def test_academic_beats_answer_seeking(self):
        self.assertEqual(self.label(block='{"category": "academic_dishonesty"}',
                                    other='{"category": "answer_seeking", "self_info": false}'),
                         "academic_dishonesty")

    def test_out_of_material_beats_self_info(self):
        self.assertEqual(self.label(other='{"category": "none", "self_info": true}', material=MISSING),
                         "out_of_material")

    def test_self_info_alone_is_pii(self):
        self.assertEqual(self.label(other='{"category": "none", "self_info": true}'), "pii_disclosure")

    def test_empty_quote_is_out_of_material(self):
        self.assertEqual(self.label(material=MISSING), "out_of_material")


class FailureTest(unittest.TestCase):
    def test_failed_checks_fall_back_to_normal(self):
        state, _ = classify(block=TimeoutError(), other="not json", material=RuntimeError())
        self.assertEqual(state.label.value, "normal")  # 교재 확인이 실패하면 교재에 있는 것으로 본다
        self.assertEqual(state.section, TOP1)
        self.assertEqual(len(state.errors), 3)

    def test_unknown_names_are_ignored(self):
        state, _ = classify(block='{"category": "something_else"}', other='{"category": 3, "self_info": "yes"}')
        self.assertEqual(state.label.value, "normal")


class EndToEndTest(unittest.TestCase):
    def test_tutor_and_guard_responses_are_valid(self):
        replies = {prompts.BLOCK_SYSTEM: NONE_BLOCK, prompts.OTHER_SYSTEM: NONE_OTHER, prompts.MATERIAL_SYSTEM: FOUND,
                   v1_prompts.ANSWER_SYSTEM: "하루 이체 한도는 5만 원이에요.",
                   v1_prompts.UNSUPPORTED_SYSTEM: '{"unsupported": []}'}
        pipe = Pipeline(SEARCHER, FakeLLM(replies))
        tutor = asyncio.run(pipe.tutor({"messages": QUESTION})).to_tutor_response()
        self.assertIsNotNone(parse_tutor(tutor))
        self.assertEqual((tutor["action"], tutor["citations"]), ("answer", [TOP1]))
        guard = asyncio.run(pipe.guard({"messages": QUESTION, "candidate_answer": "5만 원이에요."})).to_guard_response()
        self.assertIsNotNone(parse_guard(guard, True))


class V3Test(unittest.TestCase):
    """v3는 v2에 설정을 고정한 것이다. 다수결, 순서, 안전 규칙을 확인한다."""

    def setUp(self):
        from pipelines.v3.pipeline import Pipeline as V3
        self.V3 = V3

    def prepare(self, replies):
        from core.state import State
        llm = FakeLLM(replies)
        state = State.from_request({"messages": QUESTION})
        asyncio.run(self.V3(SEARCHER, llm).prepare(state))
        return state, llm

    def test_three_votes_per_check(self):
        _, llm = self.prepare({prompts.BLOCK_SYSTEM: NONE_BLOCK, prompts.OTHER_SYSTEM: NONE_OTHER,
                               prompts.MATERIAL_SYSTEM: FOUND})
        self.assertEqual(len(llm.calls), 9)

    def test_majority_wins(self):
        answers = iter(['{"category": "off_topic"}', NONE_BLOCK, NONE_BLOCK])
        state, _ = self.prepare({prompts.BLOCK_SYSTEM: lambda user: next(answers),
                                 prompts.OTHER_SYSTEM: NONE_OTHER, prompts.MATERIAL_SYSTEM: FOUND})
        self.assertEqual(state.label.value, "normal")

    def test_answer_seeking_beats_academic(self):
        state, _ = self.prepare({prompts.BLOCK_SYSTEM: '{"category": "academic_dishonesty"}',
                                 prompts.OTHER_SYSTEM: '{"category": "answer_seeking", "self_info": false}',
                                 prompts.MATERIAL_SYSTEM: FOUND})
        self.assertEqual(state.label.value, "answer_seeking")

    def test_blocked_candidate_is_unsafe_without_model_call(self):
        llm = FakeLLM({prompts.BLOCK_SYSTEM: '{"category": "harmful_request"}', prompts.OTHER_SYSTEM: NONE_OTHER,
                       prompts.MATERIAL_SYSTEM: FOUND})
        state = asyncio.run(self.V3(SEARCHER, llm).guard({"messages": QUESTION, "candidate_answer": "네, 해 드릴게요."}))
        out = state.to_guard_response()["output"]
        self.assertEqual((out["safe"], out["grounded"], out["unsupported"]), (False, True, []))
        self.assertEqual(len(llm.calls), 9)  # 확인 호출뿐이다

    def test_uses_second_prompts(self):
        self.assertIs(self.V3.answer_systems, prompts.ANSWER_SYSTEMS)
        self.assertIs(self.V3.unsupported_system, prompts.UNSUPPORTED_SYSTEM)


class SlowLLM(FakeLLM):
    """정해 둔 시스템 프롬프트의 호출만 늦게 답한다."""

    def __init__(self, replies, slow, seconds):
        super().__init__(replies)
        self.slow, self.seconds = slow, seconds

    async def chat(self, system, user, **kwargs):
        if system in self.slow:
            await asyncio.sleep(self.seconds)
        return await super().chat(system, user, **kwargs)


class V4Test(unittest.TestCase):
    """v4는 v3에서 다수결을 빼고 요청 전체의 마감 시간을 둔 것이다."""

    CHECKS = {prompts.BLOCK_SYSTEM: NONE_BLOCK, prompts.OTHER_SYSTEM: NONE_OTHER, prompts.MATERIAL_SYSTEM: FOUND}

    def setUp(self):
        from pipelines.v4.pipeline import Pipeline as V4
        self.V4 = V4

    def answer_replies(self, text="하루 이체 한도는 5만 원이에요."):
        return dict(self.CHECKS, **{system: text for system in prompts.ANSWER_SYSTEMS.values()})

    def test_one_call_per_check(self):
        llm = FakeLLM(self.answer_replies())
        state = asyncio.run(self.V4(SEARCHER, llm).tutor({"messages": QUESTION}))
        self.assertEqual(len(llm.calls), 4)  # 확인 셋 + 답변 하나
        self.assertEqual(parse_tutor(state.to_tutor_response())["citations"], [TOP1])

    def test_answer_seeking_beats_academic(self):
        llm = FakeLLM(dict(self.answer_replies(), **{
            prompts.BLOCK_SYSTEM: '{"category": "academic_dishonesty"}',
            prompts.OTHER_SYSTEM: '{"category": "answer_seeking", "self_info": false}'}))
        state = asyncio.run(self.V4(SEARCHER, llm).tutor({"messages": QUESTION}))
        self.assertEqual(state.label.value, "answer_seeking")

    def test_blocked_candidate_is_unsafe_without_model_call(self):
        llm = FakeLLM(dict(self.CHECKS, **{prompts.BLOCK_SYSTEM: '{"category": "harmful_request"}'}))
        state = asyncio.run(self.V4(SEARCHER, llm).guard({"messages": QUESTION, "candidate_answer": "네, 해 드릴게요."}))
        out = state.to_guard_response()["output"]
        self.assertEqual((out["safe"], out["grounded"], out["unsupported"]), (False, True, []))
        self.assertEqual(len(llm.calls), 3)

    def test_slow_answer_hits_deadline(self):
        llm = SlowLLM(self.answer_replies(), set(prompts.ANSWER_SYSTEMS.values()), 1.0)
        pipe = self.V4(SEARCHER, llm)
        pipe.deadline = 0.2
        state = asyncio.run(pipe.tutor({"messages": QUESTION}))
        response = state.to_tutor_response()
        self.assertEqual((response["answer"], response["citations"]), (FALLBACK_ANSWER, []))
        self.assertIn("generate:Deadline", state.errors)

    def test_slow_checks_fall_back_to_normal(self):
        llm = SlowLLM(dict(self.answer_replies(), **{prompts.BLOCK_SYSTEM: '{"category": "harmful_request"}'}),
                      {prompts.BLOCK_SYSTEM}, 1.0)
        pipe = self.V4(SEARCHER, llm)
        pipe.prepare_limit, pipe.deadline = 0.2, 5
        state = asyncio.run(pipe.tutor({"messages": QUESTION}))
        self.assertEqual(state.label, None)  # 입력 가드가 못 끝냈다 → normal로 답한다
        self.assertEqual(state.to_tutor_response()["action"], "answer")
        self.assertEqual(state.section, TOP1)
        self.assertIn("prepare:Deadline", state.errors)

    def test_slow_output_check_hits_deadline(self):
        llm = SlowLLM(dict(self.CHECKS, **{prompts.UNSUPPORTED_SYSTEM: '{"unsupported": ["5만 원"]}'}),
                      {prompts.UNSUPPORTED_SYSTEM}, 1.0)
        pipe = self.V4(SEARCHER, llm)
        pipe.deadline = 0.2
        state = asyncio.run(pipe.guard({"messages": QUESTION, "candidate_answer": "한도는 5만 원이에요."}))
        out = state.to_guard_response()["output"]
        self.assertEqual((out["grounded"], out["unsupported"], out["safe"]), (True, [], True))
        self.assertIn("check_output:Deadline", state.errors)


if __name__ == "__main__":
    unittest.main()
