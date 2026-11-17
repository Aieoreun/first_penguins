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



class V5Test(unittest.TestCase):
    """v5는 v4에서 정답만 요구의 힌트 답변 프롬프트만 바꾼 것이다."""

    def test_only_answer_seeking_prompt_differs(self):
        from pipelines.v5 import prompts as v5_prompts
        changed = [label for label, system in prompts.ANSWER_SYSTEMS.items() if v5_prompts.ANSWER_SYSTEMS[label] != system]
        self.assertEqual([label.value for label in changed], ["answer_seeking"])
        self.assertEqual(set(v5_prompts.ANSWER_SYSTEMS), set(prompts.ANSWER_SYSTEMS))

    def test_answer_seeking_uses_new_prompt(self):
        from pipelines.v5 import prompts as v5_prompts
        from pipelines.v5.pipeline import Pipeline as V5
        hint = "자료에는 하루 이체 한도가 정해져 있다고 나와 있어요. 그 한도와 비교해 보세요."
        llm = FakeLLM({prompts.BLOCK_SYSTEM: NONE_BLOCK, prompts.MATERIAL_SYSTEM: FOUND,
                       prompts.OTHER_SYSTEM: '{"category": "answer_seeking", "self_info": false}',
                       v5_prompts.SCAFFOLD_ANSWER_SEEKING_SYSTEM: hint})
        state = asyncio.run(V5(SEARCHER, llm).tutor({"messages": QUESTION}))
        response = parse_tutor(state.to_tutor_response())
        self.assertEqual(response["action"], "scaffold")
        self.assertIn(hint, response["answer"])
        self.assertEqual(len(llm.calls), 4)
        self.assertEqual(state.errors, [])


class V6Test(unittest.TestCase):
    """v6는 v5에서 정상 답변 프롬프트를 바꾸고, 교재에 없는 질문에 고정 문장을 낸다."""

    def setUp(self):
        from pipelines.v6 import prompts as v6_prompts
        from pipelines.v6.pipeline import Pipeline as V6
        self.V6, self.p = V6, v6_prompts

    def test_out_of_material_uses_fixed_sentence_without_model_call(self):
        llm = FakeLLM({prompts.BLOCK_SYSTEM: NONE_BLOCK, prompts.OTHER_SYSTEM: NONE_OTHER,
                       prompts.MATERIAL_SYSTEM: '{"asked": "x", "quote": "", "section": ""}'})
        state = asyncio.run(self.V6(SEARCHER, llm).tutor({"messages": QUESTION}))
        response = parse_tutor(state.to_tutor_response())
        self.assertEqual((response["action"], response["answer"], response["citations"]),
                         ("abstain", self.p.ABSTAIN_ANSWER, []))
        self.assertEqual(len(llm.calls), 3)  # 확인 셋뿐. 답변 호출이 없다

    def test_time_limits(self):
        from pipelines.v4.pipeline import Pipeline as V4
        self.assertEqual((V4.material_timeout, V4.prepare_limit, V4.deadline), (10, 14, 24))  # v4는 그대로다
        self.assertEqual((self.V6.material_timeout, self.V6.prepare_limit, self.V6.deadline), (13, 14.5, 26))

    def test_normal_uses_short_answer_prompt(self):
        llm = FakeLLM({prompts.BLOCK_SYSTEM: NONE_BLOCK, prompts.OTHER_SYSTEM: NONE_OTHER, prompts.MATERIAL_SYSTEM: FOUND,
                       self.p.ANSWER_SYSTEM: "하루 이체 한도는 5만 원이에요."})
        state = asyncio.run(self.V6(SEARCHER, llm).tutor({"messages": QUESTION}))
        response = parse_tutor(state.to_tutor_response())
        self.assertEqual(response["action"], "answer")
        self.assertIn("5만 원", response["answer"])
        self.assertEqual(state.errors, [])


class V7Test(unittest.TestCase):
    """v7은 v6에서 차단 확인 프롬프트만 과목별로 바꾼 것이다."""

    def setUp(self):
        from pipelines.v7 import prompts as v7_prompts
        from pipelines.v7.pipeline import Pipeline as V7
        self.V7, self.p = V7, v7_prompts
        self.subject = TOP1.split("-")[0]

    def test_three_subject_prompts_extend_the_common_one(self):
        self.assertEqual(set(self.p.BLOCK_SYSTEMS), {"PY", "PR", "FIN"})
        for system in self.p.BLOCK_SYSTEMS.values():
            self.assertNotEqual(system, prompts.BLOCK_SYSTEM)
            self.assertIn("# 출력", system)
        self.assertEqual(self.p.block_system("없는 과목"), prompts.BLOCK_SYSTEM)

    def test_block_check_uses_the_subject_prompt(self):
        subject_system = self.p.BLOCK_SYSTEMS[self.subject]
        llm = FakeLLM({subject_system: '{"category": "harmful_request"}', prompts.OTHER_SYSTEM: NONE_OTHER,
                       prompts.MATERIAL_SYSTEM: FOUND})
        state = asyncio.run(self.V7(SEARCHER, llm).tutor({"messages": QUESTION}))
        self.assertIn(subject_system, llm.calls)
        self.assertNotIn(prompts.BLOCK_SYSTEM, llm.calls)
        self.assertEqual(state.to_tutor_response()["action"], "refuse")
        self.assertEqual(len(llm.calls), 3)  # 호출 수는 v6과 같다


class V8Test(unittest.TestCase):
    """v8은 v6에서 출력 가드가 잡은 구절을 코드로 짧게 다듬는다."""

    def test_trim(self):
        from pipelines.v8.pipeline import trim
        self.assertEqual(trim("추가 금액은 1천 원이에요"), "1천 원")
        self.assertEqual(trim("다람쥐 틴즈 체크카드는 만 14세 이상이 사용할 수 있어요."), "만 14세 이상이 사용할 수 있어요")
        self.assertEqual(trim("5만 원"), "5만 원")
        self.assertEqual(trim("한도는 5만"), "한도는 5만")  # 떼면 너무 짧아지니 그대로 둔다

    def test_spans_are_trimmed_and_still_in_candidate(self):
        from pipelines.v8.pipeline import Pipeline as V8
        candidate = "하루 이체 한도는 10만 원이에요. 천천히 확인해 보세요."
        llm = FakeLLM({prompts.BLOCK_SYSTEM: NONE_BLOCK, prompts.OTHER_SYSTEM: NONE_OTHER, prompts.MATERIAL_SYSTEM: FOUND,
                       prompts.UNSUPPORTED_SYSTEM: '{"unsupported": ["하루 이체 한도는 10만 원이에요"]}'})
        state = asyncio.run(V8(SEARCHER, llm).guard({"messages": QUESTION, "candidate_answer": candidate}))
        out = state.to_guard_response()["output"]
        self.assertEqual((out["grounded"], out["unsupported"]), (False, ["10만 원"]))
        self.assertIn(out["unsupported"][0], candidate)
        self.assertEqual(state.debug["raw_unsupported"], ["하루 이체 한도는 10만 원이에요"])


class VxTest(unittest.TestCase):
    """vx는 v8에 실험 스위치를 얹은 것이다. 스위치는 클래스 속성이라 여기서 켜고 끈다."""

    CANDIDATE = "하루 이체 한도는 10만 원이에요. 천천히 확인해 보세요."

    def make(self, **switches):
        from pipelines.vx.pipeline import Pipeline as Vx
        return type("VxCase", (Vx,), switches)

    def base(self, **more):
        replies = {prompts.BLOCK_SYSTEM: NONE_BLOCK, prompts.OTHER_SYSTEM: NONE_OTHER, prompts.MATERIAL_SYSTEM: FOUND,
                   prompts.UNSUPPORTED_SYSTEM: '{"unsupported": ["하루 이체 한도는 10만 원이에요"]}'}
        replies.update(more)
        return replies

    def guard(self, cls, llm):
        return asyncio.run(cls(SEARCHER, llm).guard({"messages": QUESTION, "candidate_answer": self.CANDIDATE}))

    def test_switches_off_is_v8(self):
        llm = FakeLLM(self.base())
        out = self.guard(self.make(), llm).to_guard_response()["output"]
        self.assertEqual((out["grounded"], out["unsupported"]), (False, ["10만 원"]))
        self.assertEqual(len(llm.calls), 4)

    def test_curriculum_prompt(self):
        from pipelines.vx import prompts as vx_prompts
        llm = FakeLLM(self.base(**{vx_prompts.BLOCK_KB: NONE_BLOCK}))
        self.guard(self.make(curriculum=True), llm)
        self.assertIn(vx_prompts.BLOCK_KB, llm.calls)
        self.assertNotIn(prompts.BLOCK_SYSTEM, llm.calls)

    def test_two_stage_second_call_only_when_gate_blocks(self):
        from pipelines.vx import prompts as vx_prompts
        gate = vx_prompts.gate_of(prompts.BLOCK_SYSTEM)
        llm = FakeLLM(self.base(**{gate: '{"block": false, "confidence": 90}'}))
        state = self.guard(self.make(two_stage=True), llm)
        self.assertNotIn(prompts.BLOCK_SYSTEM, llm.calls)
        self.assertEqual(state.decision.value, "allow")
        llm = FakeLLM(self.base(**{gate: '{"block": true, "confidence": 90}', prompts.BLOCK_SYSTEM: '{"category": "off_topic"}'}))
        state = self.guard(self.make(two_stage=True), llm)
        self.assertEqual(state.category.value, "off_topic")

    def test_low_confidence_flag_is_dropped(self):
        llm = FakeLLM(self.base(**{prompts.BLOCK_SYSTEM: '{"category": "off_topic", "confidence": 70}'}))
        self.assertEqual(self.guard(self.make(min_conf=85), llm).decision.value, "allow")
        llm = FakeLLM(self.base(**{prompts.BLOCK_SYSTEM: '{"category": "off_topic", "confidence": 90}'}))
        self.assertEqual(self.guard(self.make(min_conf=85), llm).decision.value, "block")

    def test_kept_category_survives_low_confidence(self):
        cls = self.make(min_conf=85, conf_keep=frozenset({"harmful_request"}))
        llm = FakeLLM(self.base(**{prompts.BLOCK_SYSTEM: '{"category": "harmful_request", "confidence": 70}'}))
        self.assertEqual(self.guard(cls, llm).decision.value, "block")
        llm = FakeLLM(self.base(**{prompts.BLOCK_SYSTEM: '{"category": "off_topic", "confidence": 70}'}))
        self.assertEqual(self.guard(cls, llm).decision.value, "allow")

    def test_material_prompt_without_section(self):
        from pipelines.vx import prompts as vx_prompts
        self.assertNotIn("section", vx_prompts.MATERIAL_NO_SECTION)
        self.assertIn('"quote"', vx_prompts.MATERIAL_NO_SECTION)
        llm = FakeLLM(self.base(**{vx_prompts.MATERIAL_NO_SECTION: '{"asked": "한도", "quote": "하루 이체 한도는 5만 원"}'}))
        state = self.guard(self.make(material_no_section=True), llm)
        self.assertNotIn(prompts.MATERIAL_SYSTEM, llm.calls)
        self.assertEqual(state.section, TOP1)

    def test_recheck_calls_twice_when_answers_agree(self):
        llm = FakeLLM(self.base())
        self.guard(self.make(input_recheck=True), llm)
        self.assertEqual(llm.calls.count(prompts.BLOCK_SYSTEM), 2)
        self.assertEqual(llm.calls.count(prompts.OTHER_SYSTEM), 2)

    def test_number_hint_and_two_sections(self):
        from pipelines.vx.pipeline import numbers_missing
        self.assertEqual(numbers_missing("한도는 10만 원이고 5만 원은 아니에요 (FIN-01-04)", "하루 한도는 5만 원"), ["10만"])
        seen = {}

        def answer(user):
            seen["user"] = user
            return '{"unsupported": ["하루 이체 한도는 77만 원이에요"]}'

        # 교재에 없는 숫자(77만)를 쓴다. 교재의 다른 곳에 나오는 숫자는 코드가 잡지 못한다
        candidate = "하루 이체 한도는 77만 원이에요. 천천히 확인해 보세요."
        llm = FakeLLM(self.base(**{prompts.UNSUPPORTED_SYSTEM: answer}))
        cls = self.make(num_hint=True, out_sections=2)
        state = asyncio.run(cls(SEARCHER, llm).guard({"messages": QUESTION, "candidate_answer": candidate}))
        out = state.to_guard_response()["output"]
        self.assertIn("[코드가 확인한 것]", seen["user"])
        self.assertIn("77만", seen["user"].split("[코드가 확인한 것]")[1])
        self.assertEqual((out["grounded"], out["unsupported"]), (False, ["77만 원"]))

    def test_output_votes_need_a_majority(self):
        replies = iter(['{"unsupported": ["하루 이체 한도는 10만 원이에요"]}', '{"unsupported": []}', '{"unsupported": []}'])
        llm = FakeLLM(self.base(**{prompts.UNSUPPORTED_SYSTEM: lambda user: next(replies)}))
        out = self.guard(self.make(out_votes=3), llm).to_guard_response()["output"]
        self.assertEqual((out["grounded"], out["unsupported"]), (True, []))


if __name__ == "__main__":
    unittest.main()
