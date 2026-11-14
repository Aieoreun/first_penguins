"""검색, 구절 찾기, v1 요청 흐름 검사. 모델은 가짜를 쓴다(네트워크를 타지 않는다)."""
import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.kb import load_chunks  # noqa: E402
from core.schema import FALLBACK_ANSWER  # noqa: E402
from core.search import Searcher  # noqa: E402
from core.spans import locate, locate_all  # noqa: E402
from grader.scoring import parse_guard, parse_tutor  # noqa: E402
from pipelines.v1 import prompts  # noqa: E402
from pipelines.v1.pipeline import Pipeline  # noqa: E402

CHUNKS = load_chunks(ROOT / "kb")
SEARCHER = Searcher(CHUNKS)
QUESTION = [{"role": "user", "content": "다람쥐 첫걸음 통장의 하루 이체 한도는 얼마야"}]


class FakeLLM:
    """시스템 프롬프트별로 정해 둔 글을 돌려준다. 값이 예외면 던진다."""
    can_embed = False

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    async def chat(self, system, user, **kwargs):
        self.calls.append(system)
        reply = self.replies[system]
        if isinstance(reply, Exception):
            raise reply
        return reply(user) if callable(reply) else reply

    async def embed(self, texts, timeout):
        return None


def top_section():
    return SEARCHER.search(QUESTION[0]["content"])[0].section_id


def run(coro):
    return asyncio.run(coro)


class SearchTest(unittest.TestCase):
    def test_bm25_only_without_vectors(self):
        found = SEARCHER.search(QUESTION[0]["content"])
        self.assertTrue(0 < len(found) <= 10)
        self.assertTrue(found[0].section_id.startswith("FIN"))

    def test_embedding_adds_candidates(self):
        vectors = [[1.0, 0.0] if i == len(CHUNKS) - 1 else [0.0, 1.0] for i in range(len(CHUNKS))]
        s = Searcher(CHUNKS, vectors)
        found = s.search(QUESTION[0]["content"], [1.0, 0.0], embedding_top=1)
        self.assertIn(CHUNKS[-1], found)

    def test_section_text_has_every_chunk(self):
        sid = top_section()
        text = SEARCHER.section_text(sid)
        self.assertEqual(text.count(f"# {sid}"), 1)
        for c in CHUNKS:
            if c.section_id == sid:
                self.assertIn(c.text.split("\n", 1)[1][:30], text)


class SpanTest(unittest.TestCase):
    CAND = "이 과정의 학점 예제에서는 85점 이상이면 A를 받아요. 따라서 85점은 A 조건을 만족해요."

    def test_exact(self):
        self.assertEqual(locate(self.CAND, "85점 이상이면 A를 받아요"), "85점 이상이면 A를 받아요")

    def test_spacing_and_punctuation(self):
        self.assertEqual(locate(self.CAND, "85점이상이면 A를 받아요."), "85점 이상이면 A를 받아요")

    def test_close_match_returns_candidate_text(self):
        found = locate(self.CAND, "85점 이상이면 A를 받습니다")
        self.assertIsNotNone(found)
        self.assertIn(found, self.CAND)

    def test_unrelated_is_dropped(self):
        self.assertIsNone(locate(self.CAND, "전혀 상관없는 다른 문장입니다"))

    def test_locate_all_skips_bad_and_overlapping(self):
        self.assertEqual(locate_all(self.CAND, ["85점 이상이면 A를 받아요", "85점 이상", None, ""]),
                         ["85점 이상이면 A를 받아요"])


class TutorFlowTest(unittest.TestCase):
    def tutor(self, replies):
        llm = FakeLLM(replies)
        state = run(Pipeline(SEARCHER, llm).tutor({"messages": QUESTION}))
        response = state.to_tutor_response()
        self.assertIsNotNone(parse_tutor(response))
        return response, state, llm

    def test_normal_answer_cites_and_appends_section(self):
        sid = top_section()
        r, _, llm = self.tutor({
            prompts.CLASSIFY_SYSTEM: '{"category": "normal", "section": "%s"}' % sid,
            prompts.ANSWER_SYSTEM: "하루 이체 한도는 5만 원이에요. (XX-00-00)"})
        self.assertEqual(r["action"], "answer")
        self.assertEqual(r["citations"], [sid])
        self.assertEqual(r["answer"], f"하루 이체 한도는 5만 원이에요. ({sid})")
        self.assertEqual(len(llm.calls), 2)

    def test_unknown_section_falls_back_to_top_result(self):
        r, _, _ = self.tutor({
            prompts.CLASSIFY_SYSTEM: '{"category": "normal", "section": "ZZ-99-99"}',
            prompts.ANSWER_SYSTEM: "답이에요."})
        self.assertEqual(r["citations"], [top_section()])

    def test_fixed_answer_skips_generation(self):
        r, _, llm = self.tutor({prompts.CLASSIFY_SYSTEM: '{"category": "jailbreak_injection", "section": ""}'})
        self.assertEqual(r["action"], "refuse")
        self.assertEqual(r["citations"], [])
        self.assertEqual(len(llm.calls), 1)

    def test_classify_failure_is_normal(self):
        r, state, _ = self.tutor({
            prompts.CLASSIFY_SYSTEM: TimeoutError(),
            prompts.ANSWER_SYSTEM: "답이에요."})
        self.assertEqual(r["action"], "answer")
        self.assertIn("classify:TimeoutError", state.errors)

    def test_generation_failure_gives_apology_without_citation(self):
        r, _, _ = self.tutor({
            prompts.CLASSIFY_SYSTEM: '{"category": "normal", "section": "%s"}' % top_section(),
            prompts.ANSWER_SYSTEM: RuntimeError()})
        self.assertEqual(r, {"answer": FALLBACK_ANSWER, "action": "answer", "citations": []})

    def test_abstain_cites_only_with_second_sentence(self):
        sid = top_section()
        classify = '{"category": "out_of_material", "section": "%s"}' % sid
        one, _, _ = self.tutor({prompts.CLASSIFY_SYSTEM: classify,
                                prompts.ABSTAIN_SYSTEM: "그 내용은 자료에 나와 있지 않아요."})
        two, _, _ = self.tutor({prompts.CLASSIFY_SYSTEM: classify,
                                prompts.ABSTAIN_SYSTEM: "그 내용은 자료에 나와 있지 않아요. 하루 이체 한도는 안내되어 있어요."})
        self.assertEqual((one["action"], one["citations"]), ("abstain", []))
        self.assertEqual(two["citations"], [sid])
        self.assertNotIn(sid, two["answer"])

    def test_academic_dishonesty_is_generated_scaffold(self):
        r, _, _ = self.tutor({
            prompts.CLASSIFY_SYSTEM: '{"category": "academic_dishonesty", "section": ""}',
            prompts.SCAFFOLD_ACADEMIC_SYSTEM: "완성 답안 대신 힌트를 드릴게요."})
        self.assertEqual((r["action"], r["citations"]), ("scaffold", []))

    def test_bad_request_body_still_answers(self):
        llm = FakeLLM({prompts.CLASSIFY_SYSTEM: "{}", prompts.ANSWER_SYSTEM: ""})
        state = run(Pipeline(SEARCHER, llm).tutor("not a dict"))
        self.assertIsNotNone(parse_tutor(state.to_tutor_response()))


class GuardFlowTest(unittest.TestCase):
    CAND = "하루 이체 한도는 10만 원이에요."

    def guard(self, replies, candidate=CAND):
        llm = FakeLLM(replies)
        body = {"messages": QUESTION}
        if candidate is not None:
            body["candidate_answer"] = candidate
        state = run(Pipeline(SEARCHER, llm).guard(body))
        response = state.to_guard_response()
        self.assertIsNotNone(parse_guard(response, candidate is not None))
        return response, state, llm

    def test_no_candidate_no_output(self):
        r, _, llm = self.guard({prompts.CLASSIFY_SYSTEM: '{"category": "normal", "section": ""}'}, candidate=None)
        self.assertNotIn("output", r)
        self.assertEqual(len(llm.calls), 1)

    def test_allowed_input_finds_span_in_candidate(self):
        r, _, _ = self.guard({
            prompts.CLASSIFY_SYSTEM: '{"category": "normal", "section": "%s"}' % top_section(),
            prompts.UNSUPPORTED_SYSTEM: '{"unsupported": ["10만원이에요"]}'})
        self.assertEqual(r["output"], {"grounded": False, "unsupported": ["10만 원이에요"],
                                       "safe": True, "category": "normal"})

    def test_allowed_input_no_span_is_grounded(self):
        r, _, _ = self.guard({
            prompts.CLASSIFY_SYSTEM: '{"category": "normal", "section": ""}',
            prompts.UNSUPPORTED_SYSTEM: '{"unsupported": ["후보에 없는 엉뚱한 구절입니다"]}'})
        self.assertTrue(r["output"]["grounded"])
        self.assertEqual(r["output"]["unsupported"], [])

    def test_unsupported_failure_is_grounded(self):
        r, _, _ = self.guard({
            prompts.CLASSIFY_SYSTEM: '{"category": "normal", "section": ""}',
            prompts.UNSUPPORTED_SYSTEM: TimeoutError()})
        self.assertEqual((r["output"]["grounded"], r["output"]["safe"]), (True, True))

    def test_blocked_input_uses_complied_judgement(self):
        classify = '{"category": "harmful_request", "section": ""}'
        yes, _, _ = self.guard({prompts.CLASSIFY_SYSTEM: classify, prompts.COMPLIED_SYSTEM: '{"complied": true}'})
        no, _, _ = self.guard({prompts.CLASSIFY_SYSTEM: classify, prompts.COMPLIED_SYSTEM: '{"complied": false}'})
        self.assertEqual(yes["input"], {"decision": "block", "category": "harmful_content"})
        self.assertEqual((yes["output"]["safe"], yes["output"]["grounded"], yes["output"]["unsupported"]),
                         (False, True, []))
        self.assertTrue(no["output"]["safe"])

    def test_complied_failure_is_unsafe(self):
        r, _, _ = self.guard({
            prompts.CLASSIFY_SYSTEM: '{"category": "off_topic", "section": ""}',
            prompts.COMPLIED_SYSTEM: RuntimeError()})
        self.assertFalse(r["output"]["safe"])

    def test_candidate_is_part_of_search_query(self):
        seen = {}

        def classify(user):
            seen["user"] = user
            return '{"category": "normal", "section": ""}'
        self.guard({prompts.CLASSIFY_SYSTEM: classify, prompts.UNSUPPORTED_SYSTEM: '{"unsupported": []}'})
        self.assertNotIn(self.CAND, seen["user"])  # 후보 답변은 검색어에만 쓰고 분류기에는 보여 주지 않는다


if __name__ == "__main__":
    unittest.main()
