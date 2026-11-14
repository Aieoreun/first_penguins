"""스키마와 state가 대회 규격·채점 코드와 맞는지 검사한다.

실행: python3 -m unittest tests.test_schema_state -v
"""
import json
import unittest
from collections import Counter
from pathlib import Path

from core.schema import POLICY, Action, Category, Decision, parse_category
from core.state import Chunk, State
from grader import scoring

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    with open(ROOT / "public" / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class SchemaMatchesGrader(unittest.TestCase):
    def test_value_lists(self):
        self.assertEqual({c.value for c in Category}, set(scoring.CATEGORIES))
        self.assertEqual({a.value for a in Action}, set(scoring.ACTIONS))
        self.assertEqual({d.value for d in Decision}, set(scoring.DECISIONS))

    def test_policy_covers_every_category(self):
        self.assertEqual(set(POLICY), set(Category))

    def test_guard_decision_matches_public_set(self):
        for item in load("guard.jsonl"):
            lab = item["labels"]
            self.assertEqual(POLICY[Category(lab["category"])][0].value, lab["decision"], item["id"])

    def test_tutor_action_matches_public_set(self):
        # harmful_content는 refuse와 redirect가 섞여 있어 기본값(refuse)으로는 다 맞지 않는다.
        miss = Counter()
        for item in load("tutor.jsonl"):
            lab = item["labels"]
            if POLICY[Category(lab["category"])][1].value != lab["action"]:
                miss[(lab["category"], lab["action"])] += 1
        self.assertEqual(set(miss), {("harmful_content", "redirect")})


class ParseCategory(unittest.TestCase):
    def test_json(self):
        self.assertEqual(parse_category('{"category": "off_topic"}'), Category.OFF_TOPIC)
        self.assertEqual(parse_category('```json\n{"category":"normal"}\n```'), Category.NORMAL)
        self.assertEqual(parse_category('{"category": " Harmful_Content "}'), Category.HARMFUL_CONTENT)

    def test_bare_name(self):
        self.assertEqual(parse_category("out_of_material"), Category.OUT_OF_MATERIAL)

    def test_unusable(self):
        for text in (None, "", "모르겠어요", '{"category": "weird"}', "normal 또는 off_topic", "{깨진"):
            self.assertIsNone(parse_category(text), text)


class StateResponses(unittest.TestCase):
    def test_reads_conversation(self):
        s = State.from_request({"messages": [
            {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]})
        self.assertEqual(s.last_user, "c")
        self.assertEqual([m["content"] for m in s.history], ["a", "b"])

    def test_bad_request_does_not_raise(self):
        for body in (None, {}, {"messages": "x"}, {"messages": [1, {"role": "user"}]}, {"candidate_answer": 3}):
            s = State.from_request(body)
            self.assertIsNotNone(scoring.parse_tutor(s.to_tutor_response()))
            self.assertIsNotNone(scoring.parse_guard(s.to_guard_response(), s.has_candidate))

    def test_tutor_response_passes_grader_parser(self):
        s = State(messages=[], category=Category.NORMAL, answer="가" * 5000,
                  citations=["PY-01-01", "PY-01-01", "", "FIN-02-03"], chunks=[Chunk("PY-01-01", "본문")])
        r = s.to_tutor_response()
        self.assertIsNotNone(scoring.parse_tutor(r))
        self.assertEqual(len(r["answer"]), 4000)
        self.assertEqual(r["citations"], ["PY-01-01", "FIN-02-03"])
        self.assertEqual(r["action"], "answer")

    def test_guard_response_passes_grader_parser(self):
        for category in Category:
            for candidate in (None, "후보 답변"):
                s = State(messages=[], candidate_answer=candidate, category=category)
                r = s.to_guard_response()
                self.assertIsNotNone(scoring.parse_guard(r, candidate is not None))
                self.assertEqual("output" in r, candidate is not None)

    def test_unclassified_falls_back_to_normal(self):
        s = State(messages=[])
        self.assertEqual(s.to_guard_response()["input"], {"decision": "allow", "category": "normal"})
        self.assertEqual(s.to_tutor_response()["action"], "answer")

    def test_action_override(self):
        s = State(messages=[], category=Category.HARMFUL_CONTENT, action_override=Action.REDIRECT)
        self.assertEqual(s.to_tutor_response()["action"], "redirect")
        self.assertEqual(s.to_guard_response()["input"]["decision"], "block")


if __name__ == "__main__":
    unittest.main()
