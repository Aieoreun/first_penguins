"""스키마와 state가 대회 규격·채점 코드와 맞는지 검사한다.

실행: python3 -m unittest tests.test_schema_state -v
"""
import json
import unittest
from pathlib import Path

from core.schema import (
    ANSWER_MAX_CHARS,
    FIXED_ANSWER_ACTIONS,
    POLICY,
    Action,
    Category,
    Decision,
    Label,
    parse_label,
)
from core.state import Chunk, State
from grader import scoring

ROOT = Path(__file__).resolve().parent.parent
HARMFUL = (Label.HARMFUL_REQUEST, Label.HARMFUL_SPEECH)


def load(name):
    with open(ROOT / "public" / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def labels_for(category):
    return [label for label, policy in POLICY.items() if policy.category.value == category]


class SchemaMatchesGrader(unittest.TestCase):
    def test_value_lists(self):
        self.assertEqual({c.value for c in Category}, set(scoring.CATEGORIES))
        self.assertEqual({a.value for a in Action}, set(scoring.ACTIONS))
        self.assertEqual({d.value for d in Decision}, set(scoring.DECISIONS))

    def test_policy_covers_every_label_and_category(self):
        self.assertEqual(set(POLICY), set(Label))
        self.assertEqual({p.category for p in POLICY.values()}, set(Category))

    def test_only_harmful_is_split(self):
        for category in Category:
            expected = 2 if category is Category.HARMFUL_CONTENT else 1
            self.assertEqual(len(labels_for(category.value)), expected, category)
        self.assertEqual(set(labels_for("harmful_content")), set(HARMFUL))

    def test_guard_decision_matches_public_set(self):
        for item in load("guard.jsonl"):
            lab = item["labels"]
            for label in labels_for(lab["category"]):
                self.assertEqual(POLICY[label].decision.value, lab["decision"], item["id"])

    def test_tutor_action_matches_public_set(self):
        # 범주마다 정답 action이 표에 있는 action 중 하나여야 한다. 유해만 둘이고 나머지는 하나다.
        for item in load("tutor.jsonl"):
            lab = item["labels"]
            allowed = {POLICY[label].action.value for label in labels_for(lab["category"])}
            self.assertIn(lab["action"], allowed, item["id"])
        self.assertEqual({POLICY[label].action for label in HARMFUL}, {Action.REFUSE, Action.REDIRECT})

    def test_fixed_answers(self):
        for label, policy in POLICY.items():
            has_fixed = policy.fixed_answer is not None
            self.assertEqual(has_fixed, policy.action in FIXED_ANSWER_ACTIONS, label)
            if has_fixed:
                self.assertTrue(0 < len(policy.fixed_answer) <= ANSWER_MAX_CHARS, label)


class ParseLabel(unittest.TestCase):
    def test_json(self):
        self.assertEqual(parse_label('{"category": "off_topic"}'), Label.OFF_TOPIC)
        self.assertEqual(parse_label('```json\n{"category":"normal"}\n```'), Label.NORMAL)
        self.assertEqual(parse_label('{"category": " Harmful_Speech "}'), Label.HARMFUL_SPEECH)

    def test_bare_name(self):
        self.assertEqual(parse_label("out_of_material"), Label.OUT_OF_MATERIAL)
        self.assertEqual(parse_label("harmful_request"), Label.HARMFUL_REQUEST)

    def test_spec_category_name_is_accepted(self):
        self.assertEqual(parse_label('{"category": "harmful_content"}'), Label.HARMFUL_REQUEST)

    def test_unusable(self):
        for text in (None, "", "모르겠어요", '{"category": "weird"}', "normal 또는 off_topic", "{깨진"):
            self.assertIsNone(parse_label(text), text)


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

    def test_generated_answer_is_trimmed_and_citations_deduplicated(self):
        s = State(messages=[], label=Label.NORMAL, answer="가" * 5000,
                  citations=["PY-01-01", "PY-01-01", "", "FIN-02-03"], chunks=[Chunk("PY-01-01", "본문")])
        self.assertTrue(s.needs_generation)
        r = s.to_tutor_response()
        self.assertIsNotNone(scoring.parse_tutor(r))
        self.assertEqual(len(r["answer"]), 4000)
        self.assertEqual(r["citations"], ["PY-01-01", "FIN-02-03"])
        self.assertEqual(r["action"], "answer")

    def test_fixed_answer_replaces_generated_text(self):
        for label, policy in POLICY.items():
            if policy.fixed_answer is None:
                continue
            s = State(messages=[], label=label, answer="모델이 쓴 글", citations=["PY-01-01"])
            self.assertFalse(s.needs_generation)
            r = s.to_tutor_response()
            self.assertEqual(r, {"answer": policy.fixed_answer, "action": policy.action.value, "citations": []})
            self.assertIsNotNone(scoring.parse_tutor(r))

    def test_every_label_passes_grader_parser(self):
        for label in Label:
            for candidate in (None, "후보 답변"):
                s = State(messages=[], candidate_answer=candidate, label=label)
                r = s.to_guard_response()
                self.assertIsNotNone(scoring.parse_guard(r, candidate is not None))
                self.assertEqual("output" in r, candidate is not None)
                self.assertIsNotNone(scoring.parse_tutor(s.to_tutor_response()))

    def test_harmful_labels_share_category_but_not_action(self):
        request = State(messages=[], label=Label.HARMFUL_REQUEST)
        speech = State(messages=[], label=Label.HARMFUL_SPEECH)
        for s in (request, speech):
            self.assertEqual(s.to_guard_response()["input"], {"decision": "block", "category": "harmful_content"})
        self.assertEqual(request.to_tutor_response()["action"], "refuse")
        self.assertEqual(speech.to_tutor_response()["action"], "redirect")

    def test_unclassified_falls_back_to_normal(self):
        s = State(messages=[])
        self.assertEqual(s.to_guard_response()["input"], {"decision": "allow", "category": "normal"})
        self.assertEqual(s.to_tutor_response()["action"], "answer")


if __name__ == "__main__":
    unittest.main()
