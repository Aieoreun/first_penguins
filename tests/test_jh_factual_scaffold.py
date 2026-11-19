import unittest

from core.schema import Label
from pipelines import load
from pipelines.kimjaehak.factual_scaffold import ContextFast, Pipeline


class FactualScaffoldTests(unittest.TestCase):
    def test_only_answer_seeking_prompt_changes(self):
        self.assertIs(load("jh_r2_factual_scaffold"), Pipeline)
        self.assertEqual(set(Pipeline.answer_systems), set(ContextFast.answer_systems))
        changed = [label for label in Pipeline.answer_systems
                   if Pipeline.answer_systems[label] != ContextFast.answer_systems[label]]
        self.assertEqual(changed, [Label.ANSWER_SEEKING])
        for name in ("guard", "_classify", "_search", "_check_output", "_generate"):
            self.assertIs(getattr(Pipeline,name),getattr(ContextFast,name))

    def test_english_instructions_preserve_policy_and_guard(self):
        from pipelines.kimjaehak.english_answer import Pipeline as English
        self.assertIs(load("jh_r2_english_answer"), English)
        changed = {label for label in English.answer_systems
                   if English.answer_systems[label] != ContextFast.answer_systems[label]}
        self.assertEqual(changed, {Label.NORMAL, Label.PII_DISCLOSURE})
        for name in ("guard", "_classify", "_search", "_check_output", "_generate"):
            self.assertIs(getattr(English,name),getattr(ContextFast,name))
