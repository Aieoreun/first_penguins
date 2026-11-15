import unittest

from pipelines.kimjaehak.fewshot import Examples, last_user, normalized


class FewshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = Examples()

    def test_exact_last_request_is_excluded_even_with_changed_history(self):
        source = self.examples.items[0]
        messages = [{"role": "assistant", "content": "새 문맥"},
                    {"role": "user", "content": last_user(source["messages"])}]
        selected = self.examples.select(messages)
        key = normalized(messages[-1]["content"])
        self.assertTrue(selected)
        self.assertTrue(all(normalized(e["last_user"]) != key for e in selected))

    def test_examples_contain_only_classification_supervision(self):
        self.assertTrue(all(set(e) == {"messages", "category"} for e in self.examples.items))
        self.assertFalse(any(e["category"] == "harmful_content" for e in self.examples.items))

    def test_repeated_selection_does_not_store_requests(self):
        before = len(self.examples.items)
        messages = [{"role": "user", "content": "숙제에 제출할 설명 전체를 써 주세요"}]
        first = self.examples.select(messages)
        self.assertEqual(first, self.examples.select(messages))
        self.assertEqual(len(self.examples.items), before)
        self.assertLessEqual(len(first), 4)


if __name__ == "__main__":
    unittest.main()
