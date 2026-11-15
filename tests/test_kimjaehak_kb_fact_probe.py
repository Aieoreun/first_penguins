import unittest

from experiments.jh_generate_kb_facts import valid_examples


class QuotedFactTests(unittest.TestCase):
    def test_only_exact_source_quotes_without_section_leakage_survive(self):
        material = "가상 도서관에서는 한 번에 세 권을 빌려요. 대출 기간은 일주일이에요."
        rows = [
            {"quote": "한 번에 세 권을 빌려요.", "question": "가상 도서관의 대출 권수는 얼마인가요?", "answer_fact": "가상 도서관에서는 세 권을 빌린다."},
            {"quote": "대출 기간은 이주일이에요.", "question": "가상 도서관의 대출 기간은 얼마인가요?", "answer_fact": "대출 기간은 이주일이다."},
        ]
        self.assertEqual(valid_examples(rows, "LIB-01", material), rows[:1])
        leaked = [{**rows[0], "question": "LIB-01의 대출 권수는 얼마인가요?"}]
        self.assertEqual(valid_examples(leaked, "LIB-01", material), [])

    def test_malformed_or_duplicate_examples_are_not_counted(self):
        row = {"quote": "한 번에 세 권을 빌려요.", "question": "가상 도서관의 대출 권수는 얼마인가요?", "answer_fact": "가상 도서관에서는 세 권을 빌린다."}
        self.assertEqual(valid_examples([row, row], "LIB-01", row["quote"]), [row])
        self.assertEqual(valid_examples(None, "LIB-01", "교재"), [])
        self.assertEqual(valid_examples([{**row, "question": "위 문장에서 대출 권수는 얼마인가요?"}], "LIB-01", row["quote"]), [])


if __name__ == "__main__":
    unittest.main()
