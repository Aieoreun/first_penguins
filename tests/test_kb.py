"""교재 조각이 결정 6대로 만들어지는지 검사한다.

실행: python3 -m unittest tests.test_kb -v
"""
import json
import unittest
from pathlib import Path

from core.kb import load_chunks

ROOT = Path(__file__).resolve().parent.parent


class LoadChunks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = load_chunks(ROOT / "kb")

    def test_counts(self):
        self.assertEqual(len({c.section_id for c in self.chunks}), 48)
        self.assertEqual(sum(c.heading == "표" for c in self.chunks), 23)
        self.assertEqual(sum(c.heading != "표" for c in self.chunks), 196)

    def test_every_chunk_carries_its_titles(self):
        for c in self.chunks:
            self.assertTrue(c.text.startswith(f"# {c.section_id} {c.title}\n## {c.heading}\n"), c.section_id)
            self.assertTrue(c.title and c.heading, c.section_id)
            self.assertGreater(len(c.text), len(c.section_id) + len(c.title) + len(c.heading) + 10)

    def test_section_ids_are_file_names(self):
        files = {p.stem for p in (ROOT / "kb").glob("*/*.md")}
        self.assertEqual({c.section_id for c in self.chunks}, files)

    def test_no_text_is_lost(self):
        # 조각을 이어 붙이면 섹션 본문의 모든 줄이 들어 있어야 한다.
        for path in (ROOT / "kb").glob("*/*.md"):
            joined = "\n".join(c.text for c in self.chunks if c.section_id == path.stem)
            for line in path.read_text(encoding="utf-8").split("\n")[1:]:
                if line.strip():
                    self.assertIn(line.lstrip("# ").strip(), joined, path.stem)

    def test_public_evidence_sections_exist(self):
        ids = {c.section_id for c in self.chunks}
        with open(ROOT / "public" / "tutor.jsonl", encoding="utf-8") as f:
            for line in f:
                for sid in json.loads(line)["labels"]["evidence"]:
                    self.assertIn(sid, ids)


if __name__ == "__main__":
    unittest.main()
