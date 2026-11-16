import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pipelines import load
Six, All = load('jh_r5_dual6'), load('jh_r5_dual_all')
from pipelines.kimjaehak.material_budget import Pipeline as Baseline
from pipelines.kimjaehak.dual_index import DenseView, candidate_pool, short_chunks, valid_vectors
from core.kb import KBChunk, load_chunks
from core.search import Searcher, load_vectors, save_vectors
from core.schema import Label


def chunk(sid, body, heading='Topic'):
    return KBChunk(sid, 'test', 'Title', heading, f'# {sid} Title\n## {heading}\n{body}')


class ShortIndexTests(unittest.TestCase):
    def test_atomic_qa_code_and_table_preserve_parent_and_text(self):
        code = '```python\nx = 1\n\nprint(x)\n```'
        qa = '### Q. Question?\nA. Answer.\n\nMore answer.'
        original = chunk('S1', 'Intro.\n\n'+code+'\n\n'+qa+'\n\n### Q. Next?\nA. Next.')
        parts = short_chunks([original], target=10)
        self.assertEqual(len(parts), 4)
        self.assertTrue(all(c.section_id == 'S1' and c.heading == 'Topic' for c in parts))
        self.assertTrue(any(code in c.text for c in parts))
        self.assertTrue(any(qa in c.text for c in parts))
        table = chunk('S2', 'name,value\nA,1\nB,2', '표')
        self.assertEqual(short_chunks([table], target=1), [table])

    def test_kb_content_is_retained_without_cross_parent_mixing(self):
        originals = load_chunks(Path(__file__).resolve().parents[1] / 'kb')
        parts = short_chunks(originals)
        self.assertGreater(len(parts), len(originals))
        for original in originals:
            children = [c for c in parts if c.section_id == original.section_id and c.heading == original.heading]
            joined = '\n'.join(c.text for c in children)
            for line in original.text.splitlines():
                if line.strip():
                    self.assertIn(line, joined)

    def test_cache_text_model_and_dimension_validation(self):
        chunks = [chunk('S1', 'hello')]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'index.json'
            save_vectors(path, 'model', chunks, [[1, 0]])
            self.assertIsNone(load_vectors(path, 'other-model', chunks))
            self.assertIsNone(load_vectors(path, 'model', [chunk('S1', 'changed')]))
            vectors = load_vectors(path, 'model', chunks)
            with self.assertRaises(ValueError):
                DenseView(chunks, vectors, dimension=3)
        for vectors in (None, [], [[0, 0]], [[float('nan'), 1]], [[1, 0], [1]]):
            self.assertFalse(valid_vectors(vectors, 1))

    def test_section_round_robin_prevents_duplicate_child_domination(self):
        routes = [[chunk(s, s) for s in ids] for ids in (
            ['B1', 'B1', 'B2', 'B3'], ['L1', 'L2', 'L3'], ['S1', 'S1', 'S2', 'S3'])]
        pool = candidate_pool('current', [chunk('last', '')], routes)
        self.assertEqual(pool, ['current','last','B1','L1','S1','B2','L2','S2','B3','L3','S3'])


class DualPipelineTests(unittest.IsolatedAsyncioTestCase):
    def make_case(self, cls=Six, guard=False):
        p = cls.__new__(cls)
        chunks = [chunk('S'+str(i), 'material '+str(i)) for i in range(12)]
        vectors = [[1, i/12] for i in range(12)]
        p.searcher = Searcher(chunks, vectors)
        p.long_view = DenseView(chunks, vectors)
        p.short_view = DenseView(chunks[::-1], vectors)
        p.llm = SimpleNamespace(can_embed=True, embed=AsyncMock(return_value=[[1, 0]]),
                                chat=AsyncMock(return_value='{"section":"S1"}'))
        s = p._new_state({'messages':[{'role':'user','content':'material 1'}]}, guard)
        s.label, s.section = Label.NORMAL, 'S0'
        return p, s

    async def test_single_embedding_and_input_chunks_equal_baseline(self):
        p,s = self.make_case()
        b,bs = self.make_case(Baseline)
        await p._search(s)
        await b._search(bs)
        self.assertEqual(s.chunks, bs.chunks)
        self.assertEqual(p.llm.embed.await_args, b.llm.embed.await_args)
        self.assertEqual(p.llm.embed.await_count, 1)
        self.assertEqual(len(s._dual_routes), 3)
        self.assertTrue(s._dual_routes[2])

    async def test_all_pool_is_identical_to_six_before_slicing(self):
        p,s = self.make_case()
        await p._search(s)
        six = p._source_candidates(s)
        all_ids = All._source_candidates(p, s)  # same instance retains six cap
        self.assertEqual(six, all_ids)
        p.source_extract_sections = None
        all_ids = p._source_candidates(s)
        self.assertGreater(len(all_ids), 6)
        self.assertEqual(six, all_ids[:6])
        self.assertEqual(len(all_ids), len(set(all_ids)))
        payload = p._source_payload(s, all_ids)
        self.assertEqual(payload['materials'][0]['text'], p.searcher.section_text(all_ids[0]))

    async def test_guard_exact_parent_search_without_short_lookup(self):
        p,s = self.make_case(guard=True)
        b,bs = self.make_case(Baseline, guard=True)
        with patch.object(p.short_view, 'search', side_effect=AssertionError('guard short lookup')):
            await p._search(s, extra='candidate')
        await b._search(bs, extra='candidate')
        self.assertEqual(s.chunks, bs.chunks)
        self.assertEqual(p.llm.embed.await_args, b.llm.embed.await_args)
        self.assertFalse(hasattr(s, '_dual_routes'))

    async def test_embedding_failure_uses_bm25_without_second_call(self):
        p,s = self.make_case()
        p.llm.embed.side_effect = TimeoutError()
        await p._search(s)
        self.assertEqual(s.errors, ['embed:TimeoutError'])
        self.assertEqual(s._dual_routes[1:], ([], []))
        self.assertEqual(p.llm.embed.await_count, 1)

    async def test_concurrent_queries_do_not_share_capture(self):
        p,s = self.make_case()
        other = p._new_state({'messages':[{'role':'user','content':'unmatched'}]}, False)
        async def embed(texts, timeout):
            await asyncio.sleep(0)
            return [[1, 0]] if texts[0] == 'material 1' else [[0, 1]]
        p.llm.embed.side_effect = embed
        await asyncio.gather(p._search(s), p._search(other))
        self.assertNotEqual(s._dual_routes[2][0], other._dual_routes[2][0])
        self.assertIsInstance(p.searcher, Searcher)

    async def test_nonstream_prompt_budgets_and_guard_behavior_inherited(self):
        for cls in (Six, All):
            for method in ('guard','_generate','_source_response','_validate_source','_source_payload',
                           '_prepare_within','_ask','_classify'):
                self.assertIs(getattr(cls, method), getattr(Baseline, method))
            self.assertEqual(cls.source_extract_system, Baseline.source_extract_system)
            self.assertEqual((cls.deadline,cls.tutor_prepare_limit,cls.source_extract_timeout), (26,16,8))
            self.assertFalse(getattr(cls, 'source_stream_early_finish', False))
            p,s = self.make_case(cls)
            await p._search(s)
            await p._generate(s)
            self.assertEqual(s.section, 'S1')
            self.assertEqual(p.llm.chat.await_args.kwargs,
                             {'effort':'low','max_tokens':2000,'timeout':8,'json_mode':True})
            ids = [m['section'] for m in json.loads(p.llm.chat.await_args.args[1])['materials']]
            self.assertEqual(len(ids), 6 if cls is Six else 12)


if __name__ == '__main__':
    unittest.main()
