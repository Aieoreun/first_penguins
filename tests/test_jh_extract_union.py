import unittest
from types import SimpleNamespace

from core.schema import Label
from core.state import State
from pipelines.kimjaehak.extractive_answer import QueryUnion, QueryUnionTables


class ExtractUnionTests(unittest.TestCase):
    def setup_case(self, cls):
        p=cls.__new__(cls)
        chunks=[SimpleNamespace(section_id='S',text='first',heading='one'),
                SimpleNamespace(section_id='S',text='second (표: S-table1.csv)',heading='two'),
                SimpleNamespace(section_id='S',text='table data',heading='표'),
                SimpleNamespace(section_id='other',text='wrong table',heading='표')]
        p.searcher=SimpleNamespace(chunks=chunks,bm25=SimpleNamespace(scores=lambda q:[3,0,0,100] if q=='earlier\nlast' else [0,3,0,100]))
        s=State(messages=[{'role':'user','content':'earlier'},{'role':'user','content':'last'}],label=Label.NORMAL,section='S')
        return p,s

    def test_union_uses_both_queries_and_keeps_source_order(self):
        p,s=self.setup_case(QueryUnion)
        self.assertEqual(p._excerpt(s),'first\n\nsecond (표: S-table1.csv)')
        self.assertEqual(s.section,'S')

    def test_reference_adds_only_same_section_tables(self):
        p,s=self.setup_case(QueryUnionTables)
        self.assertEqual(p._excerpt(s),'first\n\nsecond (표: S-table1.csv)\n\ntable data')
        p.searcher.chunks[1].text='second without reference'
        self.assertNotIn('table data',p._excerpt(s))

    def test_same_query_deduplicates_and_missing_source_is_empty(self):
        p,s=self.setup_case(QueryUnion)
        s.messages=[{'role':'user','content':'last'}]
        self.assertEqual(p._excerpt(s),'second (표: S-table1.csv)')
        s.section='missing'
        self.assertEqual(p._excerpt(s),'')

    def test_registry(self):
        from pipelines import load
        self.assertIs(load('jh_r3_extract_union'),QueryUnion)
        self.assertIs(load('jh_r3_extract_union_tables'),QueryUnionTables)
