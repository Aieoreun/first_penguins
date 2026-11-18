import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pipelines import load
Short = load('jh_r6_short_material')
NoSection = load('jh_r6_material_nosection')
Verify = load('jh_r6_context_verify')
from pipelines.kimjaehak.dual_granularity import All as DualAll
from pipelines.kimjaehak.material_budget import Pipeline as Base
from pipelines.kimjaehak.round6 import MATERIAL_NO_SECTION
from pipelines.v2 import prompts
from core.schema import Label
from core.state import Chunk
from core.search import Searcher


class Round6Tests(unittest.IsolatedAsyncioTestCase):
    def case(self, cls, guard=False, context=False):
        p = cls.__new__(cls)
        messages = [{'role':'user', 'content':'material question'}]
        if context:
            messages = [{'role':'user','content':'prior'}, {'role':'assistant','content':'prior answer'}] + messages
        s = p._new_state({'messages':messages}, guard)
        s.label, s.section = Label.NORMAL, 'S0'
        s.chunks = [Chunk('S0', '# S0\n## Heading\noriginal paragraph\n\nsecond paragraph')]
        s._dual_routes = ([], [], [Chunk('S0','# S0\n## Heading\nsecond paragraph')] +
                          [Chunk('S'+str(i), 'additional '+str(i)) for i in range(1,8)])
        p.searcher = Searcher(s.chunks + [Chunk('S1','material question answer')])
        p.llm = SimpleNamespace(chat=AsyncMock(return_value='{"section":"S1"}'))
        return p,s

    def test_short_evidence_preserves_original_and_skips_covered_text(self):
        p,s = self.case(Short)
        found = p._short_material_chunks(s)
        self.assertEqual(found[:1], s.chunks)
        self.assertEqual([c.section_id for c in found], ['S0','S1','S2','S3','S4'])
        self.assertEqual(len(s.chunks), 1)
        self.assertIsNot(found, s.chunks)

    async def test_only_tutor_material_payload_changes(self):
        for guard in (True, False):
            p,s = self.case(Short, guard)
            before = list(s.chunks)
            with patch.object(DualAll,'_collect_checks',new_callable=AsyncMock) as original:
                await p._collect_checks(s, 'check', 'original material')
                args = original.await_args.args
                self.assertEqual(args[:2], (s, 'check'))
                if guard:
                    self.assertEqual(args[2], 'original material')
                else:
                    self.assertIn('additional 1',args[2])
                    self.assertIn('original paragraph',args[2])
                    self.assertNotIn('additional 5',args[2])
            self.assertEqual(s.chunks, before)

    def test_missing_short_routes_preserves_material(self):
        p,s = self.case(Short)
        del s._dual_routes
        self.assertEqual(p._short_material_chunks(s),s.chunks)

    async def test_nosection_only_tutor_material_system_changes(self):
        for guard,name in ((False,'material'),(True,'material'),(False,'block'),(False,'other')):
            p,s = self.case(NoSection,guard)
            with patch.object(Base,'_ask',new_callable=AsyncMock,return_value={'quote':'x'}) as original:
                result = await p._ask(s,name,prompts.MATERIAL_SYSTEM,'user','low',10)
                args = original.await_args.args
                self.assertEqual(args[0:2],(s,name))
                self.assertEqual(args[2],MATERIAL_NO_SECTION if not guard and name=='material' else prompts.MATERIAL_SYSTEM)
                self.assertEqual(args[3:],('user','low',10))
                self.assertEqual(result,{'quote':'x'})

    def test_nosection_keeps_quote_asked_and_confidence_contract(self):
        self.assertNotIn('근거 섹션(section)',MATERIAL_NO_SECTION)
        self.assertNotIn('"section":',MATERIAL_NO_SECTION)
        for field in ('"asked":','"quote":','"confidence":'):
            self.assertIn(field,MATERIAL_NO_SECTION)

    async def test_context_verification_only_reopens_context_source_gate(self):
        for guard,context,label,expected in ((False,True,Label.NORMAL,True),
                (False,False,Label.NORMAL,False),(True,True,Label.NORMAL,False),
                (False,True,Label.ANSWER_SEEKING,False)):
            p,s = self.case(Verify,guard,context)
            s.label=label
            s.debug={'learned_section':True,'tutor_section_gate':.99,'other':'kept'}
            with patch.object(Base,'_generate',new_callable=AsyncMock):
                await p._generate(s)
            self.assertEqual('learned_section' not in s.debug,expected)
            self.assertEqual('tutor_section_gate' not in s.debug,expected)
            self.assertEqual(s.debug['other'],'kept')
            self.assertEqual(s.section,'S0')

    async def test_verification_uses_original_nonstream_source_call(self):
        p,s=self.case(Verify,context=True)
        s.debug['tutor_section_gate']=.99
        await p._generate(s)
        self.assertEqual(s.section,'S1')
        p.llm.chat.assert_awaited_once()
        self.assertEqual(p.llm.chat.await_args.kwargs,{'effort':'low','max_tokens':2000,'timeout':8,'json_mode':True})

    def test_existing_guard_and_request_limits_unchanged(self):
        for cls in (Short,NoSection,Verify):
            self.assertIs(cls.guard,Base.guard)
            self.assertIs(cls._check_output,Base._check_output)
            self.assertEqual((cls.deadline,cls.tutor_prepare_limit,cls.tutor_material_timeout),(26,16,13))


if __name__=='__main__':unittest.main()
