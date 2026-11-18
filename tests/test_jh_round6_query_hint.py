import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pipelines.kimjaehak.round6_query_hint import SourceHint, HINT_RULE
from pipelines.kimjaehak.round6_retrieval import ResolvedFocused
from pipelines.kimjaehak.material_budget import Pipeline as Base


class QueryHintTests(unittest.IsolatedAsyncioTestCase):
    async def test_hint_only_survives_successful_resolved_search(self):
        p=SourceHint.__new__(SourceHint)
        for success in (True,False):
            state=SimpleNamespace(debug={},_resolved_source_hint='stale')
            async def search(state,extra=''):
                state._pending_resolved_hint='resolved question'
                state.debug['query_resolved']=success
            with patch.object(ResolvedFocused,'_search',side_effect=search):
                await p._search(state)
            self.assertEqual(state._resolved_source_hint,'resolved question' if success else None)
            self.assertIsNone(state._pending_resolved_hint)

    async def test_failed_search_drops_stale_hint(self):
        p=SourceHint.__new__(SourceHint)
        state=SimpleNamespace(debug={'query_resolved':True},_resolved_source_hint='stale')
        with patch.object(ResolvedFocused,'_search',side_effect=RuntimeError):
            with self.assertRaises(RuntimeError): await p._search(state)
        self.assertIsNone(state._resolved_source_hint)
        self.assertIsNone(state._pending_resolved_hint)

    async def test_validated_query_is_request_local(self):
        p=SourceHint.__new__(SourceHint); state=SimpleNamespace()
        with patch.object(ResolvedFocused,'_resolve_query',new=AsyncMock(return_value='current question')):
            self.assertEqual(await p._resolve_query(state),'current question')
        self.assertEqual(state._pending_resolved_hint,'current question')
        self.assertFalse(hasattr(p,'_pending_resolved_hint'))

    async def test_payload_and_instruction_only_change_with_hint(self):
        p=SourceHint.__new__(SourceHint)
        p.searcher=SimpleNamespace(section_text=lambda sid:'real textbook')
        p.llm=SimpleNamespace(chat=AsyncMock(return_value='{"section":"A"}'))
        state=SimpleNamespace(messages=[{'role':'user','content':'original question'}])
        for hint in (None,'resolved question'):
            state._resolved_source_hint=hint
            await p._source_response(state,['A'],3.5)
            args=p.llm.chat.call_args
            payload=json.loads(args.args[1])
            self.assertEqual(payload['conversation'],state.messages)
            self.assertEqual(payload['materials'],[{'section':'A','text':'real textbook'}])
            self.assertEqual(payload.get('retrieval_query'),hint)
            self.assertEqual(args.args[0],p.source_extract_system+(HINT_RULE if hint else ''))
            self.assertEqual(args.kwargs,{'effort':'low','max_tokens':2000,'timeout':3.5,'json_mode':True})
        self.assertNotIn(HINT_RULE,p.source_extract_system)
        self.assertIs(SourceHint.guard,Base.guard)
        self.assertIs(SourceHint._collect_checks,Base._collect_checks)


if __name__=='__main__': unittest.main()
