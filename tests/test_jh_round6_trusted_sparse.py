import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

import numpy as np
from core.search import Searcher
from core.state import Chunk
from pipelines.kimjaehak.material_budget import Pipeline as Base
from pipelines.kimjaehak.round6_trusted_sparse import Pipeline


def setup(cls, category='normal', confidence=.96, section_confidence=.92):
    p=cls.__new__(cls)
    p.searcher=Searcher([Chunk('A','title A\nalpha'),Chunk('B','title B\nbeta')],[[1.,0.],[0.,1.]])
    p.llm=SimpleNamespace(can_embed=True,embed=AsyncMock(return_value=[[1.,0.]]),
                          chat=AsyncMock(side_effect=AssertionError('Unexpected chat')))
    p.vectorizer=p.section_vectorizer=SimpleNamespace(transform=lambda text:text)
    p.classifier=SimpleNamespace(classes_=np.array([category,'off_topic']),
                                predict_proba=lambda text:np.array([[confidence,1-confidence]]))
    p.section_classifier=SimpleNamespace(classes_=np.array(['B','A']),
                                predict_proba=lambda text:np.array([[section_confidence,1-section_confidence]]))
    p._known_sections={'A','B'}
    return p


class TrustedTests(unittest.IsolatedAsyncioTestCase):
    async def test_final_response_equal_through_real_classification_and_generation(self):
        for category in ('normal','pii_disclosure'):
            for confidence in (.90,.92,.99):
                base,fast=setup(Base,category,section_confidence=confidence),setup(Pipeline,category,section_confidence=confidence)
                responses=[]
                for p in (base,fast):
                    state=p._new_state({'messages':[{'role':'user','content':'alpha question'}]},False)
                    await p._search(state); await p._classify(state); await p._generate(state)
                    responses.append(state.to_tutor_response())
                    p.llm.chat.assert_not_awaited()
                self.assertEqual(responses[0],responses[1])
                self.assertEqual(responses[1]['citations'],['B'])
                base.llm.embed.assert_awaited_once();fast.llm.embed.assert_not_awaited()

    async def test_below_threshold_unknown_or_guard_uses_original_embedding(self):
        for change in ('input','section','unknown','guard','category','require_retrieved'):
            p=setup(Pipeline,confidence=.949 if change=='input' else .96,
                    section_confidence=.899 if change=='section' else .92,
                    category='emotional_distress' if change=='category' else 'normal')
            if change=='unknown':p._known_sections={'A'}
            if change=='require_retrieved':p.learned_section_require_retrieved=True
            state=p._new_state({'messages':[{'role':'user','content':'alpha'}]},change=='guard')
            await p._search(state)
            p.llm.embed.assert_awaited_once()

    async def test_concurrent_guard_retains_real_client_and_no_global_mutation(self):
        p=setup(Pipeline);original=p.llm
        states=[p._new_state({'messages':[{'role':'user','content':'alpha'}]},guard) for guard in (False,True)]
        await asyncio.gather(*(p._search(s) for s in states))
        p.llm.embed.assert_awaited_once()
        self.assertIs(p.llm,original);self.assertTrue(p.llm.can_embed)
        self.assertTrue(p.searcher.has_embeddings)
        self.assertTrue(states[0].debug.get('trusted_embedding_skipped'))
        self.assertFalse(states[1].debug.get('trusted_embedding_skipped'))

    async def test_embedding_off_does_not_mark_skip(self):
        p=setup(Pipeline);p.llm.can_embed=False
        state=p._new_state({'messages':[{'role':'user','content':'alpha'}]},False)
        await p._search(state)
        self.assertFalse(state.debug.get('trusted_embedding_skipped'))
        p.llm.embed.assert_not_awaited()
        self.assertIs(Pipeline._classify,Base._classify)
        self.assertIs(Pipeline._generate,Base._generate)
        self.assertIs(Pipeline._check_output,Base._check_output)


if __name__=='__main__': unittest.main()
