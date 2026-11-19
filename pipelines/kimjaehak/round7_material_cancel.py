"""Cancel material work only when completed votes make it policy-irrelevant."""
import asyncio

from pipelines.v1.pipeline import CLASSIFY_EFFORT, CLASSIFY_TIMEOUT
from pipelines.v2 import prompts
from pipelines.v2.pipeline import _majority
from .classifier_matrix import CharMLP50
from .voted import BLOCKS, OTHERS


def material_irrelevant(blocks,others,minimum):
    block_votes=[r['category'] for r in blocks
                 if isinstance(r.get('category'),str) and r['category'] in BLOCKS]
    other_votes=[r['category'] for r in others
                 if isinstance(r.get('category'),str) and r['category'] in OTHERS]
    block=_majority(block_votes) or 'none'
    if block!='none' and block_votes.count(block)<minimum:block='none'
    other=_majority(other_votes) or 'none'
    # Every valid non-none BLOCK/OTHER category precedes out_of_material in
    # both existing policy orders. self_info alone does not, so keep material.
    return block!='none' or other!='none'


class CancelGuardMaterial(CharMLP50):
    material_cancel_scope='guard'

    def _cancel_target(self,state):
        guard=getattr(state,'_material_budget_guard',None)
        return (guard is True and self.material_cancel_scope in {'guard','both'}
                or guard is False and self.material_cancel_scope in {'tutor','both'})

    async def _collect_checks(self,state,check,material):
        if not self._cancel_target(state):
            return await super()._collect_checks(state,check,material)
        tasks=[asyncio.create_task(self._vote_group(state,'block',prompts.BLOCK_SYSTEM,check)),
               asyncio.create_task(self._vote_group(state,'other',prompts.OTHER_SYSTEM,check)),
               asyncio.create_task(self._ask(state,'material',prompts.MATERIAL_SYSTEM,material,
                                            CLASSIFY_EFFORT,CLASSIFY_TIMEOUT))]
        try:
            blocks,others=await asyncio.gather(*tasks[:2])
            if not tasks[2].done() and material_irrelevant(blocks,others,self.block_min_votes):
                tasks[2].cancel()
                await asyncio.gather(tasks[2],return_exceptions=True)
                state._r7_material_cancelled=True
                return blocks,others,{}
            return blocks,others,await tasks[2]
        finally:
            for task in tasks:
                if not task.done():task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)

    async def _classify(self,state):
        try:return await super()._classify(state)
        finally:
            if getattr(state,'_r7_material_cancelled',False):state.debug['r7_material_cancelled']=True


class CancelTutorMaterial(CancelGuardMaterial):
    material_cancel_scope='tutor'


class CancelBothMaterial(CancelGuardMaterial):
    material_cancel_scope='both'
