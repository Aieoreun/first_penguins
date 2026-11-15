"""Skip a third vote exactly when it cannot change the configured outcome."""
import asyncio
import time

from pipelines.v1.pipeline import CLASSIFY_EFFORT, CLASSIFY_TIMEOUT
from pipelines.v2 import prompts
from .voted import BLOCKS, OTHERS, Pipeline as Voted


def needs_third(answers, *, other=False, unanimous=False):
    allowed = OTHERS if other else BLOCKS
    categories = [obj["category"] for obj in answers
                  if isinstance(obj.get("category"), str) and obj["category"] in allowed]
    if unanimous and not other:
        return len(categories) == 2 and categories[0] == categories[1] != "none"
    category_open = not categories or (len(categories) == 2 and categories[0] != categories[1])
    if not other:
        return category_open
    personal = [obj["self_info"] for obj in answers if isinstance(obj.get("self_info"), bool)]
    personal_open = not personal or (len(personal) == 2 and personal[0] != personal[1])
    return category_open or personal_open


class Pipeline(Voted):
    adaptive_votes = True

    async def _tie_vote(self, state, name, system, check, remaining):
        return await self._ask(state, name, system, check, CLASSIFY_EFFORT, remaining)

    async def _vote_group(self, state, name, system, check):
        started = time.monotonic()
        pair = await asyncio.gather(*(self._ask(state, name, system, check,
            CLASSIFY_EFFORT, CLASSIFY_TIMEOUT) for _ in range(2)))
        if needs_third(pair, other=name == "other", unanimous=self.block_min_votes == 3):
            left = CLASSIFY_TIMEOUT - (time.monotonic() - started)
            if left > .05:
                pair.append(await self._tie_vote(state, name, system, check, left))
            else:
                state.errors.append(f"{name}:tie_budget")
        return pair

    async def _collect_checks(self, state, check, material):
        # Tie resolution begins as soon as its short pair returns, independently
        # of the heavier material check. Each group retains the original budget.
        return await asyncio.gather(
            self._vote_group(state, "block", prompts.BLOCK_SYSTEM, check),
            self._vote_group(state, "other", prompts.OTHER_SYSTEM, check),
            self._ask(state, "material", prompts.MATERIAL_SYSTEM, material, CLASSIFY_EFFORT, CLASSIFY_TIMEOUT))


class Unanimous(Pipeline):
    block_min_votes = 3
