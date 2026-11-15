"""Choose the most consistent of three independent grounding checks."""
import asyncio
from dataclasses import replace

from core.schema import Decision
from .candidate import Pipeline as Candidate
from .context_output import Pipeline as ContextOutput


def covered_positions(candidate, phrases):
    positions = set()
    for phrase in phrases:
        start = 0
        while phrase and (at := candidate.find(phrase, start)) >= 0:
            positions.update(i for i in range(at, at + len(phrase)) if not candidate[i].isspace())
            start = at + len(phrase)
    return positions


def consensus_index(candidate, votes):
    """Select an actual model verdict; never splice together new error text."""
    masks = [covered_positions(candidate, phrases) for phrases in votes]
    def agreement(a, b):
        return 2 * len(a & b) / (len(a) + len(b)) if a or b else 1.0
    return max(range(len(masks)), key=lambda i: sum(
        agreement(masks[i], other) for other in masks))


class Pipeline(Candidate):
    output_votes = 3

    async def _check_output(self, state):
        if state.decision is Decision.BLOCK:
            return await super()._check_output(state)
        # Each call has its own mutable output and error state. The parent
        # request deadline still bounds all three concurrent calls together.
        trials = [replace(state, unsupported=[], errors=[], debug=dict(state.debug))
                  for _ in range(self.output_votes)]
        await asyncio.gather(*(ContextOutput._check_output(self, trial) for trial in trials))
        good = [trial for trial in trials if not trial.errors]
        for trial in trials:
            state.errors.extend(trial.errors)
        state.safe = True
        state.unsupported = (good[consensus_index(state.candidate_answer or "",
                             [trial.unsupported for trial in good])].unsupported if good else [])
        state.grounded = not state.unsupported
