"""Use Peace's original guards with Seungwon's v2 search and answer generation.

Each variant runs in its own process. No evaluation labels/IDs enter this module.
Peace prompts and validators are kept in peace/ unchanged except relative imports.
"""
import asyncio
from dataclasses import dataclass
from typing import Optional

from core.schema import Decision, Label, parse_label
from core.state import State
from pipelines.v2.pipeline import Pipeline as V2
from pipelines.v1.pipeline import CLASSIFY_TIMEOUT, SECOND_TIMEOUT, FALLBACK_CHUNKS
from .peace.guard_classifier import classify_input
from .peace.guard_output import check_grounding, check_safety
from .peace.guard_schemas import validate_guard_response


@dataclass
class PeaceState(State):
    """Retain Peace's independent decision, rather than silently recomputing it."""
    peace_decision: Optional[Decision] = None

    @property
    def decision(self):
        return self.peace_decision if self.peace_decision is not None else super().decision


class Pipeline(V2):
    peace_input = False
    peace_output = False

    async def _stage(self, state, name, operation, timeout):
        try:
            return await asyncio.wait_for(operation, timeout)
        except Exception as exc:
            # Never log exception messages: providers may include request content.
            state.errors.append(f"{name}:{type(exc).__name__}")
            return None

    async def _classify(self, state):
        if not self.peace_input:
            return await super()._classify(state)
        verdict = await self._stage(
            state, "peace_input",
            classify_input(self.llm._chat, self.llm.model, state.messages), CLASSIFY_TIMEOUT)
        if verdict is not None:
            state.label = parse_label(verdict.category)
            state.peace_decision = Decision(verdict.decision)
            # Peace has no harmful subcategory; use v2's documented harmful_content
            # alias (harmful_request/refuse). Measure this loss before adding a call.
        else:
            state.label = Label.NORMAL
        state.section = state.chunks[0].section_id if state.chunks else None

    def _evidence(self, state):
        # Same evidence scope as v2 output verification; only the format is adapted.
        if state.section:
            return [{"section_id": state.section, "text": self.searcher.section_text(state.section)}]
        return [{"section_id": c.section_id, "text": c.text}
                for c in state.chunks[:FALLBACK_CHUNKS]]

    async def _check_output(self, state):
        if not self.peace_output:
            return await super()._check_output(state)
        candidate = state.candidate_answer or ""
        # Preserve each successful result when the other independent stage fails.
        # Fallback matches the existing input-based defaults, not an extra API call.
        state.safe = state.decision is Decision.ALLOW
        ground, safe = await asyncio.gather(
            self._stage(state, "peace_grounding", check_grounding(
                self.llm._chat, self.llm.model, state.messages, candidate,
                self._evidence(state)), SECOND_TIMEOUT),
            self._stage(state, "peace_safety", check_safety(
                self.llm._chat, self.llm.model, state.messages, candidate), SECOND_TIMEOUT),
        )
        if ground is not None:
            state.grounded, state.unsupported = ground.grounded, ground.unsupported
        if safe is not None:
            state.safe = safe.safe
        validate_guard_response(state.to_guard_response(), candidate)

    async def tutor(self, body):
        state = PeaceState.from_request(body)
        await self.prepare(state)
        if state.needs_generation:
            await self._generate(state)
        return state

    async def guard(self, body):
        state = PeaceState.from_request(body)
        await self.prepare(state, extra=state.candidate_answer or "")
        if state.has_candidate:
            await self._check_output(state)
        return state


class PeaceInput(Pipeline):
    peace_input = True


class PeaceOutput(Pipeline):
    peace_output = True


class PeaceBoth(Pipeline):
    peace_input = True
    peace_output = True
