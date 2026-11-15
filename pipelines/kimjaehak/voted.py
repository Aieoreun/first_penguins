"""Selective three-vote classification with bounded request concurrency.

Vote only on short block/other checks; perform the expensive material check once.
Failed calls are excluded from ballots. The 26-second budget includes queue time.
"""
import asyncio
import time

from core.schema import FALLBACK_ANSWER, Label
from pipelines.v1.pipeline import CLASSIFY_EFFORT, CLASSIFY_TIMEOUT
from pipelines.v2 import prompts
from pipelines.v2.pipeline import _majority, _priority
from .calibrated import Pipeline as Calibrated, ResponseState, low_signal

BLOCKS = {"none", "harmful_request", "harmful_speech", "jailbreak_injection", "academic_dishonesty", "off_topic"}
OTHERS = {"none", "emotional_distress", "answer_seeking", "low_signal_input"}


class Pipeline(Calibrated):
    request_slots = 32
    deadline = 26
    short_votes = 3

    def __init__(self, searcher, llm):
        super().__init__(searcher, llm)
        self._slots = asyncio.Semaphore(self.request_slots)

    async def _classify(self, state):
        if low_signal(state.last_user):
            return await super()._classify(state)
        check = prompts.check_user(state.request, state.history, state.last_user)
        material = prompts.material_user(state.chunks, state.request, state.history, state.last_user)
        calls = []
        for _ in range(self.short_votes):
            calls.extend([
                self._ask(state, "block", prompts.BLOCK_SYSTEM, check, CLASSIFY_EFFORT, CLASSIFY_TIMEOUT),
                self._ask(state, "other", prompts.OTHER_SYSTEM, check, CLASSIFY_EFFORT, CLASSIFY_TIMEOUT),
            ])
        calls.append(self._ask(state, "material", prompts.MATERIAL_SYSTEM, material,
                               CLASSIFY_EFFORT, CLASSIFY_TIMEOUT))
        answers = await asyncio.gather(*calls)
        blocks, others, material = answers[:-1:2], answers[1:-1:2], answers[-1]
        block_votes = [x["category"] for x in blocks
                       if isinstance(x.get("category"), str) and x["category"] in BLOCKS]
        other_votes = [x["category"] for x in others
                       if isinstance(x.get("category"), str) and x["category"] in OTHERS]
        personal = [x["self_info"] for x in others if isinstance(x.get("self_info"), bool)]
        flags = {_majority(block_votes) or "none": True, _majority(other_votes) or "none": True,
                 "self_info": bool(_majority(personal)),
                 "out_of_material": isinstance(material.get("quote"), str) and not material["quote"].strip()}
        state.label = next((label for key, label in _priority(self.seek_first) if flags.get(key)), Label.NORMAL)
        state.section = state.chunks[0].section_id if state.chunks else None
        state.debug = {"valid_block_votes": len(block_votes), "valid_other_votes": len(other_votes)}

    async def _run(self, body, guard):
        started = time.monotonic()
        state = ResponseState.from_request(body)
        async def work():
            async with self._slots:
                await self._prepare_within(state, extra=(state.candidate_answer or "") if guard else "")
                left = self.deadline - (time.monotonic() - started)
                if guard and state.has_candidate:
                    if not await self._within(self._check_output(state), left, state, "check_output"):
                        state.grounded, state.unsupported = True, []
                elif not guard and state.needs_generation:
                    if not await self._within(self._generate(state), left, state, "generate"):
                        state.answer, state.citations = FALLBACK_ANSWER, []
        try:
            await asyncio.wait_for(work(), self.deadline)
        except asyncio.TimeoutError:
            state.errors.append("request:Deadline")
            if not guard:
                state.answer, state.citations = FALLBACK_ANSWER, []
        return state

    async def tutor(self, body):
        return await self._run(body, False)

    async def guard(self, body):
        return await self._run(body, True)
