"""Selective three-vote classification with bounded request concurrency.

Vote only on short block/other checks; perform the expensive material check once.
Failed calls are excluded from ballots. The 26-second budget includes queue time.
"""
import asyncio
import time

from core.schema import FALLBACK_ANSWER, Decision, Label
from pipelines.v1.pipeline import CLASSIFY_EFFORT, CLASSIFY_TIMEOUT
from pipelines.v2 import prompts
from pipelines.v2.pipeline import _majority, _priority
from .calibrated import Pipeline as Calibrated, ResponseState, low_signal

BLOCKS = {"none", "harmful_request", "harmful_speech", "jailbreak_injection", "academic_dishonesty", "off_topic"}
OTHERS = {"none", "emotional_distress", "answer_seeking", "low_signal_input"}


class Pipeline(Calibrated):
    state_class = ResponseState
    request_slots = 32
    deadline = 26
    short_votes = 3
    other_vote_count = None  # None preserves the same count as block votes.
    block_min_votes = 1
    material_max_chunks = None

    def __init__(self, searcher, llm):
        super().__init__(searcher, llm)
        self._slots = asyncio.Semaphore(self.request_slots)

    async def _block_check(self, state, check, index):
        return await self._ask(state, "block", prompts.BLOCK_SYSTEM, check, CLASSIFY_EFFORT, CLASSIFY_TIMEOUT)

    async def _collect_checks(self, state, check, material):
        calls, kinds = [], []
        other_count = self.short_votes if self.other_vote_count is None else self.other_vote_count
        for index in range(max(self.short_votes, other_count)):
            if index < self.short_votes:
                calls.append(self._block_check(state, check, index))
                kinds.append("block")
            if index < other_count:
                calls.append(self._ask(state, "other", prompts.OTHER_SYSTEM, check, CLASSIFY_EFFORT, CLASSIFY_TIMEOUT))
                kinds.append("other")
        calls.append(self._ask(state, "material", prompts.MATERIAL_SYSTEM, material,
                               CLASSIFY_EFFORT, CLASSIFY_TIMEOUT))
        answers = await asyncio.gather(*calls)
        blocks = [answer for answer, kind in zip(answers[:-1], kinds) if kind == "block"]
        others = [answer for answer, kind in zip(answers[:-1], kinds) if kind == "other"]
        return blocks, others, answers[-1]

    async def _classify(self, state):
        if low_signal(state.last_user):
            return await super()._classify(state)
        check = prompts.check_user(state.request, state.history, state.last_user)
        material = prompts.material_user(state.chunks[:self.material_max_chunks], state.request, state.history, state.last_user)
        blocks, others, material = await self._collect_checks(state, check, material)
        block_votes = [x["category"] for x in blocks
                       if isinstance(x.get("category"), str) and x["category"] in BLOCKS]
        other_votes = [x["category"] for x in others
                       if isinstance(x.get("category"), str) and x["category"] in OTHERS]
        personal = [x["self_info"] for x in others if isinstance(x.get("self_info"), bool)]
        block = _majority(block_votes) or "none"
        if block != "none" and block_votes.count(block) < self.block_min_votes:
            block = "none"
        flags = {block: True, _majority(other_votes) or "none": True,
                 "self_info": bool(_majority(personal)),
                 "out_of_material": isinstance(material.get("quote"), str) and not material["quote"].strip()}
        state.label = next((label for key, label in _priority(self.seek_first) if flags.get(key)), Label.NORMAL)
        state.section = state.chunks[0].section_id if state.chunks else None
        # Ephemeral diagnostics, never emitted by the HTTP server or its logs.
        state.debug = {"valid_block_votes": len(block_votes), "valid_other_votes": len(other_votes),
                       "block_votes": block_votes, "other_votes": other_votes,
                       "self_info_votes": personal, "out_of_material": flags["out_of_material"],
                       "model_section": material.get("section"),
                       "block_calls": len(blocks), "other_calls": len(others)}

    def _new_state(self, body, guard):
        return self.state_class.from_request(body)

    async def _run(self, body, guard):
        started = time.monotonic()
        state = self._new_state(body, guard)
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
            elif state.has_candidate:
                state.safe = state.decision is Decision.ALLOW
        return state

    async def tutor(self, body):
        return await self._run(body, False)

    async def guard(self, body):
        return await self._run(body, True)
