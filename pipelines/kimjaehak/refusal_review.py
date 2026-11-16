"""Review explicit refusals instead of equating blocked input with unsafe output."""
import asyncio
import json
import re

from core.schema import Decision
from pipelines.v1.pipeline import _json_object
from .hybrid_output import Pipeline as Hybrid
from .learned_output import Consensus
from .precise_output import POLICY_SYSTEM, POLICY_HINTS


REFUSAL = re.compile(
    r"(?:도울|도와드릴|제공할|알려드릴|공개할|작성해\s*드릴)\s*수\s*없"
    r"|동의하지\s*않|거절(?:할게|합니다|해요)"
)


class RefusalReview:
    refusal_review = True
    refusal_timeout = 8
    refusal_pattern = REFUSAL
    policy_system = POLICY_SYSTEM

    async def _check_output(self, state):
        await super()._check_output(state)
        if state.decision is not Decision.BLOCK or not self.refusal_pattern.search(state.candidate_answer or ""):
            return
        # A refusal word only triggers a semantic check; it never implies safe.
        # In particular, a disclaimer followed by compliance remains unsafe.
        try:
            raw = await asyncio.wait_for(self.llm.chat(
                self.policy_system + "\n" + POLICY_HINTS.get(state.category.value, ""),
                json.dumps({"conversation": state.messages,
                            "candidate_answer": state.candidate_answer}, ensure_ascii=False),
                effort="low", max_tokens=1800, timeout=self.refusal_timeout, json_mode=True),
                self.refusal_timeout)
            safe = (_json_object(raw) or {}).get("safe")
            if not isinstance(safe, bool):
                raise ValueError("Invalid safety verdict")
            state.safe = safe
            state.debug["refusal_reviewed"] = True
        except Exception as exc:
            state.errors.append(f"refusal_review:{type(exc).__name__}")


class Pipeline(RefusalReview, Consensus):
    pass


class HybridPipeline(RefusalReview, Hybrid):
    pass


class BroadReview(RefusalReview):
    # These are inspection triggers, not safety verdicts. Include inflected
    # refusals and redirection; semantic checking still rejects compliance.
    refusal_pattern = re.compile(
        r"수(?:는|도)?\s*없|못(?:해|하)|삼가|대신"
        r"|(?:좋지|옳지|적절하지|동의하지)\s*않"
    )


class BroadPipeline(BroadReview, Hybrid):
    pass
