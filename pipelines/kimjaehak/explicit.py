"""Require an explicit answer-only request, as the input policy specifies."""
import re

from core.schema import Label
from .adaptive import Pipeline as Adaptive
from .calibrated import Pipeline as Calibrated


def answer_only_requested(text):
    compact = re.sub(r"\s+", "", text).lower()
    return bool(re.search(
        r"정답만|답만|(?:설명|해설|풀이)(?:은|는|을|를)?(?:없이|빼고|생략)|"
        r"(?:only|just)(?:the)?(?:answer|result|output)|(?:answer|result|output)only|"
        r"(?:without|no)(?:any)?(?:explanation|reasoning|working)", compact))


class Pipeline(Calibrated):
    async def _ask(self, state, name, system, user, effort, timeout):
        obj = await super()._ask(state, name, system, user, effort, timeout)
        if name == "other" and obj.get("category") == "answer_seeking" and not answer_only_requested(state.last_user):
            obj = {**obj, "category": "none"}
        return obj

    async def _classify(self, state):
        # Meaningful two-letter responses such as ㅇㅇ and ㄱㄱ remain contextual.
        compact = re.sub(r"\s+", "", state.last_user)
        if re.fullmatch(r"[ㅋㅎㅠㅜ]{2,}|[ㄱ-ㅎㅏ-ㅣ]{3,}", compact):
            state.label = Label.LOW_SIGNAL_INPUT
            state.section = state.chunks[0].section_id if state.chunks else None
            return
        await super()._classify(state)


class AdaptivePipeline(Pipeline, Adaptive):
    """Apply explicit intent boundaries before adaptive vote aggregation."""
