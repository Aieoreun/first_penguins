"""v2 흐름. 입력 가드를 셋으로 나눠 동시에 부른다. 리라이터는 켜고 끌 수 있다.

  (리라이터) → 검색 → [차단 확인 | 그 밖의 확인 | 교재 확인] 동시에 → 코드가 우선순위대로 합침
  → 답변 생성 또는 출력 가드 (v1과 같다)

차단 확인과 그 밖의 확인은 자기 몫의 범주 중 하나(또는 none)를 고르고, 교재 확인은 답이 되는 문장을 베껴 온다. 범주는 코드가 docs/sw/02_설계.md 2.4절의 우선순위로 정한다.
어느 확인이 실패하면 그 확인의 칸은 모두 해당 없음으로 본다(교재 확인은 "교재에 있음").
"""
import asyncio
import os
from typing import Optional

from core.schema import Label
from core.state import Chunk, State
from pipelines.v1.pipeline import CLASSIFY_EFFORT, CLASSIFY_TIMEOUT, EMBED_TIMEOUT, QUERY_MAX_CHARS, _json_object
from pipelines.v1.pipeline import Pipeline as V1Pipeline

from . import prompts

REWRITE = os.environ.get("V2_REWRITE", "0") == "1"        # 리라이터를 쓸지
REWRITE_EFFORT = os.environ.get("V2_REWRITE_EFFORT", "minimal")
REWRITE_TIMEOUT = float(os.environ.get("V2_REWRITE_TIMEOUT", "6"))
SECTION_SOURCE = os.environ.get("V2_SECTION", "top1")     # 근거 섹션: top1(낱말 검색 1위) 또는 model(교재 확인이 고른 것)

# 겹칠 때의 우선순위 (02_설계.md 2.4절). 앞에 있는 것이 이긴다
_PRIORITY = [
    ("harmful_request", Label.HARMFUL_REQUEST),
    ("harmful_speech", Label.HARMFUL_SPEECH),
    ("jailbreak_injection", Label.JAILBREAK_INJECTION),
    ("emotional_distress", Label.EMOTIONAL_DISTRESS),
    ("academic_dishonesty", Label.ACADEMIC_DISHONESTY),
    ("answer_seeking", Label.ANSWER_SEEKING),
    ("off_topic", Label.OFF_TOPIC),
    ("low_signal_input", Label.LOW_SIGNAL_INPUT),
    ("out_of_material", Label.OUT_OF_MATERIAL),
    ("self_info", Label.PII_DISCLOSURE),
]


class Pipeline(V1Pipeline):
    async def _ask(self, state: State, name: str, system: str, user: str, effort: str, timeout: float) -> dict:
        try:
            text = await self.llm.chat(system, user, effort=effort, max_tokens=1500, timeout=timeout, json_mode=True)
            obj = _json_object(text)
            if obj is None:
                state.errors.append(f"{name}:parse")
            return obj or {}
        except Exception as e:
            state.errors.append(f"{name}:{type(e).__name__}")
            return {}

    async def _rewrite(self, state: State) -> None:
        obj = await self._ask(state, "rewrite", prompts.REWRITE_SYSTEM,
                              prompts.dialogue(state.history, state.last_user), REWRITE_EFFORT, REWRITE_TIMEOUT)
        state.request = obj.get("request") if isinstance(obj.get("request"), str) else ""
        state.query = obj.get("query") if isinstance(obj.get("query"), str) else ""

    async def _search(self, state: State, extra: str = "") -> None:
        # 검색어는 대화 전체다. 리라이터의 검색어는 뒤에 덧붙이기만 한다(experiments/eval_rewrite.py 결과)
        await super()._search(state, extra=(state.query + "\n" + extra).strip())

    async def _classify(self, state: State) -> None:
        h, last, req = state.history, state.last_user, state.request
        block, other, material = await asyncio.gather(
            self._ask(state, "block", prompts.BLOCK_SYSTEM, prompts.check_user(req, h, last),
                      CLASSIFY_EFFORT, CLASSIFY_TIMEOUT),
            self._ask(state, "other", prompts.OTHER_SYSTEM, prompts.check_user(req, h, last),
                      CLASSIFY_EFFORT, CLASSIFY_TIMEOUT),
            self._ask(state, "material", prompts.MATERIAL_SYSTEM, prompts.material_user(state.chunks, req, h, last),
                      CLASSIFY_EFFORT, CLASSIFY_TIMEOUT),
        )
        flags = {str(block.get("category")): True, str(other.get("category")): True}
        flags["self_info"] = other.get("self_info") is True
        # 교재 확인이 답이 되는 문장을 베껴 오지 못했으면 교재에 없는 것으로 본다. 호출이 실패했으면 있는 것으로 본다
        quote = material.get("quote")
        flags["out_of_material"] = isinstance(quote, str) and not quote.strip()
        state.label = next((label for key, label in _PRIORITY if flags.get(key)), Label.NORMAL)

        candidates = {c.section_id for c in state.chunks}
        picked = material.get("section")
        picked = picked.strip() if isinstance(picked, str) and picked.strip() in candidates else None
        top1 = state.chunks[0].section_id if state.chunks else None
        state.section = (picked or top1) if SECTION_SOURCE == "model" else (top1 or picked)
        state.debug = {
            "flags": {k: v for k, v in flags.items() if v}, "model_section": picked,
            # 확신 정도. 지금은 판정에 쓰지 않고 실험에서 들여다보기만 한다
            "block": [block.get("category"), block.get("confidence")],
            "other": [other.get("category"), other.get("confidence")],
            "self_info": [other.get("self_info"), other.get("self_info_confidence")],
            "material": [flags["out_of_material"], material.get("confidence")],
        }

    async def prepare(self, state: State, extra: str = "") -> None:
        if REWRITE:
            await self._rewrite(state)
        await self._search(state, extra)
        await self._classify(state)
