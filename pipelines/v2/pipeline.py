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

# 아래는 실험용 스위치다. 기본값은 실험 전의 동작이다 (docs/sw/04_실험기록.md)
VOTES = int(os.environ.get("V2_VOTES", "1"))                    # 각 확인을 몇 번 불러 다수결할지
SEEK_FIRST = os.environ.get("V2_SEEK_FIRST", "0") == "1"        # 정답만 요구를 과제 대행보다 앞에 둘지
MATERIAL_EFFORT = os.environ.get("V2_MATERIAL_EFFORT", "")      # 교재 확인의 추론 수준(비우면 분류와 같다)
MATERIAL_INPUT = os.environ.get("V2_MATERIAL_INPUT", "chunks")  # 교재 확인에 주는 것: chunks(검색 조각) 또는 sections(상위 섹션 전체)
MATERIAL_SECTIONS = int(os.environ.get("V2_MATERIAL_SECTIONS", "3"))
PROMPTS = os.environ.get("V2_PROMPTS", "1")                     # 답변·구절 찾기 프롬프트: 1(v1 것) 또는 2(둘째 판)
SAFE_RULE = os.environ.get("V2_SAFE_RULE", "0") == "1"          # 차단된 입력의 후보 답변을 규칙으로 안전하지 않다고 볼지
ANSWER_EFFORT_V2 = os.environ.get("V2_ANSWER_EFFORT", "")       # 답변 생성 추론 수준(비우면 v1과 같다)
ANSWER_MAX_TOKENS = int(os.environ.get("V2_ANSWER_MAX_TOKENS", "2000"))

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


def _majority(values):
    """가장 많이 나온 값. 동률이면 먼저 나온 값."""
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(values, key=lambda v: counts[v]) if values else None


def _priority(seek_first: bool):
    if not seek_first:
        return _PRIORITY
    order = [p for p in _PRIORITY if p[0] != "answer_seeking"]
    at = next(i for i, p in enumerate(order) if p[0] == "academic_dishonesty")
    order.insert(at, ("answer_seeking", Label.ANSWER_SEEKING))
    return order


class Pipeline(V1Pipeline):
    if PROMPTS == "2":
        answer_systems = prompts.ANSWER_SYSTEMS
        unsupported_system = prompts.UNSUPPORTED_SYSTEM
    if ANSWER_EFFORT_V2:
        answer_effort = ANSWER_EFFORT_V2
    answer_max_tokens = ANSWER_MAX_TOKENS
    safe_by_rule = SAFE_RULE
    votes = VOTES            # 각 확인을 몇 번 불러 다수결할지
    seek_first = SEEK_FIRST  # 정답만 요구를 과제 대행보다 앞에 둘지
    material_timeout = CLASSIFY_TIMEOUT  # 교재 확인 호출의 제한 시간(초). 세 확인 중 가장 느리다
    block_prompt = prompts.BLOCK_SYSTEM_LEGACY  # 차단 확인 프롬프트. v11은 자기 것(prompts.BLOCK_SYSTEM에 덧붙인 것)으로 바꿔 부른다

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

    def _material_chunks(self, state: State):
        if MATERIAL_INPUT != "sections":
            return state.chunks
        ids = list(dict.fromkeys(c.section_id for c in state.chunks))[:MATERIAL_SECTIONS]
        return [Chunk(sid, self.searcher.section_text(sid)) for sid in ids]

    async def _classify(self, state: State) -> None:
        h, last, req = state.history, state.last_user, state.request
        check = prompts.check_user(req, h, last)
        material_user = prompts.material_user(self._material_chunks(state), req, h, last)
        calls = []
        for _ in range(max(1, self.votes)):
            calls += [
                self._ask(state, "block", self.block_prompt, check, CLASSIFY_EFFORT, CLASSIFY_TIMEOUT),
                self._ask(state, "other", prompts.OTHER_SYSTEM, check, CLASSIFY_EFFORT, CLASSIFY_TIMEOUT),
                self._ask(state, "material", prompts.MATERIAL_SYSTEM, material_user,
                          MATERIAL_EFFORT or CLASSIFY_EFFORT, self.material_timeout),
            ]
        answers = await asyncio.gather(*calls)
        blocks, others, materials = answers[0::3], answers[1::3], answers[2::3]
        block, other, material = blocks[0], others[0], materials[0]

        def missing(m):  # 답이 되는 문장을 못 베껴 왔는가. 호출이 실패했으면 교재에 있는 것으로 본다
            quote = m.get("quote")
            return isinstance(quote, str) and not quote.strip()

        flags = {
            str(_majority([str(b.get("category")) for b in blocks])): True,
            str(_majority([str(o.get("category")) for o in others])): True,
        }
        flags["self_info"] = _majority([o.get("self_info") is True for o in others])
        flags["out_of_material"] = _majority([missing(m) for m in materials])
        state.label = next((label for key, label in _priority(self.seek_first) if flags.get(key)), Label.NORMAL)

        candidates = {c.section_id for c in state.chunks}
        picked = material.get("section")
        picked = picked.strip() if isinstance(picked, str) and picked.strip() in candidates else None
        top1 = state.chunks[0].section_id if state.chunks else None
        state.section = (picked or top1) if SECTION_SOURCE == "model" else (top1 or picked)
        state.debug = {
            "flags": {k: v for k, v in flags.items() if v}, "model_section": picked,
            # 확신 정도. 지금은 판정에 쓰지 않고 실험에서 들여다보기만 한다 (첫 번째 호출의 값)
            "block": [block.get("category"), block.get("confidence")],
            "other": [other.get("category"), other.get("confidence")],
            "self_info": [other.get("self_info"), other.get("self_info_confidence")],
            "material": [missing(material), material.get("confidence")],
        }

    async def prepare(self, state: State, extra: str = "") -> None:
        if REWRITE:
            await self._rewrite(state)
        await self._search(state, extra)
        await self._classify(state)
