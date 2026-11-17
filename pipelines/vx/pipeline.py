"""vx 흐름. 실험용이다. v8에 가드 개선 후보를 스위치로 하나씩 켤 수 있게 얹었다 (docs/sw/05_로컬_점수표.md).

스위치를 모두 끄면 v8과 똑같이 돈다. 환경변수로 켠다.

입력 가드
  VX_CURRICULUM=1     차단 확인 프롬프트에 교재 목차를 넣는다
  VX_TWO_STAGE=1      차단 확인을 두 단계로: "차단인가"만 먼저 묻고, 차단이라고 한 것만 범주를 묻는다
  VX_MIN_CONF=85      차단 확인·그 밖 확인이 이 값보다 낮은 확신으로 잡은 판정은 버린다
  VX_INPUT_RECHECK=1  차단 확인·그 밖 확인을 두 번 부르고, 답이 갈릴 때만 세 번째를 불러 다수결한다
  VX_MATERIAL_RETRY=1 교재 확인이 실패하면 한 번 더 부른다
출력 가드
  VX_NUM_HINT=1       후보 답변의 숫자 중 교재에 없는 것을 코드로 찾아 모델에게 알려 준다
  VX_OUT_SECTIONS=2   근거 검사에 주는 교재를 검색 상위 섹션 둘로 늘린다 (기본 1)
  VX_OUT_VOTES=3      근거 검사를 여러 번 불러 다수결한다 (기본 1)
  VX_SAFE_CHECK=1     차단된 입력의 후보 답변이 요청을 들어줬는지 모델에게 묻는다 (기본은 묻지 않고 안전하지 않다고 본다)
"""
import asyncio
import os
import re

from core.schema import Decision
from core.spans import locate_all
from pipelines.v1 import prompts as v1_prompts
from pipelines.v1.pipeline import FALLBACK_CHUNKS, GUARD_EFFORT, SECOND_TIMEOUT, _json_object
from pipelines.v8.pipeline import Pipeline as V8Pipeline
from pipelines.v8.pipeline import trim
from pipelines.vx import prompts


def _on(name: str) -> bool:
    return os.environ.get(name, "") not in ("", "0")


_SECTION_ID = re.compile(r"[A-Z]{2,3}-\d\d-\d\d")
_NUMBER = re.compile(r"\d[\d,.]*(?:\s?[만천백억])?")
_SQUEEZE = re.compile(r"[\s,]")


def numbers_missing(candidate: str, material: str):
    """후보 답변의 숫자 가운데 교재에 같은 글자로 나오지 않는 것. 띄어쓰기와 쉼표는 무시한다."""
    flat = _SQUEEZE.sub("", material)
    found = []
    for raw in _NUMBER.findall(_SECTION_ID.sub("", candidate)):
        number = _SQUEEZE.sub("", raw).rstrip(".")
        if number and number not in flat and number not in found:
            found.append(number)
    return found


class Pipeline(V8Pipeline):
    curriculum = _on("VX_CURRICULUM")
    two_stage = _on("VX_TWO_STAGE")
    min_conf = float(os.environ.get("VX_MIN_CONF", "0") or 0)
    input_recheck = _on("VX_INPUT_RECHECK")
    material_retry = _on("VX_MATERIAL_RETRY")
    num_hint = _on("VX_NUM_HINT")
    out_sections = int(os.environ.get("VX_OUT_SECTIONS", "1") or 1)
    out_votes = int(os.environ.get("VX_OUT_VOTES", "1") or 1)
    safe_by_rule = not _on("VX_SAFE_CHECK")

    # ---------- 입력 가드 ----------

    async def _ask(self, state, name, system, user, effort, timeout):
        base = super()._ask
        if name == "material":
            answer = await base(state, name, system, user, effort, timeout)
            if not answer and self.material_retry:
                answer = await base(state, "material_retry", system, user, effort, timeout)
            return answer
        if name not in ("block", "other"):
            return await base(state, name, system, user, effort, timeout)
        if name == "block" and self.curriculum:
            system = prompts.BLOCK_KB

        async def once():
            if name == "block" and self.two_stage:
                first = await base(state, "gate", prompts.gate_of(system), user, effort, timeout)
                if first.get("block") is not True:
                    return {"category": "none", "confidence": first.get("confidence")}
            return await base(state, name, system, user, effort, timeout)

        if self.input_recheck:
            first, second = await asyncio.gather(once(), once())
            answer = first
            if first.get("category") != second.get("category"):
                third = await once()
                answer = second if second.get("category") == third.get("category") else (
                    first if first.get("category") == third.get("category") else third)
        else:
            answer = await once()

        category, confidence = answer.get("category"), answer.get("confidence")
        if self.min_conf and category not in (None, "none") and isinstance(confidence, (int, float)) and confidence < self.min_conf:
            answer = dict(answer, category="none")
        return answer

    # ---------- 출력 가드 ----------

    def _output_material(self, state) -> str:
        ids = [state.section] if state.section else []
        ids += [c.section_id for c in state.chunks]
        ids = list(dict.fromkeys(ids))[:self.out_sections]
        if not ids:
            return "\n\n".join(c.text for c in state.chunks[:FALLBACK_CHUNKS])
        return "\n\n".join(self.searcher.section_text(i) for i in ids)

    async def _check_output(self, state) -> None:
        plain = self.out_sections == 1 and self.out_votes == 1 and not self.num_hint
        if state.decision is Decision.BLOCK or plain:
            await super()._check_output(state)  # v8과 같다 (구간 다듬기 포함)
            return
        candidate = state.candidate_answer or ""
        state.safe = True
        material = self._output_material(state)
        user = v1_prompts.unsupported_user(material, candidate)
        if self.num_hint:
            missing = numbers_missing(candidate, material)
            if missing:
                user += prompts.NUMBER_HINT.format(numbers=", ".join(missing))

        async def once():
            try:
                text = await self.llm.chat(self.unsupported_system, user, effort=GUARD_EFFORT, max_tokens=1500,
                                           timeout=SECOND_TIMEOUT, json_mode=True)
                phrases = (_json_object(text) or {}).get("unsupported")
                if not isinstance(phrases, list):
                    state.errors.append("unsupported:parse")
                    return None
                return locate_all(candidate, phrases)
            except Exception as e:
                state.errors.append(f"unsupported:{type(e).__name__}")
                return None

        answers = [a for a in await asyncio.gather(*(once() for _ in range(max(1, self.out_votes)))) if a is not None]
        flagged = [a for a in answers if a]
        # 과반이 근거 없는 구절을 냈을 때만 잡는다. 그중 가장 짧게 잡은 답을 쓴다. 모두 실패하면 근거 있음 (결정 15)
        spans = min(flagged, key=lambda a: sum(len(p) for p in a)) if len(flagged) * 2 > len(answers) else []
        state.debug = dict(state.debug or {}, raw_unsupported=list(spans))
        state.unsupported = [trim(p) for p in spans]
        state.grounded = not state.unsupported
