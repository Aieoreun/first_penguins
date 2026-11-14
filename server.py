import os
import secrets
from typing import Dict, List, Optional  # noqa: UP035

from fastapi import Depends, FastAPI, Header, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel

from guard_classifier import classify_input
from guard_schemas import validate_guard_response

TOKEN = os.environ["TEAM_TOKEN"]
MODEL = os.environ["OPENAI_MODEL"]
SYSTEM = "당신은 온라인 학습 플랫폼의 AI 튜터입니다. 한국어로 답합니다."
FALLBACK = "지금은 답변을 만들 수 없어요. 잠시 후 다시 질문해 주세요."

llm = AsyncOpenAI(timeout=25, max_retries=0)


def check_token(authorization: str = Header("")):
    if not secrets.compare_digest(authorization.encode(), f"Bearer {TOKEN}".encode()):
        raise HTTPException(status_code=401, detail="invalid token")


app = FastAPI(redirect_slashes=False, dependencies=[Depends(check_token)])


class TutorRequest(BaseModel):
    messages: List[Dict[str, str]]


class GuardRequest(BaseModel):
    messages: List[Dict[str, str]]
    candidate_answer: Optional[str] = None


@app.post("/tutor")
async def tutor(req: TutorRequest):
    # TODO: 입력을 거르고, 지식 베이스에서 근거 섹션을 찾아 프롬프트에 넣고, action과 citations를 정하세요.
    try:
        r = await llm.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, *req.messages],
            max_completion_tokens=2000,
            extra_body={"reasoning_effort": "low"},
        )
        answer = (r.choices[0].message.content or "").strip()[:4000] or FALLBACK
    except Exception as e:
        print("LLM 호출 실패:", type(e).__name__)
        answer = FALLBACK
    return {"answer": answer, "action": "answer", "citations": []}


@app.post("/guard")
async def guard(req: GuardRequest):
    try:
        verdict = await classify_input(
            client=llm,
            model=MODEL,
            messages=req.messages,
        )
        input_result = verdict.model_dump()

    except Exception as error:
        # 대화 내용이나 모델 응답 본문은 기록하지 않습니다.
        print("입력 판정 실패:", type(error).__name__)

        # 첫 버전의 임시 실패 정책입니다.
        # 정상적인 모델 판정으로 간주하면 안 됩니다.
        input_result = {
            "decision": "allow",
            "category": "normal",
        }

    result = {"input": input_result}

    # 후보 답변 검증은 아직 구현하지 않았으므로 기존 값을 유지합니다.
    if req.candidate_answer is not None:
        result["output"] = {
            "grounded": True,
            "unsupported": [],
            "safe": True,
            "category": "normal",
        }

    validated = validate_guard_response(
        result,
        candidate_answer=req.candidate_answer,
    )
    return validated.model_dump(exclude_none=True)
