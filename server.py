import os
import secrets
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional  # noqa: UP035

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from openai import AsyncOpenAI
from pydantic import BaseModel

from guard_supervisor import GuardSupervisor
from kb_search import KnowledgeSearch

TOKEN = os.environ["TEAM_TOKEN"]
MODEL = os.environ["OPENAI_MODEL"]
SYSTEM = "당신은 온라인 학습 플랫폼의 AI 튜터입니다. 한국어로 답합니다."
FALLBACK = "지금은 답변을 만들 수 없어요. 잠시 후 다시 질문해 주세요."

# Only this logger is enabled; SDK request URLs/bodies are not logged here.
runtime_logger = logging.getLogger("guard_runtime")
runtime_logger.setLevel(logging.INFO)
if not runtime_logger.handlers:
    runtime_logger.addHandler(logging.StreamHandler())
runtime_logger.propagate = False

llm = AsyncOpenAI(timeout=25, max_retries=0)


def check_token(authorization: str = Header("")):
    if not secrets.compare_digest(authorization.encode(), f"Bearer {TOKEN}".encode()):
        raise HTTPException(status_code=401, detail="invalid token")


@asynccontextmanager
async def lifespan(app):
    app.state.searcher = KnowledgeSearch()
    app.state.guard = GuardSupervisor(
        llm, MODEL, app.state.searcher,
        review_mode=os.environ.get("GUARD_REVIEW_MODE", "off"),
        safety_review_enabled=os.environ.get("GUARD_SAFETY_REVIEW", "off") == "on",
    )
    try:
        yield
    finally:
        await app.state.searcher.close()


app = FastAPI(
    redirect_slashes=False,
    dependencies=[Depends(check_token)],
    lifespan=lifespan,
)


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
async def guard(req: GuardRequest, response: Response):
    request_id = uuid.uuid4().hex
    response.headers["X-Guard-Request-ID"] = request_id
    try:
        return await app.state.guard.evaluate(req.messages, req.candidate_answer, request_id=request_id)
    except Exception as error:
        # SPEC §8: never log request contents, model output, or exception bodies.
        print("가드 단계 실패:", type(error).__name__)
        raise HTTPException(status_code=503, detail="Guard evaluation failed",
                            headers={"X-Guard-Request-ID": request_id}) from None
