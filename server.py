import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional  # noqa: UP035

from fastapi import Depends, FastAPI, Header, HTTPException, Response, Request
from openai import AsyncOpenAI
from pydantic import BaseModel

from guard_supervisor import GuardSupervisor
from kb_search import KnowledgeSearch

TOKEN = os.environ["TEAM_TOKEN"]
MODEL = os.environ["OPENAI_MODEL"]
SYSTEM = "당신은 온라인 학습 플랫폼의 AI 튜터입니다. 한국어로 답합니다."
FALLBACK = "지금은 답변을 만들 수 없어요. 잠시 후 다시 질문해 주세요."

from server_logging import configure_server_log, server_event

configure_server_log()

llm = AsyncOpenAI(timeout=25, max_retries=0)


def check_token(authorization: str = Header("")):
    if not secrets.compare_digest(authorization.encode(), f"Bearer {TOKEN}".encode()):
        raise HTTPException(status_code=401, detail="invalid token")


@asynccontextmanager
async def lifespan(app):
    server_event("server_start", review_mode=os.environ.get("GUARD_REVIEW_MODE", "off"),
                 safety_review=os.environ.get("GUARD_SAFETY_REVIEW", "off"),
                 privacy_enabled=os.environ.get("GUARD_PRIVACY_CATEGORY", "on"),
                 privacy_wait_cap=os.environ.get("GUARD_PRIVACY_WAIT_CAP", "8"))
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
        server_event("server_stop")


app = FastAPI(
    redirect_slashes=False,
    dependencies=[Depends(check_token)],
    lifespan=lifespan,
)


@app.middleware("http")
async def record_http(request, call_next):
    started = time.monotonic()
    request_id = uuid.uuid4().hex
    request.state.telemetry_id = request_id
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        endpoint = request.url.path if request.url.path in {"/guard", "/tutor"} else "other"
        server_event("http_complete", endpoint=endpoint, status=status,
                     seconds=round(time.monotonic() - started, 6), request_id=request_id)


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
        server_event("tutor_model_failure", error_type=type(e).__name__)
        answer = FALLBACK
    return {"answer": answer, "action": "answer", "citations": []}


@app.post("/guard")
async def guard(req: GuardRequest, response: Response, request: Request):
    request_id = request.state.telemetry_id
    response.headers["X-Guard-Request-ID"] = request_id
    try:
        return await app.state.guard.evaluate(req.messages, req.candidate_answer, request_id=request_id)
    except Exception as error:
        # SPEC §8: never log request contents, model output, or exception bodies.
        print("가드 단계 실패:", type(error).__name__)
        raise HTTPException(status_code=503, detail="Guard evaluation failed",
                            headers={"X-Guard-Request-ID": request_id}) from None
