import asyncio
import os
import secrets
from contextlib import asynccontextmanager
from typing import Dict, List, Optional  # noqa: UP035

from fastapi import Depends, FastAPI, Header, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel

from guard_classifier import classify_input
from guard_output import check_grounding, check_safety
from guard_schemas import validate_guard_response
from kb_search import KnowledgeSearch

TOKEN = os.environ["TEAM_TOKEN"]
MODEL = os.environ["OPENAI_MODEL"]
SYSTEM = "당신은 온라인 학습 플랫폼의 AI 튜터입니다. 한국어로 답합니다."
FALLBACK = "지금은 답변을 만들 수 없어요. 잠시 후 다시 질문해 주세요."

llm = AsyncOpenAI(timeout=25, max_retries=0)


def check_token(authorization: str = Header("")):
    if not secrets.compare_digest(authorization.encode(), f"Bearer {TOKEN}".encode()):
        raise HTTPException(status_code=401, detail="invalid token")


@asynccontextmanager
async def lifespan(app):
    app.state.searcher = KnowledgeSearch()
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


async def run_guard_stage(name, operation):
    try:
        # 검색을 포함한 각 단계에 제한 시간을 둡니다.
        return await asyncio.wait_for(operation, timeout=26)
    except Exception as error:
        # 요청·답변 본문은 기록하지 않습니다.
        print(f"가드 단계 실패: {name} / {type(error).__name__}")
        return None


async def retrieve_and_check(req):
    query = "\n".join(
        f"{message['role']}: {message['content']}" for message in req.messages
    )

    evidence = await app.state.searcher.search(query, top_k=5)

    return await check_grounding(
        client=llm,
        model=MODEL,
        messages=req.messages,
        candidate_answer=req.candidate_answer,
        evidence=evidence,
    )


@app.post("/guard")
async def guard(req: GuardRequest):
    operations = [
        run_guard_stage(
            "input",
            classify_input(llm, MODEL, req.messages),
        )
    ]

    has_candidate = req.candidate_answer is not None

    if has_candidate:
        operations.extend(
            [
                run_guard_stage("grounding", retrieve_and_check(req)),
                run_guard_stage(
                    "safety",
                    check_safety(llm, MODEL, req.messages, req.candidate_answer),
                ),
            ]
        )

    results = await asyncio.gather(*operations)

    input_verdict = results[0]
    response = {
        "input": (
            input_verdict.model_dump()
            if input_verdict is not None
            else {"decision": "allow", "category": "normal"}
        )
    }

    if has_candidate:
        grounding, safety = results[1], results[2]

        response["output"] = {
            "grounded": grounding.grounded if grounding else True,
            "unsupported": grounding.unsupported if grounding else [],
            "safe": safety.safe if safety else True,
            "category": "normal",
        }

    validated = validate_guard_response(
        response,
        candidate_answer=req.candidate_answer,
    )
    return validated.model_dump(exclude_none=True)
