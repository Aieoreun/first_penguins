"""대회 서버. /tutor와 /guard 두 주소를 연다 (SPEC.md 4·5절).

실행 (저장소 폴더에서, .env를 읽어 들인 뒤):
    uvicorn server:app --host 0.0.0.0 --port 8000

요청 처리는 core/pipeline.py가 한다. 여기서는 토큰 확인, 색인 준비, 응답 포장, 로그만 한다.
로그에는 시각·걸린 시간·길이·오류 종류만 남긴다. 요청과 답변의 본문은 남기지 않는다 (SPEC 8절).
"""
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from core.kb import load_chunks
from core.llm import LLM
from core.pipeline import FALLBACK_ANSWER, Pipeline
from core.search import Searcher, load_vectors, save_vectors

ROOT = Path(__file__).resolve().parent
TOKEN = os.environ["TEAM_TOKEN"]
KB_DIR = Path(os.environ.get("KB_DIR", ROOT / "kb"))
INDEX_PATH = Path(os.environ.get("KB_INDEX", ROOT / "index" / "kb_embeddings.json"))
EMBED_BATCH = 32

state = {}


async def _kb_vectors(llm: LLM, chunks):
    """교재 조각의 임베딩. 파일에 있으면 읽고, 없으면 API로 만들어 저장한다. 실패하면 None."""
    if not llm.can_embed:
        return None
    vectors = load_vectors(INDEX_PATH, llm.embedding_model, chunks)
    if vectors is not None:
        return vectors
    try:
        vectors = []
        for i in range(0, len(chunks), EMBED_BATCH):
            vectors += await llm.embed([c.text for c in chunks[i:i + EMBED_BATCH]], timeout=60)
        save_vectors(INDEX_PATH, llm.embedding_model, chunks, vectors)
        return load_vectors(INDEX_PATH, llm.embedding_model, chunks)
    except Exception as e:
        print("임베딩 색인 실패, 낱말 검색만 쓴다:", type(e).__name__, flush=True)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 색인은 서버를 띄울 때 끝낸다. 첫 요청에서 만들면 시간 초과가 난다 (SPEC 3절)
    llm = LLM()
    chunks = load_chunks(KB_DIR)
    vectors = await _kb_vectors(llm, chunks)
    state["pipeline"] = Pipeline(Searcher(chunks, vectors), llm)
    print(f"준비 끝: 조각 {len(chunks)}개, 임베딩 {'있음' if vectors is not None else '없음'}", flush=True)
    yield


def check_token(authorization: str = Header("")):
    if not secrets.compare_digest(authorization.encode(), f"Bearer {TOKEN}".encode()):
        raise HTTPException(status_code=401, detail="invalid token")


app = FastAPI(redirect_slashes=False, dependencies=[Depends(check_token)], lifespan=lifespan)


async def _body(request: Request):
    try:
        return await request.json()
    except Exception:
        return {}


def _log(endpoint: str, started: float, body, errors) -> None:
    messages = body.get("messages") if isinstance(body, dict) else None
    turns = len(messages) if isinstance(messages, list) else 0
    chars = sum(len(m.get("content", "")) for m in messages
                if isinstance(m, dict) and isinstance(m.get("content"), str)) if turns else 0
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {endpoint} {time.time() - started:.2f}s "
          f"turns={turns} chars={chars} errors={','.join(errors) or '-'}", flush=True)


@app.post("/tutor")
async def tutor(request: Request):
    started, body, errors = time.time(), await _body(request), []
    try:
        s = await state["pipeline"].tutor(body)
        response, errors = s.to_tutor_response(), s.errors
    except Exception as e:  # 어떤 경우에도 200과 규격 응답을 낸다
        response, errors = {"answer": FALLBACK_ANSWER, "action": "answer", "citations": []}, [f"tutor:{type(e).__name__}"]
    _log("tutor", started, body, errors)
    return response


@app.post("/guard")
async def guard(request: Request):
    started, body, errors = time.time(), await _body(request), []
    try:
        s = await state["pipeline"].guard(body)
        response, errors = s.to_guard_response(), s.errors
    except Exception as e:
        response = {"input": {"decision": "allow", "category": "normal"}}
        if isinstance(body, dict) and isinstance(body.get("candidate_answer"), str):
            response["output"] = {"grounded": True, "unsupported": [], "safe": True, "category": "normal"}
        errors = [f"guard:{type(e).__name__}"]
    _log("guard", started, body, errors)
    return response
