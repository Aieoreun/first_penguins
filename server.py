import logging
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from app.config import ClientPool, load_settings
from app.guard_classifier import GuardClassifier
from app.guard_verifier import GuardVerifier
from app.kb_index import KBIndex
from app.tutor_pipeline import TutorPipeline

# SPEC.md §8: Log ONLY non-content metadata (timing, lengths, status, error type).
# Never log or store request messages, candidate_answer, or response answer text.
logger = logging.getLogger("first_penguins")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class AppState:
    settings: Any = None
    pool: Optional[ClientPool] = None
    kb_index: Optional[KBIndex] = None
    guard_classifier: Optional[GuardClassifier] = None
    guard_verifier: Optional[GuardVerifier] = None
    tutor_pipeline: Optional[TutorPipeline] = None


state = AppState()


def _init_sync_components() -> None:
    if state.settings is not None:
        return
    settings = load_settings()
    pool = ClientPool(settings)
    kb_index = KBIndex(settings, embed_client=pool.embed_client)
    guard_classifier = GuardClassifier(settings, pool)
    guard_classifier.load_or_train()
    guard_verifier = GuardVerifier(settings, pool, kb_index)
    tutor_pipeline = TutorPipeline(settings, pool, kb_index, guard_classifier=guard_classifier)

    state.settings = settings
    state.pool = pool
    state.kb_index = kb_index
    state.guard_classifier = guard_classifier
    state.guard_verifier = guard_verifier
    state.tutor_pipeline = tutor_pipeline


@asynccontextmanager
async def lifespan(_app: FastAPI):
    t0 = time.perf_counter()
    _init_sync_components()
    if state.kb_index is not None and state.pool is not None and state.pool.embed_client is not None:
        try:
            await state.kb_index.ensure_embeddings()
        except Exception as exc:
            logger.warning("startup_embedding_skipped err=%s", type(exc).__name__)
    logger.info("startup_ready elapsed_ms=%.1f", (time.perf_counter() - t0) * 1000)
    try:
        yield
    finally:
        if state.pool is not None:
            await state.pool.close()


def check_token(authorization: str = Header("")):
    token = state.settings.team_token if state.settings is not None else load_settings().team_token
    if not secrets.compare_digest(authorization.encode(), f"Bearer {token}".encode()):
        raise HTTPException(status_code=401, detail="invalid token")


app = FastAPI(
    redirect_slashes=False,
    dependencies=[Depends(check_token)],
    lifespan=lifespan,
)


class TutorRequest(BaseModel):
    spec_version: Optional[str] = "1.0"
    session_id: Optional[str] = None
    messages: List[Dict[str, str]]


class GuardRequest(BaseModel):
    spec_version: Optional[str] = "1.0"
    session_id: Optional[str] = None
    messages: List[Dict[str, str]]
    candidate_answer: Optional[str] = None


@app.post("/tutor")
async def tutor(req: TutorRequest):
    t0 = time.perf_counter()
    _init_sync_components()
    err_name = "none"
    try:
        res = await state.tutor_pipeline.handle(req.messages)
    except Exception as exc:
        err_name = type(exc).__name__
        res = {
            "answer": "지금은 답변을 만들 수 없어요. 잠시 후 다시 질문해 주세요.",
            "action": "answer",
            "citations": [],
        }
    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "endpoint=/tutor status=200 latency_ms=%.1f msg_count=%d answer_len=%d action=%s citations_cnt=%d err=%s",
        latency_ms,
        len(req.messages),
        len(res.get("answer", "")),
        res.get("action", ""),
        len(res.get("citations", [])),
        err_name,
    )
    return res


@app.post("/guard")
async def guard(req: GuardRequest):
    t0 = time.perf_counter()
    _init_sync_components()
    err_name = "none"
    decision = "allow"
    category = "normal"
    try:
        decision, category, _conf = await state.guard_classifier.classify(req.messages)
        result: Dict[str, Any] = {
            "input": {"decision": decision, "category": category}
        }
        if req.candidate_answer is not None:
            result["output"] = await state.guard_verifier.verify(
                req.messages, req.candidate_answer, decision, category
            )
    except Exception as exc:
        err_name = type(exc).__name__
        result = {"input": {"decision": "allow", "category": "normal"}}
        if req.candidate_answer is not None:
            result["output"] = {
                "grounded": True,
                "unsupported": [],
                "safe": True,
                "category": "normal",
            }
    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "endpoint=/guard status=200 latency_ms=%.1f msg_count=%d has_cand=%s decision=%s category=%s err=%s",
        latency_ms,
        len(req.messages),
        req.candidate_answer is not None,
        result["input"]["decision"],
        result["input"]["category"],
        err_name,
    )
    return result
