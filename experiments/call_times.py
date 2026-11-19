"""동시 64개로 공개 400문항을 돌리며 모델 호출 하나하나의 걸린 시간을 잰다 (2026-11-16).

제한 시간은 환경변수로 길게 풀어 두고 돈다(끊기지 않은 분포를 보려고).
둘째 인자로 초를 주면 교재 확인을 먼저 부르고 차단 확인·그 밖 확인은 그만큼 늦게 부른다 (이승원 제안).

실행 (.env를 읽어 들인 뒤):
    CLASSIFY_TIMEOUT=40 SECOND_TIMEOUT=40 V6_PREPARE_LIMIT=80 V6_DEADLINE=160 V6_MATERIAL_TIMEOUT=40 \\
        python experiments/call_times.py <이름> [늦추는 초]
    python experiments/call_times.py report
결과는 experiments/.cache/times_<이름>.json (저장소에 올리지 않는다).
"""
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.kb import load_chunks  # noqa: E402
from core.llm import LLM  # noqa: E402
from core.search import Searcher, load_vectors  # noqa: E402
from experiments.split import load_items  # noqa: E402
from pipelines.v2 import prompts as P  # noqa: E402
from pipelines.v6.pipeline import Pipeline  # noqa: E402

CACHE = Path(__file__).resolve().parent / ".cache"
NAMES = {P.BLOCK_SYSTEM_LEGACY: "block", P.OTHER_SYSTEM: "other", P.MATERIAL_SYSTEM: "material", P.UNSUPPORTED_SYSTEM: "unsupported"}


async def run(name: str, delay: float) -> None:
    calls = []

    class TimedLLM(LLM):
        async def chat(self, system, user, **kw):
            started, ok = time.monotonic(), True
            try:
                return await super().chat(system, user, **kw)
            except Exception:
                ok = False
                raise
            finally:
                calls.append({"name": NAMES.get(system, "answer"), "sec": round(time.monotonic() - started, 2), "ok": ok})

    class Staggered(Pipeline):
        async def _ask(self, state, tag, system, user, effort, timeout):
            if delay and tag in ("block", "other"):
                await asyncio.sleep(delay)
            return await super()._ask(state, tag, system, user, effort, timeout)

    llm = TimedLLM()
    chunks = load_chunks(ROOT / "kb")
    pipe = Staggered(Searcher(chunks, load_vectors(ROOT / "index" / "kb_embeddings.json", llm.embedding_model, chunks)), llm)
    sem = asyncio.Semaphore(64)

    async def one(p):
        async with sem:
            body = {"messages": p["messages"]}
            if p.get("candidate_answer"):
                body["candidate_answer"] = p["candidate_answer"]
            started = time.monotonic()
            state = await (pipe.tutor(body) if p["endpoint"] == "tutor" else pipe.guard(body))
            return {"id": p["id"], "sec": round(time.monotonic() - started, 2), "errors": state.errors}

    started = time.monotonic()
    rows = await asyncio.gather(*(one(p) for p in load_items("all")))
    CACHE.mkdir(exist_ok=True)
    (CACHE / f"times_{name}.json").write_text(json.dumps(
        {"delay": delay, "total_sec": round(time.monotonic() - started, 1), "rows": rows, "calls": calls}, ensure_ascii=False), encoding="utf-8")


def report() -> None:
    def q(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(len(xs) * p))]

    for path in sorted(CACHE.glob("times_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        secs = [r["sec"] for r in d["rows"]]
        line = f"{path.stem[6:]} (늦춤 {d.get('delay', 0)}초): 전체 {d['total_sec']}초, 요청 중앙 {q(secs, .5)} / 90% {q(secs, .9)} / 최대 {max(secs)}"
        for name in ("material", "block", "other", "answer"):
            xs = [c["sec"] for c in d["calls"] if c["name"] == name]
            if xs:
                line += f" | {name} 중앙 {q(xs, .5)} 90% {q(xs, .9)} 10초 넘김 {sum(x > 10 for x in xs) / len(xs):.1%}"
        print(line)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "report":
        report()
    else:
        asyncio.run(run(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.0))
