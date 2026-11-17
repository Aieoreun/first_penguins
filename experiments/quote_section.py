"""교재 확인이 베껴 온 문장이 들어 있는 섹션을 코드로 찾아 근거 섹션으로 쓰면 낱말 검색 1위보다 나은지 본다 (2026-11-17).

지금은 근거 섹션을 낱말 검색(BM25) 1위 조각의 섹션으로 정한다(정답 근거가 있는 공개 문항에서 91% 맞음).
모델에게 섹션 ID를 적게 했을 때는 76%였다. 여기서는 모델이 적은 ID가 아니라, 베껴 온 문장이 실제로 들어 있는 조각을 코드로 찾는다.

정답 근거가 있는 공개 튜터 문항만 쓴다. 문항마다 검색 한 번, 교재 확인 호출 한 번이다.
실행 (.env를 읽어 들인 뒤):
    python experiments/quote_section.py run1
    python experiments/quote_section.py report
결과는 experiments/.cache/quote_<이름>.json (저장소에 올리지 않는다).
"""
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.kb import load_chunks  # noqa: E402
from core.llm import LLM  # noqa: E402
from core.search import Searcher, load_vectors  # noqa: E402
from core.spans import locate  # noqa: E402
from core.state import State  # noqa: E402
from experiments.split import load_items  # noqa: E402
from pipelines.v1.pipeline import CLASSIFY_EFFORT  # noqa: E402
from pipelines.v2 import prompts as P  # noqa: E402
from pipelines.v8.pipeline import Pipeline  # noqa: E402

CACHE = Path(__file__).resolve().parent / ".cache"


def evidence_ids(item):
    out = []
    for e in item["labels"].get("evidence") or []:
        out.append(e if isinstance(e, str) else e.get("section_id") or e.get("id") or "")
    return [e for e in out if e]


async def run(name: str) -> None:
    llm = LLM()
    chunks = load_chunks(ROOT / "kb")
    pipe = Pipeline(Searcher(chunks, load_vectors(ROOT / "index" / "kb_embeddings.json", llm.embedding_model, chunks)), llm)
    items = [p for p in load_items("all") if p["endpoint"] == "tutor" and evidence_ids(p)]
    sem = asyncio.Semaphore(8)

    async def one(p):
        async with sem:
            state = State.from_request({"messages": p["messages"]})
            await pipe._search(state)
            user = P.material_user(pipe._material_chunks(state), state.request, state.history, state.last_user)
            answer = await pipe._ask(state, "material", P.MATERIAL_SYSTEM, user, CLASSIFY_EFFORT, 30)
            quote = answer.get("quote") if isinstance(answer.get("quote"), str) else ""
            # 베낀 문장이 들어 있는 조각을 검색 순서대로 찾는다. 띄어쓰기 차이는 locate가 넘어간다
            hits = [c.section_id for c in state.chunks if quote.strip() and locate(c.text, quote)]
            model_section = answer.get("section") if isinstance(answer.get("section"), str) else ""
            return {"id": p["id"], "gold": evidence_ids(p), "category": p["labels"]["category"],
                    "candidates": [c.section_id for c in state.chunks], "quote_len": len(quote.strip()),
                    "quote_sections": list(dict.fromkeys(hits)), "model_section": model_section.strip(), "errors": state.errors}

    rows = await asyncio.gather(*(one(p) for p in items))
    CACHE.mkdir(exist_ok=True)
    (CACHE / f"quote_{name}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def report() -> None:
    for path in sorted(CACHE.glob("quote_*.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        n = len(rows)
        top1 = lambda r: r["candidates"][0] if r["candidates"] else None  # noqa: E731
        quote = lambda r: r["quote_sections"][0] if r["quote_sections"] else None  # noqa: E731
        ok = lambda r, s: s in r["gold"]  # noqa: E731
        found = [r for r in rows if quote(r)]
        differ = [r for r in found if quote(r) != top1(r)]
        print(f"\n##### {path.stem[6:]}: 문항 {n}, 단계 실패 {sum(bool(r['errors']) for r in rows)}")
        print(f"낱말 검색 1위가 맞음: {sum(ok(r, top1(r)) for r in rows)}/{n}")
        print(f"모델이 적은 섹션 ID가 맞음: {sum(ok(r, r['model_section']) for r in rows)}/{n}")
        print(f"문장을 베껴 옴: {sum(r['quote_len'] > 0 for r in rows)}/{n}, 그 문장을 검색 조각에서 찾음: {len(found)}/{n}")
        print(f"  찾은 것 중 그 섹션이 맞음: {sum(ok(r, quote(r)) for r in found)}/{len(found)}")
        print(f"베낀 문장의 섹션을 쓰고, 못 찾으면 낱말 검색 1위를 씀: {sum(ok(r, quote(r) or top1(r)) for r in rows)}/{n}")
        print(f"두 방법이 다른 섹션을 가리킨 문항 {len(differ)}개: 베낀 문장 쪽이 맞음 {sum(ok(r, quote(r)) for r in differ)}, "
              f"낱말 검색 1위 쪽이 맞음 {sum(ok(r, top1(r)) for r in differ)}, 둘 다 틀림 "
              f"{sum(not ok(r, quote(r)) and not ok(r, top1(r)) for r in differ)}")
        print(f"낱말 검색 1위가 틀린 문항 {sum(not ok(r, top1(r)) for r in rows)}개 중 베낀 문장이 맞힌 것: "
              f"{sum(ok(r, quote(r)) for r in rows if not ok(r, top1(r)) and quote(r))}")
        print("못 찾은 문항의 범주:", dict(Counter(r["category"] for r in rows if not quote(r))))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    if mode == "report":
        report()
    else:
        asyncio.run(run(mode))
