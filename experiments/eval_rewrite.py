"""리라이터가 만든 검색어로 낱말 검색 1위가 정답 섹션인 비율을 잰다.

공개 400문항에 리라이터를 한 번씩 돌려 결과를 experiments/.cache/rewrite_<이름>.json에 저장하고,
정답 근거가 있는 문항에서 검색어 조합별 적중률을 찍는다. 임베딩은 쓰지 않는다.

실행: python experiments/eval_rewrite.py --name first        (다시 부르지 않고 표만: --reuse)
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.kb import load_chunks  # noqa: E402
from core.search import Searcher  # noqa: E402
from pipelines.v2 import prompts  # noqa: E402

CACHE = Path(__file__).resolve().parent / ".cache"


def load_items():
    items = []
    for endpoint in ("tutor", "guard"):
        for line in (ROOT / "public" / f"{endpoint}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                items.append(json.loads(line))
    return items


async def rewrite_all(items, effort, workers):
    from core.llm import LLM
    llm = LLM()
    sem = asyncio.Semaphore(workers)

    async def one(item):
        async with sem:
            history, last = item["messages"][:-1], item["messages"][-1]["content"]
            started = time.time()
            try:
                text = await llm.chat(prompts.REWRITE_SYSTEM, prompts.dialogue(history, last),
                                      effort=effort, max_tokens=1500, timeout=20, json_mode=True)
                obj = json.loads(text[text.find("{"):text.rfind("}") + 1])
            except Exception as e:
                obj = {"error": type(e).__name__}
            return {"id": item["id"], "request": obj.get("request", ""), "query": obj.get("query", ""),
                    "error": obj.get("error"), "seconds": round(time.time() - started, 2)}

    return await asyncio.gather(*(one(it) for it in items))


def report(items, rows):
    by_id = {r["id"]: r for r in rows}
    searcher = Searcher(load_chunks(ROOT / "kb"))
    ev = [it for it in items if it["labels"].get("evidence")]
    variants = {
        "대화 전체 (지금)": lambda it, r: "\n".join(m["content"] for m in it["messages"]),
        "마지막 발화만": lambda it, r: it["messages"][-1]["content"],
        "검색어만": lambda it, r: r["query"],
        "요청 풀이만": lambda it, r: r["request"],
        "요청 풀이 + 검색어": lambda it, r: r["request"] + "\n" + r["query"],
        "마지막 발화 + 검색어": lambda it, r: it["messages"][-1]["content"] + "\n" + r["query"],
        "대화 전체 + 검색어": lambda it, r: "\n".join(m["content"] for m in it["messages"]) + "\n" + r["query"],
        "대화 전체 + 요청 풀이 + 검색어": lambda it, r: "\n".join(m["content"] for m in it["messages"]) + "\n" + r["request"] + "\n" + r["query"],
    }
    print(f"리라이터 실패 {sum(bool(r['error']) for r in rows)}/{len(rows)}, "
          f"걸린 시간 중앙 {sorted(r['seconds'] for r in rows)[len(rows) // 2]}초, 최대 {max(r['seconds'] for r in rows)}초")
    print(f"\n검색어 | 1위 | 3위 안 | 10조각 안   (정답 근거가 있는 {len(ev)}문항, 낱말 검색)")
    for name, make in variants.items():
        top1 = top3 = top10 = 0
        for it in ev:
            gold = it["labels"]["evidence"][0]
            q = make(it, by_id[it["id"]])
            if it.get("candidate_answer"):
                q += "\n" + it["candidate_answer"]
            found = searcher.search(q)
            order = list(dict.fromkeys(c.section_id for c in found))
            top1 += order[:1] == [gold]
            top3 += gold in order[:3]
            top10 += gold in order
        print(f"{name} | {top1} ({top1 / len(ev):.0%}) | {top3} | {top10}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    items = load_items()
    path = CACHE / f"rewrite_{a.name}.json"
    if a.reuse:
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        rows = asyncio.run(rewrite_all(items, a.effort, a.workers))
        CACHE.mkdir(exist_ok=True)
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    report(items, rows)
