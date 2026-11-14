"""입력 가드(분류 + 근거 섹션 고르기)만 따로 재는 실험. 서버를 띄우지 않고 흐름(pipelines/)의 prepare()를 직접 부른다.

공개 세트 400문항(튜터 200, 가드 200)에 검색과 분류를 돌려서
  - 범주별 정확도와, 정답 범주가 무엇으로 잘못 분류됐는지
  - 정답 근거가 있는 문항에서 고른 섹션이 맞는 비율
을 찍는다. 문항별 결과는 experiments/.cache/classify_<흐름>_<이름>.json에 저장한다(저장소에 올리지 않는다).

실행 (저장소 폴더에서, .env를 읽어 들인 뒤):
    python experiments/eval_classify.py --pipeline v1 --name first
환경변수: OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL
"""
import argparse
import asyncio
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.kb import load_chunks  # noqa: E402
from core.llm import LLM  # noqa: E402
from core.search import Searcher, load_vectors  # noqa: E402
from core.state import State  # noqa: E402
from pipelines import load  # noqa: E402

CACHE = Path(__file__).resolve().parent / ".cache"
# 분류 프롬프트의 예시로 쓴 공개 문항. 정확도를 셀 때 뺀 값도 함께 찍는다
PROMPT_EXAMPLES = ROOT / "docs" / "sw" / "03_분류_프롬프트_초안.md"


def example_ids():
    import re
    return set(re.findall(r"`(pub-[0-9a-f]{8})`", PROMPT_EXAMPLES.read_text(encoding="utf-8")))


async def run(args):
    llm = LLM()
    chunks = load_chunks(ROOT / "kb")
    vectors = load_vectors(ROOT / "index" / "kb_embeddings.json", llm.embedding_model, chunks)
    pipe = load(args.pipeline)(Searcher(chunks, vectors), llm)
    from experiments.split import load_items
    items = load_items(args.split)
    if args.limit:
        items = items[:args.limit]
    sem = asyncio.Semaphore(args.workers)

    async def one(item):
        async with sem:
            state = State.from_request(item)
            started = time.time()
            await pipe.prepare(state, extra=state.candidate_answer or "")
            labels = item["labels"]
            return {
                "id": item["id"], "endpoint": item["endpoint"], "gold": labels["category"],
                "pred": state.category.value, "label": state.label.value if state.label else None,
                "gold_section": (labels.get("evidence") or [None])[0], "section": state.section,
                "top1": state.chunks[0].section_id if state.chunks else None, "debug": state.debug,
                "in_candidates": (labels.get("evidence") or [None])[0] in {c.section_id for c in state.chunks},
                "errors": state.errors, "seconds": round(time.time() - started, 2),
            }

    rows = await asyncio.gather(*(one(it) for it in items))
    CACHE.mkdir(exist_ok=True)
    (CACHE / f"classify_{args.pipeline}_{args.split}_{args.name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    report(rows, vectors is not None)


def report(rows, has_embeddings):
    skip = example_ids()
    print(f"문항 {len(rows)}개, 임베딩 {'있음' if has_embeddings else '없음'}, 단계 실패 {sum(bool(r['errors']) for r in rows)}개")
    for name, part in (("전체", rows), ("프롬프트 예시 문항 제외", [r for r in rows if r["id"] not in skip])):
        ok = sum(r["gold"] == r["pred"] for r in part)
        print(f"범주 정확도 ({name}): {ok}/{len(part)} = {ok / len(part):.1%}")
    by = defaultdict(list)
    for r in rows:
        by[r["gold"]].append(r)
    print("\n정답 범주 | 맞음/전체 | 잘못 분류된 곳")
    for gold in sorted(by, key=lambda g: sum(r["pred"] == g for r in by[g]) / len(by[g])):
        part = by[gold]
        wrong = Counter(r["pred"] for r in part if r["pred"] != gold)
        print(f"{gold} | {sum(r['pred'] == gold for r in part)}/{len(part)} | "
              + ", ".join(f"{k} {v}" for k, v in wrong.most_common()))
    over = Counter(r["pred"] for r in rows if r["pred"] != r["gold"])
    print("\n잘못 붙인 범주(붙인 쪽 기준): " + ", ".join(f"{k} {v}" for k, v in over.most_common()))
    ev = [r for r in rows if r["gold_section"]]
    if ev:
        print(f"\n근거 섹션 (정답 근거가 있는 {len(ev)}문항): 후보에 있음 {sum(r['in_candidates'] for r in ev)}, "
              f"검색 1위가 맞음 {sum(r['top1'] == r['gold_section'] for r in ev)}, "
              f"쓴 것이 맞음 {sum(r['section'] == r['gold_section'] for r in ev)}")
    secs = sorted(r["seconds"] for r in rows)
    print(f"걸린 시간: 중앙 {secs[len(secs) // 2]}초, 90% {secs[int(len(secs) * 0.9)]}초, 최대 {secs[-1]}초")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="v1")
    ap.add_argument("--name", default="run")
    ap.add_argument("--split", default="all", choices=["all", "dev", "holdout"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    asyncio.run(run(a))
