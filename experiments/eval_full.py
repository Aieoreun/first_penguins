"""흐름 전체를 공개 세트로 채점하고, 점수를 어디서 잃는지 항목별로 찍는다.

서버를 띄우지 않고 흐름을 직접 부른 뒤 주최 측 채점 코드(grader/scoring.py, grader/judge.py)로 채점한다.
총점 계산은 harness/score_local.py와 같다. 판정 모델은 gpt-5-nano라 실제 채점과 다르다.

실행 (저장소 폴더에서, .env를 읽어 들인 뒤):
    python experiments/eval_full.py --pipeline v2 --split dev --name base
문항별 결과는 experiments/.cache/full_<흐름>_<나눔>_<이름>.json에 저장한다(저장소에 올리지 않는다).
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "grader"))
sys.path.insert(0, str(ROOT / "harness"))

import scoring  # noqa: E402
from judge import make_judge  # noqa: E402
from score_local import WEIGHTS, as_scoring_input, load_kb_dir  # noqa: E402

from core.kb import load_chunks  # noqa: E402
from core.llm import LLM  # noqa: E402
from core.search import Searcher, load_vectors  # noqa: E402
from experiments.split import load_items  # noqa: E402
from pipelines import load  # noqa: E402

CACHE = Path(__file__).resolve().parent / ".cache"


async def run(args):
    llm = LLM()
    chunks = load_chunks(ROOT / "kb")
    vectors = load_vectors(ROOT / "index" / "kb_embeddings.json", llm.embedding_model, chunks)
    pipe = load(args.pipeline)(Searcher(chunks, vectors), llm)
    kb = load_kb_dir(ROOT / "kb")
    judge = None if args.no_judge else make_judge(
        os.environ["OPENAI_BASE_URL"], os.environ["OPENAI_MODEL"], os.environ["OPENAI_API_KEY"])
    items = load_items(args.split)
    sem = asyncio.Semaphore(args.workers)
    pool = ThreadPoolExecutor(args.workers)
    loop = asyncio.get_running_loop()

    def no_judge(kind, payload):
        raise RuntimeError("판정 없음")

    def score(p, resp):
        item, v = as_scoring_input(p)
        if p["endpoint"] == "tutor":
            r = scoring.score_tutor(item, v, resp, judge or no_judge, kb)
            if judge is None:
                r["judge_error"] = False
            return r, scoring.tutor_points(r)
        r = scoring.score_guard(item, v, resp)
        return r, scoring.guard_points(r)

    async def one(p):
        async with sem:
            body = {"messages": p["messages"]}
            if p.get("candidate_answer"):
                body["candidate_answer"] = p["candidate_answer"]
            started = time.time()
            state = await (pipe.tutor(body) if p["endpoint"] == "tutor" else pipe.guard(body))
            seconds = round(time.time() - started, 2)
            resp = state.to_tutor_response() if p["endpoint"] == "tutor" else state.to_guard_response()
            r, pts = await loop.run_in_executor(pool, score, p, resp)
            return {"id": p["id"], "endpoint": p["endpoint"], "gold": p["labels"]["category"],
                    "pred": state.category.value, "label": state.label.value if state.label else None,
                    "section": state.section, "errors": state.errors, "seconds": seconds,
                    "response": resp, "score": r, "points": pts, "debug": state.debug}

    rows = await asyncio.gather(*(one(p) for p in items))
    CACHE.mkdir(exist_ok=True)
    out = CACHE / f"full_{args.pipeline}_{args.split}_{args.name}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    report(rows)


def mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return f"{sum(xs) / len(xs):.3f} ({len(xs)})" if xs else "-"


def report(rows):
    tutor = [r for r in rows if r["endpoint"] == "tutor"]
    guard = [r for r in rows if r["endpoint"] == "guard"]
    agg = scoring.aggregate([r["score"] for r in tutor], [r["score"] for r in guard], WEIGHTS)
    print(f"총점 {agg['total']} (튜터 {agg['tutor']}, 가드 {agg['guard']}), 문항 {len(rows)}, "
          f"형식 오류 {agg['invalid']}, 판정 오류 {agg['judge_errors']}, 단계 실패 {sum(bool(r['errors']) for r in rows)}")
    ok = sum(r["gold"] == r["pred"] for r in rows)
    print(f"범주 정확도 {ok}/{len(rows)} = {ok / len(rows):.1%}")

    ts = [r["score"] for r in tutor]
    print("\n[튜터] 항목별 평균 (문항 수)")
    print("  action 일치:", mean(s["action_correct"] for s in ts))
    f1 = [2 * s["citation_p"] * s["citation_r"] / (s["citation_p"] + s["citation_r"])
          if s["citation_p"] is not None and s["citation_p"] + s["citation_r"] else (0.0 if s["citation_p"] is not None else None)
          for s in ts]
    print("  인용 F1:", mean(f1))
    print("  정답 사실:", mean(s["facts"] for s in ts))
    print("  근거 충실:", mean(s["faithful"] for s in ts))
    print(f"  요청에 응해 0점: {sum(bool(s['complied']) for s in ts)}, 과한 거절: {sum(s['over_refusal'] for s in ts)}")
    right = [r for r in tutor if r["gold"] == r["pred"]]
    print(f"  범주를 맞힌 문항만의 점수: {mean(r['points'] for r in right)}, 틀린 문항만: {mean(r['points'] for r in tutor if r['gold'] != r['pred'])}")

    gs = [r["score"] for r in guard]
    print("\n[가드] 항목별")
    print(f"  차단 판정 F1: {scoring.block_f1(gs):.3f} (잘못 차단 {sum(s['false_block'] for s in gs)}, 놓친 차단 {sum(s['missed_block'] for s in gs)})")
    print("  범주 정확도(정상 아닌 문항):", mean(s["category_correct"] for s in gs if s["category_scored"]))
    outs = [s for s in gs if s["has_output"]]
    print("  후보 답변 - 근거 여부 일치:", mean(s["output_correct"] for s in outs))
    print("  후보 답변 - 구간 F1:", mean(s["span_f1"] for s in outs))
    print("  후보 답변 - 안전 일치:", mean(s["safe_correct"] for s in outs))

    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["gold"]][r["endpoint"]].append(r["points"])
    print("\n범주 | 튜터 점수 (문항) | 가드 문항 점수 (문항)")
    for cat in sorted(by):
        print(f"  {cat} | {mean(by[cat]['tutor'])} | {mean(by[cat]['guard'])}")
    for name, part in (("튜터", tutor), ("가드", guard)):
        secs = sorted(r["seconds"] for r in part)
        if secs:
            print(f"{name} 걸린 시간: 중앙 {statistics.median(secs):.1f}초, 90% {secs[int(len(secs) * 0.9)]}초, 최대 {secs[-1]}초")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="v2")
    ap.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    ap.add_argument("--name", default="run")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--report", action="store_true", help="저장된 결과로 표만 다시 찍는다")
    a = ap.parse_args()
    if a.report:
        report(json.loads((CACHE / f"full_{a.pipeline}_{a.split}_{a.name}.json").read_text(encoding="utf-8")))
    else:
        asyncio.run(run(a))
