"""전체 채점 결과를 한 표로 모은다. 같은 조건에서 잰 점수끼리 견주려고 만든 것이다 (docs/sw/05_로컬_점수표.md).

조건: 공개 400문항 전부(튜터 200 + 가드 200), 주최 측 채점 코드, 튜터 판정 포함, 동시 16개.
실행 (.env를 읽어 들인 뒤). 스위치는 pipelines/vx/pipeline.py 맨 위에 적혀 있다:
    VX_NUM_HINT=1 python experiments/eval_full.py --pipeline vx --split all --name sb_numhint
    python experiments/scoreboard.py            # experiments/.cache/full_*_all_sb_*.json을 모두 읽어 표로 찍는다
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "grader"))
sys.path.insert(0, str(ROOT / "harness"))

import scoring  # noqa: E402
from score_local import WEIGHTS  # noqa: E402

CACHE = Path(__file__).resolve().parent / ".cache"


def mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def row(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    tutor = [r for r in rows if r["endpoint"] == "tutor"]
    guard = [r for r in rows if r["endpoint"] == "guard"]
    ts, gs = [r["score"] for r in tutor], [r["score"] for r in guard]
    agg = scoring.aggregate(ts, gs, WEIGHTS)
    outs = [s for s in gs if s["has_output"]]
    name = path.stem.split("_all_sb_", 1)[1]
    flow = path.stem.split("_", 2)[1]
    return {
        "이름": f"{flow} {name}",
        "총점": agg["total"], "튜터": agg["tutor"], "가드": agg["guard"],
        "차단 F1": round(scoring.block_f1(gs), 3),
        "오검": sum(s["false_block"] for s in gs), "미검": sum(s["missed_block"] for s in gs),
        "범주": round(mean(s["category_correct"] for s in gs if s["category_scored"]), 3),
        "근거 여부": round(mean(s["output_correct"] for s in outs), 3),
        "구간": round(mean(s["span_f1"] for s in outs), 3),
        "안전": round(mean(s["safe_correct"] for s in outs), 3),
        "튜터 action": round(mean(s["action_correct"] for s in ts), 3),
        "정답 사실": round(mean(s["facts"] for s in ts), 3),
        "근거 충실": round(mean(s["faithful"] for s in ts), 3),
        "응해서 0점": sum(bool(s["complied"]) for s in ts),
        "판정 오류": agg["judge_errors"],
        "단계 실패": sum(bool(r["errors"]) for r in rows),
        "시간 중앙": round(statistics.median(r["seconds"] for r in rows), 1),
        "시간 최대": max(r["seconds"] for r in rows),
    }


def main():
    paths = sorted(CACHE.glob("full_*_all_sb_*.json"), key=lambda p: p.stat().st_mtime)
    table = [row(p) for p in paths]
    if not table:
        print("결과가 없다")
        return
    cols = list(table[0])
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in table:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
