"""예시에 확신 정도를 묶어 주면 모델이 적는 확신 정도가 쓸 만해지는지 본다 (2026-11-16, 이승원 제안).

세 확인(차단, 그 밖, 교재)의 프롬프트에서 두 가지만 바꾼다.
  1. 예시마다 확신 정도를 붙인다. 뻔한 것은 95, 짝을 이루는 헷갈리는 것은 80~85, 이유를 달아 둔 경계 사례는 70~75.
  2. 확신 정도의 기준을 한 문단 적는다.
판정하는 코드는 그대로다(확신 정도는 기록만 한다). 분류만 돌리고 답변은 만들지 않는다.

실행 (.env를 읽어 들인 뒤):
    python experiments/conf_examples.py base   # 지금 프롬프트
    python experiments/conf_examples.py conf   # 예시에 확신 정도를 붙인 프롬프트
    python experiments/conf_examples.py report # 둘을 비교
결과는 experiments/.cache/confex_<이름>.json (저장소에 올리지 않는다).
"""
import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.kb import load_chunks  # noqa: E402
from core.llm import LLM  # noqa: E402
from core.search import Searcher, load_vectors  # noqa: E402
from core.state import State  # noqa: E402
from experiments.split import load_items  # noqa: E402
from pipelines.v2 import prompts as P  # noqa: E402

CACHE = Path(__file__).resolve().parent / ".cache"

SCALE = """
# confidence를 적는 기준
- 95: 정의에 그대로 들어맞는다. 또는 어디에도 전혀 해당하지 않는 평범한 질문이다.
- 80~85: 그렇게 판단했지만 다르게 읽힐 여지가 조금 있다.
- 60~75: 어느 쪽인지 헷갈린다. 헷갈리면 숨기지 말고 낮게 적는다.
예시 끝의 "확신"이 그 예시에 적을 값이다.
"""

BLOCK_CONF = [95, 90, 85, 95, 80, 85, 80, 80, 85, 95, 70, 75, 70]
OTHER_CONF = [95, 95, 95]
MATERIAL_CONF = [95, 85, 90]


def with_confidence(system: str, values) -> str:
    head, rest = system.split("# 예시\n", 1)
    examples, tail = rest.split("\n# 출력\n", 1)
    it = iter(values)
    lines = [f"{line} · 확신 {next(it)}" if line.startswith('- "') else line for line in examples.split("\n")]
    assert next(it, None) is None, "예시 수와 확신 값 수가 다르다"
    return head + "# 예시\n" + "\n".join(lines) + "\n" + SCALE + "\n# 출력\n" + tail


def apply_conf_prompts() -> None:
    P.BLOCK_SYSTEM = with_confidence(P.BLOCK_SYSTEM, BLOCK_CONF)
    P.OTHER_SYSTEM = with_confidence(P.OTHER_SYSTEM, OTHER_CONF)
    P.MATERIAL_SYSTEM = with_confidence(P.MATERIAL_SYSTEM, MATERIAL_CONF)


async def run(name: str) -> None:
    if name == "conf":
        apply_conf_prompts()
    from pipelines.v4.pipeline import Pipeline
    llm = LLM()
    chunks = load_chunks(ROOT / "kb")
    pipe = Pipeline(Searcher(chunks, load_vectors(ROOT / "index" / "kb_embeddings.json", llm.embedding_model, chunks)), llm)
    pipe.prepare_limit = 60
    sem = asyncio.Semaphore(32)

    async def one(p):
        async with sem:
            body = {"messages": p["messages"]}
            if p.get("candidate_answer"):
                body["candidate_answer"] = p["candidate_answer"]
            state = State.from_request(body)
            await pipe.prepare(state, state.candidate_answer or "")
            return {"id": p["id"], "gold": p["labels"]["category"], "pred": state.category.value,
                    "debug": state.debug, "errors": state.errors}

    rows = await asyncio.gather(*(one(p) for p in load_items("all")))
    CACHE.mkdir(exist_ok=True)
    (CACHE / f"confex_{name}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


BLOCK_GOLD = {"harmful_request": "harmful_content", "harmful_speech": "harmful_content", "jailbreak_injection": "jailbreak_injection",
              "academic_dishonesty": "academic_dishonesty", "off_topic": "off_topic"}
OTHER_GOLD = {"emotional_distress", "answer_seeking", "low_signal_input"}
MATERIAL_GOLD = {"normal", "pii_disclosure", "answer_seeking", "out_of_material"}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def report() -> None:
    for name in ("base", "conf"):
        path = CACHE / f"confex_{name}.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        print(f"\n===== {name}: 범주 정확도 {sum(r['gold'] == r['pred'] for r in rows) / len(rows):.1%}, 단계 실패 {sum(bool(r['errors']) for r in rows)}")
        checks = {"차단 확인": [], "그 밖 확인": [], "교재 확인": []}
        for r in rows:
            d = r.get("debug") or {}
            cat, conf = (d.get("block") or [None, None])
            if cat is not None and num(conf) is not None:
                right = BLOCK_GOLD.get(cat) == r["gold"] if cat != "none" else r["gold"] not in BLOCK_GOLD.values()
                checks["차단 확인"].append((num(conf), right, cat != "none"))
            cat, conf = (d.get("other") or [None, None])
            if cat is not None and num(conf) is not None:
                right = cat == r["gold"] if cat != "none" else r["gold"] not in OTHER_GOLD
                checks["그 밖 확인"].append((num(conf), right, cat != "none"))
            miss, conf = (d.get("material") or [None, None])
            if r["gold"] in MATERIAL_GOLD and num(conf) is not None:
                checks["교재 확인"].append((num(conf), miss == (r["gold"] == "out_of_material"), bool(miss)))
        for check, vals in checks.items():
            n = len(vals)
            print(f"[{check}] {n}건, 맞음 {sum(v[1] for v in vals) / n:.1%}. 값 분포: {Counter(int(v[0]) for v in vals).most_common(8)}")
            for lo, hi in ((0, 70), (70, 85), (85, 93), (93, 101)):
                part = [v for v in vals if lo <= v[0] < hi]
                if part:
                    flagged = [v for v in part if v[2]]
                    print(f"   확신 {lo}~{hi - 1}: {len(part)}건 중 {sum(v[1] for v in part)}건 맞음 ({sum(v[1] for v in part) / len(part):.0%})"
                          + (f" / 무언가를 잡은 판정 {len(flagged)}건 중 {sum(v[1] for v in flagged)}건 맞음" if flagged else ""))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    if mode == "report":
        report()
    else:
        asyncio.run(run(mode))
