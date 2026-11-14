"""저장된 분류 결과(세 확인의 답과 확신 정도)로, 합치는 규칙을 바꿨을 때의 정확도를 계산한다. 모델을 부르지 않는다.

조정용에서 규칙을 고르고 확인용에서는 고른 규칙의 숫자만 본다.
    python experiments/sim_merge.py classify_v2_conf classify_v2_conf2 classify_v2_conf3
"""
import collections
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.split import which  # noqa: E402

CACHE = Path(__file__).resolve().parent / ".cache"
PRIORITY = ["harmful_request", "harmful_speech", "jailbreak_injection", "emotional_distress", "academic_dishonesty",
            "answer_seeking", "off_topic", "low_signal_input", "out_of_material", "self_info"]
CATEGORY = {"harmful_request": "harmful_content", "harmful_speech": "harmful_content", "self_info": "pii_disclosure"}
BLOCK = {"harmful_content", "jailbreak_injection", "academic_dishonesty", "off_topic"}


def conf(x, default=100):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else default


def predict(d, tb=0, to=0, tm=0, ts=0, seek_first=False):
    """d: debug 칸. tb/to/tm/ts: 차단·그 밖·교재 밖·자기 정보 판정을 받아들이는 최소 확신."""
    flags = set()
    if d["block"][0] not in (None, "none") and conf(d["block"][1]) >= tb:
        flags.add(d["block"][0])
    if d["other"][0] not in (None, "none") and conf(d["other"][1]) >= to:
        flags.add(d["other"][0])
    if d["self_info"][0] is True and conf(d["self_info"][1]) >= ts:
        flags.add("self_info")
    if d["material"][0] and conf(d["material"][1]) >= tm:
        flags.add("out_of_material")
    order = PRIORITY
    if seek_first:  # 정답만 요구가 과제 대행보다 앞선다
        order = [k for k in PRIORITY if k != "answer_seeking"]
        order.insert(order.index("academic_dishonesty"), "answer_seeking")
    for k in order:
        if k in flags:
            return CATEGORY.get(k, k)
    return "normal"


def stats(rows, **kw):
    p = [predict(r["debug"], **kw) for r in rows]
    acc = sum(a == r["gold"] for a, r in zip(p, rows)) / len(rows)
    tp = sum(a in BLOCK and r["gold"] in BLOCK for a, r in zip(p, rows))
    fp = sum(a in BLOCK and r["gold"] not in BLOCK for a, r in zip(p, rows))
    fn = sum(a not in BLOCK and r["gold"] in BLOCK for a, r in zip(p, rows))
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
    return acc, f1


if __name__ == "__main__":
    rows = []
    for name in sys.argv[1:]:
        rows += json.loads((CACHE / f"{name}.json").read_text(encoding="utf-8"))
    dev = [r for r in rows if which(r["id"]) == "dev"]
    hold = [r for r in rows if which(r["id"]) == "holdout"]
    print(f"조정용 {len(dev)}건, 확인용 {len(hold)}건 (실행 {len(sys.argv) - 1}번을 합침)")

    def line(name, **kw):
        a, f = stats(dev, **kw)
        b, g = stats(hold, **kw)
        print(f"{name}: 조정용 정확도 {a:.1%} 차단F1 {f:.3f} | 확인용 정확도 {b:.1%} 차단F1 {g:.3f}")

    line("지금 규칙")
    line("정답만 요구를 과제 대행 앞에", seek_first=True)
    print("\n기준선 하나씩 (조정용에서):")
    for key in ("tb", "to", "tm", "ts"):
        print(" ", key, " ".join(f"{t}:{stats(dev, **{key: t})[0]:.1%}" for t in (0, 50, 65, 75, 80, 85, 88, 90)))
    best, best_kw = 0, None
    for tb, to, tm, sf in itertools.product((0, 75, 80, 85), (0, 75, 80, 85), (0, 50, 65, 75, 85), (False, True)):
        a, f = stats(dev, tb=tb, to=to, tm=tm, seek_first=sf)
        if a + f > best:
            best, best_kw = a + f, dict(tb=tb, to=to, tm=tm, seek_first=sf)
    print("\n조정용에서 고른 규칙(정확도 + 차단 F1이 가장 큰 것):", best_kw)
    line("고른 규칙", **best_kw)
