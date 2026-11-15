"""저장된 한 표짜리 실행 세 번을 같은 문항의 세 표로 보고, 확인 호출 수를 줄였을 때의 정확도를 계산한다.

모델을 부르지 않는다. 합치는 순서는 지금 설정(정답만 요구가 과제 대행보다 앞)을 쓴다.
규칙은 조정용에서만 고르고, 확인용은 표에 적은 규칙의 숫자만 본다. 문항 내용은 찍지 않는다.
    python3 experiments/sim_votes.py
    python3 experiments/sim_votes.py classify_v2_conf classify_v2_conf2 classify_v2_conf3

모의실험 1 (부분 다수결): 확인마다 표 수를 다르게 준다. 한 표만 쓰는 확인은 어느 실행의 표를 쓰는지에
대해 평균한다. 세 표 동률은 가장 앞 실행의 값.
모의실험 2 (조건부 추가 호출): 확인마다 한 번 부르고, 조건에 걸린 확인만 두 번 더 불러 다수결한다.
첫 호출을 실행 1·2·3으로 바꿔 가며 평균한다. 동률은 첫 호출의 값.
"그 밖" 확인은 한 번의 호출이 범주와 자기 정보 여부를 함께 내므로 둘을 같이 다수결한다.
"""
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.sim_merge import BLOCK, CACHE, conf, predict  # noqa: E402
from experiments.split import prompt_example_ids, which  # noqa: E402

DEFAULT_RUNS = ["classify_v2_conf", "classify_v2_conf2", "classify_v2_conf3"]
SPLITS = ("dev", "holdout")


def majority(values):
    """가장 많이 나온 값. 동률이면 먼저 나온 값 (pipelines/v2/pipeline.py의 _majority와 같다)."""
    counts = {}
    for v in values:
        counts[repr(v)] = counts.get(repr(v), 0) + 1
    return max(values, key=lambda v: counts[repr(v)])


def merged(block, other, self_info, material):
    """확인별로 정해진 값을 지금 순서로 합친다. 확신 정도는 쓰지 않는다."""
    d = {"block": [block, None], "other": [other, None], "self_info": [self_info, None], "material": [material, None]}
    return predict(d, seek_first=True)


def score(preds, golds):
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    tp = sum(p in BLOCK and g in BLOCK for p, g in zip(preds, golds))
    fp = sum(p in BLOCK and g not in BLOCK for p, g in zip(preds, golds))
    fn = sum(p not in BLOCK and g in BLOCK for p, g in zip(preds, golds))
    return acc, (2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def value(item, check, order):
    """order: 이 확인에 쓰는 실행 번호들. 하나면 그 표, 여럿이면 다수결."""
    return majority([item["votes"][i][check] for i in order])


def fixed_scheme(items, nb, no, nm):
    """확인별 표 수(1 또는 3)를 고정한 방식. 한 표인 확인은 실행 선택의 모든 조합을 돌려준다."""
    def choices(n):
        return [(0, 1, 2)] if n == 3 else [(0,), (1,), (2,)]
    results = []
    combos = itertools.product(choices(nb), choices(no), choices(nm))
    if (nb, no, nm) == (1, 1, 1):  # 전부 한 표면 실제 실행 세 번 그대로 (한 실행이 세 확인을 모두 댄다)
        combos = [((i,), (i,), (i,)) for i in range(3)]
    for ob, oo, om in combos:
        preds = [merged(value(it, "block", ob), value(it, "other", oo), value(it, "self_info", oo),
                        value(it, "material", om)) for it in items]
        results.append(score(preds, [it["gold"] for it in items]))
    return results


def escalate(items, first, block_any=False, block_conf=None, other_any=False, material_true=False):
    """첫 호출(first번 실행)을 보고 조건에 걸린 확인만 나머지 두 표를 더 받는다. (정확도, F1, 평균 호출 수)"""
    order = (first,) + tuple(i for i in range(3) if i != first)
    preds, calls = [], 0
    for it in items:
        v = it["votes"][first]
        more_b = (block_any and v["block"] not in (None, "none")) or \
                 (block_conf is not None and conf(v["block_conf"]) < block_conf)
        more_o = other_any and v["other"] not in (None, "none")
        more_m = material_true and bool(v["material"])
        ob, oo, om = (order if more_b else (first,)), (order if more_o else (first,)), (order if more_m else (first,))
        preds.append(merged(value(it, "block", ob), value(it, "other", oo), value(it, "self_info", oo),
                            value(it, "material", om)))
        calls += 3 + 2 * (more_b + more_o + more_m)
    acc, f1 = score(preds, [it["gold"] for it in items])
    return acc, f1, calls / len(items)


def cell(results, spread=True):
    """(정확도, F1) 여러 개 -> '평균 (최소~최대)' 글자."""
    accs, f1s = [r[0] for r in results], [r[1] for r in results]
    if len(results) == 1 or not spread:
        return f"{mean(accs):6.1%} {mean(f1s):.3f}"
    return f"{mean(accs):6.1%} ({min(accs):.1%}~{max(accs):.1%}) {mean(f1s):.3f} ({min(f1s):.3f}~{max(f1s):.3f})"


def main(names):
    runs = [json.loads((CACHE / f"{n}.json").read_text(encoding="utf-8")) for n in names]
    assert len(runs) == 3, "실행 세 번이 필요하다"
    examples = prompt_example_ids()

    # 0. 합치는 함수가 저장된 pred를 그대로 다시 만드는지 (기본 순서, 한 표)
    print("== 확인: predict(기본 순서)가 저장된 pred와 같은 비율 ==")
    for n, rows in zip(names, runs):
        same = sum(predict(r["debug"]) == r["pred"] for r in rows)
        print(f"  {n}: {same}/{len(rows)} ({same / len(rows):.1%})")

    by_id = [{r["id"]: r for r in rows} for rows in runs]
    ids = [r["id"] for r in runs[0]]
    assert all(set(b) == set(ids) for b in by_id), "세 실행의 문항이 다르다"
    assert all(by_id[i][x]["gold"] == by_id[0][x]["gold"] for i in (1, 2) for x in ids), "정답이 다르다"
    items = {s: [] for s in SPLITS}
    ties = 0
    for x in ids:
        votes = []
        for b in by_id:
            d = b[x]["debug"]
            votes.append({"block": d["block"][0], "block_conf": d["block"][1], "other": d["other"][0],
                          "self_info": d["self_info"][0] is True, "material": bool(d["material"][0])})
        ties += sum(len({repr(v[c]) for v in votes}) == 3 for c in ("block", "other"))
        items[which(x, examples)].append({"gold": by_id[0][x]["gold"], "votes": votes})
    dev, hold = items["dev"], items["holdout"]
    print(f"문항: 조정용 {len(dev)}건, 확인용 {len(hold)}건. 세 표가 모두 달라 동률 규칙이 쓰인 (문항, 확인) 수: {ties}")

    # 1. 부분 다수결
    schemes = [
        ("(a) 전부 1표", 1, 1, 1),
        ("(b) 전부 3표", 3, 3, 3),
        ("(c) 차단3 그밖1 교재1", 3, 1, 1),
        ("(d) 차단3 그밖3 교재1", 3, 3, 1),
        ("(e) 차단1 그밖1 교재3", 1, 1, 3),
        ("(f) 차단3 그밖1 교재3", 3, 1, 3),
        ("(g) 차단1 그밖3 교재1 (추가)", 1, 3, 1),
        ("(h) 차단1 그밖3 교재3 (추가)", 1, 3, 3),
    ]
    print("\n== 모의실험 1: 부분 다수결 (정확도, 차단 F1. 괄호는 한 표를 대는 실행 조합별 최소~최대) ==")
    print(f"{'방식':<26} 호출 | 조정용 | 확인용")
    fixed = {}
    for name, nb, no, nm in schemes:
        d, h = fixed_scheme(dev, nb, no, nm), fixed_scheme(hold, nb, no, nm)
        fixed[name] = (nb + no + nm, d, h)
        print(f"{name:<26} {nb + no + nm:>2}  | {cell(d)} | {cell(h)}")
    print("  (a)를 실행별로 (같은 실행이 세 확인을 모두 댐, 실제 한 표 실행과 같다):")
    single = {s: [] for s in SPLITS}
    for i, n in enumerate(names):
        for s in SPLITS:
            preds = [merged(it["votes"][i]["block"], it["votes"][i]["other"], it["votes"][i]["self_info"],
                            it["votes"][i]["material"]) for it in items[s]]
            single[s].append(score(preds, [it["gold"] for it in items[s]]))
        print(f"    실행 {i + 1}: 조정용 {cell([single['dev'][i]])} | 확인용 {cell([single['holdout'][i]])}")
    for s, label in zip(SPLITS, ("조정용", "확인용")):
        accs, f1s = [r[0] for r in single[s]], [r[1] for r in single[s]]
        print(f"    {label} 실행 간 폭: 정확도 {max(accs) - min(accs):.1%}p, 차단 F1 {max(f1s) - min(f1s):.3f}"
              f" / 문항 수에서 오는 표준오차(정확도) 약 {(mean(accs) * (1 - mean(accs)) / len(items[s])) ** 0.5:.1%}p")

    # 2. 조건부 추가 호출
    block_rules = [("-", {}), ("i", {"block_any": True})]
    for t in (80, 85, 90):
        block_rules.append((f"ii<{t}", {"block_conf": t}))
    for t in (80, 85, 90):
        block_rules.append((f"i+ii<{t}", {"block_any": True, "block_conf": t}))
    trigger_sets = []
    for (bn, bk), on, mn in itertools.product(block_rules, (False, True), (False, True)):
        name = " ".join(x for x in (bn if bn != "-" else "", "iii" if on else "", "iv" if mn else "") if x) or "없음(=a)"
        trigger_sets.append((name, dict(bk, other_any=on, material_true=mn)))
    print("\n== 모의실험 2: 조건부 추가 호출 (첫 호출을 실행 1·2·3으로 바꿔 평균. 괄호는 그 셋의 최소~최대) ==")
    print("조건: i 차단이 none 아님 / ii<T 차단 확신 < T / iii 그 밖이 none 아님 / iv 교재 밖 True")
    print(f"{'조건':<16} | 조정용: 정확도 F1 평균호출 | 확인용: 정확도 F1 평균호출")
    esc = {}
    for name, kw in trigger_sets:
        d = [escalate(dev, f, **kw) for f in range(3)]
        h = [escalate(hold, f, **kw) for f in range(3)]
        esc[name] = (d, h)
        print(f"{name:<16} | {cell(d)} {mean(r[2] for r in d):.2f} | {cell(h)} {mean(r[2] for r in h):.2f}")

    # 3. 조정용에서 고르기 (정확도 + 차단 F1이 가장 큰 것, sim_merge.py와 같은 기준)
    def dev_key(results):
        return mean(r[0] for r in results) + mean(r[1] for r in results)

    print("\n== 조정용에서 고른 방식 (기준: 조정용 정확도 + 차단 F1) ==")
    cands = [(n, v) for n, v in fixed.items() if v[0] <= 5]
    n5, v5 = max(cands, key=lambda c: dev_key(c[1][1]))
    print(f"요청당 호출 5번 이하(고정 방식): {n5}, 호출 {v5[0]} | 조정용 {cell(v5[1], False)} | 확인용 {cell(v5[2], False)}")
    for budget in (4.0, 5.0):
        cands = [(n, v) for n, v in esc.items() if mean(r[2] for r in v[0]) <= budget]
        n, (d, h) = max(cands, key=lambda c: dev_key(c[1][0]))
        print(f"평균 호출 {budget:.0f}번 이하(조건부): {n} | 조정용 {cell(d, False)} 호출 {mean(r[2] for r in d):.2f}"
              f" | 확인용 {cell(h, False)} 호출 {mean(r[2] for r in h):.2f}")
    a, b = fixed["(a) 전부 1표"], fixed["(b) 전부 3표"]
    print(f"기준: (a) 조정용 {cell(a[1], False)} | 확인용 {cell(a[2], False)}  /  (b) 조정용 {cell(b[1], False)} | 확인용 {cell(b[2], False)}")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_RUNS)
