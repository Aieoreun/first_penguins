"""근거 섹션 고르기 실험: 조각 1위의 섹션 vs 섹션 단위로 점수를 모으는 방법.

지금 서버는 낱말 검색(BM25) 1위 조각이 속한 섹션을 근거 섹션으로 쓴다. 이 실험은 조각 점수를
섹션별로 모으면(상위 k개 합, 최고점 + 나머지 평균, 섹션 통째 BM25 등) 섹션 1위가 더 잘 맞는지 잰다.
모델·API·네트워크를 쓰지 않는다. 표준 라이브러리만 쓴다.

과적합 방지: 방법은 조정용(dev)에서만 고른다. 고르는 규칙은 코드에 박혀 있다
(dev 1위 정확도 → dev 3위 안 → 목록에서 먼저 나온 것). 확인용(holdout)은 고른 뒤에 숫자만 본다.
화면에는 숫자만 찍고 문항 내용은 찍지 않는다.

실행 (저장소 폴더에서):
    python3 experiments/eval_section_rank.py --dev-only   # dev 표와 고른 방법만
    python3 experiments/eval_section_rank.py              # dev에서 고른 뒤 holdout까지
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.kb import load_chunks  # noqa: E402
from core.search import BM25, Searcher  # noqa: E402
from experiments.split import load_items  # noqa: E402

DEFAULT_K1, DEFAULT_B = 1.5, 0.75


# ---------- 색인과 질의 ----------

class Index:
    """조각 BM25와 섹션 통째 BM25를 같은 매개변수로 만든다."""

    def __init__(self, chunks, k1=DEFAULT_K1, b=DEFAULT_B):
        searcher = Searcher(chunks)
        self.sections = searcher.section_ids()          # 조각이 처음 나온 순서
        pos = {s: i for i, s in enumerate(self.sections)}
        self.chunk_section = [pos[c.section_id] for c in chunks]
        self.chunk_bm25 = BM25([c.text for c in chunks], k1=k1, b=b)
        self.section_bm25 = BM25([searcher.section_text(s) for s in self.sections], k1=k1, b=b)

    def raw(self, item, beta=0.0):
        """(섹션별 조각 점수 목록, 섹션 통째 점수). beta > 0이면 마지막 사용자 발화 점수를 더한다."""
        query = "\n".join(m["content"] for m in item["messages"])
        if item.get("candidate_answer"):
            query += "\n" + item["candidate_answer"]
        cs, ss = self.chunk_bm25.scores(query), self.section_bm25.scores(query)
        if beta:
            last = next((m["content"] for m in reversed(item["messages"]) if m["role"] == "user"), "")
            cs = [x + beta * y for x, y in zip(cs, self.chunk_bm25.scores(last))]
            ss = [x + beta * y for x, y in zip(ss, self.section_bm25.scores(last))]
        per = [[] for _ in self.sections]
        for sec, score in zip(self.chunk_section, cs):
            per[sec].append(score)
        return per, ss


# ---------- 섹션 점수 방법 (per: 섹션별 조각 점수, ss: 섹션 통째 점수) ----------

def unit(xs):
    lo, hi = min(xs), max(xs)
    return [(x - lo) / (hi - lo) if hi > lo else 0.0 for x in xs]


def best_chunk(per, ss):
    return [max(p) for p in per]


def top_sum(k):
    def f(per, ss):
        return [sum(sorted(p, reverse=True)[:k]) for p in per]
    return f


def max_plus_mean(alpha):
    def f(per, ss):
        out = []
        for p in per:
            rest = sorted(p, reverse=True)[1:]
            out.append(max(p) + alpha * (sum(rest) / len(rest) if rest else 0.0))
        return out
    return f


def section_doc(per, ss):
    return list(ss)


def norm_sum(per, ss):
    return [a + b for a, b in zip(unit(best_chunk(per, ss)), unit(ss))]


METHODS = [
    ("1 기준: 조각 1위의 섹션", best_chunk),
    ("2 상위 2개 합", top_sum(2)),
    ("2 상위 3개 합", top_sum(3)),
    ("2 전체 합", top_sum(None)),
    ("3 최고 + 0.25×나머지 평균", max_plus_mean(0.25)),
    ("3 최고 + 0.5×나머지 평균", max_plus_mean(0.5)),
    ("4 섹션 통째 BM25", section_doc),
    ("5 정규화(조각 최고) + 정규화(섹션)", norm_sum),
]
BETAS = [0.5, 1.0]
PARAMS = [(k1, b) for k1 in (1.2, 1.5, 2.0) for b in (0.5, 0.75)]


# ---------- 채점 ----------

def gold_ranks(index, items, method, beta=0.0):
    """문항마다 정답 섹션의 순위(1부터). 점수가 모두 0이면 서버처럼 '못 찾음'(None)으로 친다."""
    out = []
    for item in items:
        per, ss = index.raw(item, beta)
        scores = method(per, ss)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])  # 동점은 교재 순서
        if scores[order[0]] <= 0:
            out.append(None)
            continue
        gold = {index.sections.index(s) for s in item["labels"]["evidence"]}
        out.append(min(order.index(g) for g in gold) + 1)
    return out


def top(ranks, k):
    return sum(1 for r in ranks if r is not None and r <= k)


def cell(n, total):
    return f"{n:>3}/{total} {100 * n / total:5.1f}%"


def flips(base, other):
    """(기준은 틀리고 이 방법은 맞은 문항 수, 그 반대)."""
    hit = lambda r: r == 1  # noqa: E731
    return (sum(1 for a, b in zip(base, other) if not hit(a) and hit(b)),
            sum(1 for a, b in zip(base, other) if hit(a) and not hit(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-only", action="store_true", help="확인용(holdout)을 읽지 않는다")
    args = ap.parse_args()

    chunks = load_chunks(ROOT / "kb")
    index = Index(chunks)
    evidence = lambda items: [i for i in items if i.get("labels", {}).get("evidence")]  # noqa: E731
    dev = evidence(load_items("dev"))
    print(f"교재 조각 {len(chunks)}개, 섹션 {len(index.sections)}개. dev 근거 문항 {len(dev)}개")

    # 1) 방법 1~5, dev
    rows = [(name, fn, 0.0, gold_ranks(index, dev, fn)) for name, fn in METHODS]
    order = lambda r: (-top(r[3], 1), -top(r[3], 3))  # noqa: E731  (sorted는 안정 정렬 → 먼저 나온 것 우선)
    best_agg = sorted(rows[1:], key=order)[0]
    # 2) 방법 6: 기준과 '1~5 중 dev 최고 집계'에 마지막 사용자 발화 가중
    for name, fn, _, _ in (rows[0], best_agg):
        for beta in BETAS:
            rows.append((f"6 [{name}] + β={beta} 마지막 발화", fn, beta, gold_ranks(index, dev, fn, beta)))
    picked = sorted(rows, key=order)[0]
    base = rows[0]

    print("\n[dev] 1위 정확도 / 3위 안 / 기준 대비 (얻음, 잃음)")
    for name, fn, beta, ranks in rows:
        g, l = flips(base[3], ranks)
        print(f"  {name:<46} {cell(top(ranks, 1), len(dev))}  {cell(top(ranks, 3), len(dev))}  +{g} -{l}")
    print(f"\n1~5 중 dev 최고 집계: {best_agg[0]}")
    print(f"dev에서 고른 방법(규칙: dev 1위 → dev 3위 안 → 목록 순서): {picked[0]}")

    for label, row in (("기준", base), ("고른 방법", picked)):
        miss = [r for r in row[3] if r != 1]
        print(f"dev 오답 순위 [{label}]: 오답 {len(miss)}개 = 2위 {sum(r == 2 for r in miss)}, "
              f"3위 {sum(r == 3 for r in miss)}, 4위 이하 {sum(r is not None and r > 3 for r in miss)}, "
              f"못 찾음 {sum(r is None for r in miss)}")

    # 3) 방법 7: BM25 매개변수. dev에서 기본값을 2문항 이상 이길 때만 보고 대상
    print("\n[dev] BM25 매개변수 (k1, b) → 1위 맞은 수")
    param_rows = []
    for label, row in (("기준", base), ("최고 집계", best_agg)):
        default = top(row[3], 1)
        line = []
        for k1, b in PARAMS:
            idx = index if (k1, b) == (DEFAULT_K1, DEFAULT_B) else Index(chunks, k1, b)
            ranks = gold_ranks(idx, dev, row[1])
            line.append(f"({k1},{b})={top(ranks, 1)}")
            if top(ranks, 1) - default > 1:
                param_rows.append((f"7 [{row[0]}] k1={k1} b={b}", row[1], idx, ranks))
        print(f"  {label}: " + "  ".join(line) + f"   기본값 {default}")
    print("  기본값을 2문항 이상 이긴 조합: " + (", ".join(r[0] for r in param_rows) or "없음"))

    if args.dev_only:
        return

    # 4) holdout: 고르기는 위에서 끝났다. 여기서는 숫자만 본다
    hold = evidence(load_items("holdout"))
    print(f"\n[holdout] 근거 문항 {len(hold)}개. 1위 / 3위 안 / 기준 대비 (얻음, 잃음)")
    hbase = gold_ranks(index, hold, base[1])
    for name, fn, beta, _ in rows:
        ranks = gold_ranks(index, hold, fn, beta)
        g, l = flips(hbase, ranks)
        mark = " ← 기준" if name == base[0] else (" ← dev에서 고름" if name == picked[0] else "")
        print(f"  {name:<46} {cell(top(ranks, 1), len(hold))}  {cell(top(ranks, 3), len(hold))}  +{g} -{l}{mark}")
    for name, fn, idx, _ in param_rows:
        ranks = gold_ranks(idx, hold, fn)
        print(f"  {name:<46} {cell(top(ranks, 1), len(hold))}  {cell(top(ranks, 3), len(hold))}")


if __name__ == "__main__":
    main()
