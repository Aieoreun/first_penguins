"""검색 방법 비교 실험: 낱말 검색(BM25), 임베딩 검색, 둘을 합친 검색.

공개 세트 튜터 문항 중 정답 근거가 있는 문항에서, 정답 섹션이 검색 결과의
1위·3위 안·5위 안에 드는 비율을 잰다. 표준 라이브러리만 쓴다.

실행 (저장소 폴더에서):
    python3 experiments/search_compare.py            # 낱말 검색만. API 키가 필요 없다
    python3 experiments/search_compare.py --embed    # 임베딩·혼용까지. 아래 환경변수가 필요하다

환경변수 (런박스에는 주소와 모델 이름이 미리 들어 있다):
    EMBEDDING_BASE_URL, EMBEDDING_MODEL, EMBEDDING_API_KEY (없으면 OPENAI_API_KEY)

임베딩은 experiments/.cache/에 저장해 두고 다시 쓴다. 제공 자료(교재, 공개 세트)의
임베딩뿐이다. 화면에는 숫자만 찍고 키와 문항 내용은 찍지 않는다.
"""
import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.kb import load_chunks  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
RRF_K = 60


# ---------- 낱말 검색 ----------

def tokens(text):
    """한글은 두 글자씩, 영문·숫자는 낱말 단위로 자른다."""
    out = []
    for word in re.findall(r"[가-힣]+|[a-z_]+|\d+", text.lower()):
        if "가" <= word[0] <= "힣":
            out += [word[i:i + 2] for i in range(len(word) - 1)] or [word]
        else:
            out.append(word)
    return out


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.tf = [Counter(tokens(t)) for t in texts]
        self.length = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.length) / len(self.length)
        df = Counter(w for t in self.tf for w in t)
        n = len(texts)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query):
        q = Counter(tokens(query))
        out = []
        for tf, length in zip(self.tf, self.length):
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
            out.append(s)
        return out


# ---------- 임베딩 ----------

class Embedder:
    def __init__(self):
        self.base = os.environ.get("EMBEDDING_BASE_URL", "").rstrip("/")
        self.model = os.environ.get("EMBEDDING_MODEL", "")
        self.key = os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        missing = [n for n, v in (("EMBEDDING_BASE_URL", self.base), ("EMBEDDING_MODEL", self.model),
                                  ("EMBEDDING_API_KEY", self.key)) if not v]
        if missing:
            sys.exit("환경변수가 없다: " + ", ".join(missing))
        CACHE_DIR.mkdir(exist_ok=True)
        self.cache_path = CACHE_DIR / "embeddings.json"
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}

    def _key(self, text):
        return hashlib.sha256((self.model + "\n" + text).encode("utf-8")).hexdigest()

    def call(self, texts, timeout=60):
        """API를 한 번 부른다. 벡터 목록을 돌려준다."""
        body = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(self.base + "/embeddings", data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + self.key})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())["data"]
        return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]

    def embed(self, texts, batch=32):
        """캐시에 없는 글만 불러서 채우고, 길이 1로 맞춘 벡터를 돌려준다."""
        todo = [t for t in dict.fromkeys(texts) if self._key(t) not in self.cache]
        for i in range(0, len(todo), batch):
            part = todo[i:i + batch]
            for attempt in range(3):
                try:
                    vectors = self.call(part)
                    break
                except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
                    if attempt == 2:
                        sys.exit(f"임베딩 호출 실패: {type(e).__name__} {getattr(e, 'code', '')}")
                    time.sleep(2 * (attempt + 1))
            for t, v in zip(part, vectors):
                norm = math.sqrt(sum(x * x for x in v)) or 1.0
                self.cache[self._key(t)] = [round(x / norm, 6) for x in v]
            print(f"  임베딩 {min(i + batch, len(todo))}/{len(todo)}", flush=True)
        if todo:
            self.cache_path.write_text(json.dumps(self.cache))
        return [self.cache[self._key(t)] for t in texts]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


# ---------- 순위와 채점 ----------

def ranks(scores):
    """점수 목록 -> 조각 번호별 등수(1부터)."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    out = [0] * len(scores)
    for r, i in enumerate(order, 1):
        out[i] = r
    return out


def section_order(scores, chunks):
    """조각 점수를 높은 순으로 훑으며 섹션 ID를 처음 나온 순서대로 모은다."""
    seen = []
    for i in sorted(range(len(scores)), key=lambda i: -scores[i]):
        sid = chunks[i].section_id
        if sid not in seen:
            seen.append(sid)
    return seen


def hit_rates(orders, golds):
    n = len(golds)
    return tuple(sum(g in o[:k] for o, g in zip(orders, golds)) / n for k in (1, 3, 5))


QUERIES = {
    "마지막 발화": lambda m: m[-1]["content"],
    "마지막 3개 발화": lambda m: "\n".join(x["content"] for x in m[-3:]),
    "대화 전부": lambda m: "\n".join(x["content"] for x in m),
}


def measure_latency(embedder, sample):
    """질문 하나를 임베딩하는 데 걸리는 시간. 캐시를 거치지 않고 직접 부른다."""
    def one(text):
        start = time.time()
        try:
            embedder.call([text], timeout=30)
            return time.time() - start
        except Exception:
            return None

    seq = [one(t) for t in sample[:10]]
    with ThreadPoolExecutor(64) as pool:
        start = time.time()
        par = list(pool.map(one, (sample * 7)[:64]))
        wall = time.time() - start
    for name, values in (("한 번에 하나씩 10번", seq), ("동시에 64번", par)):
        ok = [v for v in values if v is not None]
        if ok:
            print(f"  {name}: 중앙 {statistics.median(ok):.2f}초, 최대 {max(ok):.2f}초, 실패 {len(values) - len(ok)}")
        else:
            print(f"  {name}: 전부 실패")
    print(f"  동시에 64번 전체 소요: {wall:.2f}초")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true", help="임베딩 검색과 혼용 검색도 잰다")
    ap.add_argument("--no-latency", action="store_true", help="임베딩 호출 시간 측정을 건너뛴다")
    args = ap.parse_args()

    chunks = load_chunks(ROOT / "kb")
    texts = [c.text for c in chunks]
    with open(ROOT / "public" / "tutor.jsonl", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    items = [x for x in items if x["labels"]["evidence"]]
    golds = [x["labels"]["evidence"][0] for x in items]
    english = [i for i, x in enumerate(items) if not re.search(r"[가-힣]", x["messages"][-1]["content"])]
    print(f"조각 {len(chunks)}개, 근거가 있는 문항 {len(items)}개 (마지막 발화에 한글이 없는 문항 {len(english)}개)")

    bm25 = BM25(texts)
    embedder = Embedder() if args.embed else None
    if embedder:
        print("교재 조각 임베딩")
        chunk_vecs = embedder.embed(texts)

    rows = []
    for qname, qfn in QUERIES.items():
        queries = [qfn(x["messages"]) for x in items]
        kw = [bm25.scores(q) for q in queries]
        methods = {"낱말": kw}
        if embedder:
            print(f"질문 임베딩: {qname}")
            qvecs = embedder.embed(queries)
            em = [[dot(q, c) for c in chunk_vecs] for q in qvecs]
            methods["임베딩"] = em
            for label, wk, we in (("혼용 1:1", 1, 1), ("혼용 낱말1:임베딩2", 1, 2), ("혼용 낱말2:임베딩1", 2, 1)):
                fused = []
                for a, b in zip(kw, em):
                    ra, rb = ranks(a), ranks(b)
                    fused.append([wk / (RRF_K + x) + we / (RRF_K + y) for x, y in zip(ra, rb)])
                methods[label] = fused
        for mname, scores in methods.items():
            orders = [section_order(s, chunks) for s in scores]
            allr = hit_rates(orders, golds)
            engr = hit_rates([orders[i] for i in english], [golds[i] for i in english]) if english else None
            rows.append((qname, mname, allr, engr))

    print()
    print(f"{'질의':<12}{'방법':<18}{'1위':>6}{'3위 안':>8}{'5위 안':>8}   영어 문항 1위/3위 안")
    for qname, mname, allr, engr in rows:
        eng = f"{engr[0]:.0%} / {engr[1]:.0%}" if engr else "-"
        print(f"{qname:<12}{mname:<18}{allr[0]:>6.0%}{allr[1]:>8.0%}{allr[2]:>8.0%}   {eng}")

    if embedder and not args.no_latency:
        print("\n질문 하나를 임베딩하는 시간")
        measure_latency(embedder, [QUERIES["대화 전부"](x["messages"]) for x in items])


if __name__ == "__main__":
    main()
