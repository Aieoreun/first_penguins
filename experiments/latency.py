"""gpt-5-nano 응답 시간 측정.

실제로 쓸 길이의 요청으로, 추론 수준과 동시 요청 수에 따라 한 번 부르는 데
몇 초 걸리는지 잰다. 요청당 제한은 30초이고 채점 때 동시에 64개가 들어온다.

재는 요청 두 가지:
  분류: 분류 프롬프트 초안 + 교재 조각 15개 + 대화. 출력은 짧은 JSON.
  답변: 간단한 지시 + 섹션 하나 통째 + 대화. 출력은 답변 글.
        (답변 프롬프트는 아직 설계 전이다. 길이만 실제와 비슷하게 맞춘 임시 글이다)

실행 (저장소 폴더에서, .env를 불러온 뒤):
    python3 experiments/latency.py                 # 기본: 하나씩 5번, 동시에 16번, 동시에 64번
    python3 experiments/latency.py --bursts 16     # 동시 요청 수를 바꾼다
    python3 experiments/latency.py --efforts low   # 추론 수준을 바꾼다

환경변수: OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_API_KEY
표준 라이브러리만 쓴다. 화면에는 숫자와 오류 종류만 찍는다. 키, 문항 내용, 모델 답변은 찍지 않는다.
"""
import argparse
import json
import os
import random
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.kb import load_chunks  # noqa: E402
from core.schema import parse_label  # noqa: E402
from search_compare import BM25  # noqa: E402

LIMIT = 30.0  # 요청당 제한(초)
PROMPT_DOC = ROOT / "docs" / "sw" / "03_분류_프롬프트_초안.md"
GENERATE_SYSTEM = (
    "너는 중학생용 AI 튜터다. 아래 [교재]에 있는 내용만 근거로, 학생의 마지막 질문에 "
    "세 문장 안팎으로 친절하게 답한다. 교재에 없는 내용은 지어내지 않는다."
)


def classify_system():
    text = PROMPT_DOC.read_text(encoding="utf-8")
    start = text.index("````text") + len("````text")
    return text[start:text.index("````", start)].strip()


def conversation(messages):
    history = "\n".join(f"{m['role']}: {m['content']}" for m in messages[:-1]) or "(없음)"
    return f"[앞 대화]\n{history}\n\n[마지막 사용자 발화]\n{messages[-1]['content']}"


def build_requests(n):
    """공개 세트에서 문항을 뽑아 분류용·답변용 메시지를 만든다."""
    chunks = load_chunks(ROOT / "kb")
    bm25 = BM25([c.text for c in chunks])
    with open(ROOT / "public" / "tutor.jsonl", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    random.Random(0).shuffle(items)
    system = classify_system()
    classify, generate = [], []
    for item in items:
        messages = item["messages"]
        scores = bm25.scores("\n".join(m["content"] for m in messages))
        top = sorted(range(len(chunks)), key=lambda i: -scores[i])[:15]
        found = "\n\n".join(chunks[i].text for i in top)
        classify.append([
            {"role": "system", "content": system},
            {"role": "user", "content": f"[교재 검색 결과]\n{found}\n\n{conversation(messages)}"},
        ])
        if item["labels"]["evidence"]:
            sid = item["labels"]["evidence"][0]
            section = "\n\n".join(c.text for c in chunks if c.section_id == sid)
            generate.append([
                {"role": "system", "content": GENERATE_SYSTEM},
                {"role": "user", "content": f"[교재]\n{section}\n\n{conversation(messages)}"},
            ])
    return classify[:n], generate[:n]


class Client:
    def __init__(self):
        self.base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
        self.model = os.environ.get("OPENAI_MODEL", "")
        self.key = os.environ.get("OPENAI_API_KEY", "")
        missing = [n for n, v in (("OPENAI_BASE_URL", self.base), ("OPENAI_MODEL", self.model),
                                  ("OPENAI_API_KEY", self.key)) if not v]
        if missing:
            sys.exit("환경변수가 없다: " + ", ".join(missing))

    def call(self, messages, effort, max_tokens, json_mode=False, timeout=90):
        """한 번 부른다. (걸린 시간, 오류 종류 또는 None, 출력 글, usage)를 돌려준다."""
        body = {"model": self.model, "messages": messages, "max_completion_tokens": max_tokens,
                "reasoning_effort": effort}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(self.base + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     method="POST", headers={"Content-Type": "application/json",
                                                             "Authorization": "Bearer " + self.key})
        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            content = data["choices"][0]["message"].get("content") or ""
            return time.time() - start, None, content, data.get("usage") or {}
        except urllib.error.HTTPError as e:
            return time.time() - start, f"HTTP {e.code}", "", {}
        except Exception as e:
            return time.time() - start, type(e).__name__, "", {}


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p * (len(values) - 1))))]


def run(client, name, requests, effort, max_tokens, concurrency, count, check):
    picked = [requests[i % len(requests)] for i in range(count)]
    json_mode = check is not None
    start = time.time()
    with ThreadPoolExecutor(concurrency) as pool:
        results = list(pool.map(lambda m: client.call(m, effort, max_tokens, json_mode), picked))
    wall = time.time() - start
    ok = [r for r in results if r[1] is None]
    errors = Counter(r[1] for r in results if r[1] is not None)
    times = [r[0] for r in ok]
    line = f"{name:<4} 추론 {effort:<8} 동시 {concurrency:>2} × {count:>2}번 | "
    if not ok:
        print(line + "전부 실패 " + str(dict(errors)))
        return
    over = sum(t > LIMIT for t in times)
    out_tokens = [r[3].get("completion_tokens") for r in ok if r[3].get("completion_tokens") is not None]
    reason = [(r[3].get("completion_tokens_details") or {}).get("reasoning_tokens") for r in ok]
    reason = [x for x in reason if x is not None]
    in_tokens = [r[3].get("prompt_tokens") for r in ok if r[3].get("prompt_tokens") is not None]
    line += (f"중앙 {statistics.median(times):5.1f}초  90% {pct(times, 0.9):5.1f}초  최대 {max(times):5.1f}초 | "
             f"30초 초과 {over} | 실패 {sum(errors.values())}")
    if errors:
        line += " " + str(dict(errors))
    line += f" | 전체 {wall:5.1f}초"
    if in_tokens:
        line += f" | 입력 {int(statistics.mean(in_tokens)):,}토큰"
    if out_tokens:
        line += f" 출력 {int(statistics.mean(out_tokens)):,}토큰"
    if reason:
        line += f"(추론 {int(statistics.mean(reason)):,})"
    empty = sum(not r[2].strip() for r in ok)
    if empty:
        line += f" | 빈 출력 {empty}"
    if check is not None:
        line += f" | 형식 맞음 {sum(check(r[2]) for r in ok)}/{len(ok)}"
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--efforts", nargs="+", default=["minimal", "low"], help="추론 수준 (minimal, low, medium, high)")
    ap.add_argument("--bursts", nargs="+", type=int, default=[16, 64], help="동시 요청 수")
    ap.add_argument("--sequential", type=int, default=5, help="하나씩 부르는 횟수")
    args = ap.parse_args()

    client = Client()
    classify, generate = build_requests(64)
    size = lambda reqs: int(statistics.mean(sum(len(m["content"]) for m in r) for r in reqs))
    plans = [(1, args.sequential)] + [(b, b) for b in args.bursts]
    total = 2 * len(args.efforts) * sum(c for _, c in plans)
    print(f"분류 요청 평균 {size(classify):,}자, 답변 요청 평균 {size(generate):,}자. 모두 {total}번 부른다.")
    print("분류는 출력 600토큰까지, 답변은 2000토큰까지 허용한다. '형식 맞음'은 분류 이름을 꺼낼 수 있었던 응답 수다.\n")

    for effort in args.efforts:
        for concurrency, count in plans:
            run(client, "분류", classify, effort, 600, concurrency, count, lambda t: parse_label(t) is not None)
        for concurrency, count in plans:
            run(client, "답변", generate, effort, 2000, concurrency, count, None)
        print()


if __name__ == "__main__":
    main()
