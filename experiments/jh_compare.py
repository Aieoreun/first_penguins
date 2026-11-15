"""Collect PUBLIC responses over HTTP, then call the unchanged official scorer.

One fresh server per variant; generation and judging never contend for API slots.
Only bundled public/*.jsonl can enter this development recorder. It is not a
production request recorder or answer cache. Production code never imports it.
"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipelines.kimjaehak.config import BASELINE_ENV, INITIAL_VARIANTS, VARIANTS
from harness.score_local import score_local, load_kb_dir, as_scoring_input
from grader.judge import make_judge
from grader import scoring


def evaluator_hashes():
    paths = sorted((ROOT / "grader").glob("*.py")) + [ROOT / "harness/score_local.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def request_key(endpoint, payload):
    return json.dumps([endpoint, payload["messages"], payload.get("candidate_answer")],
                      ensure_ascii=False, sort_keys=True)


def load_public(limit=0, endpoints=("tutor", "guard")):
    items = []
    for endpoint in endpoints:
        rows = [json.loads(s) for s in (ROOT / "public" / f"{endpoint}.jsonl").read_text().splitlines() if s.strip()]
        random.Random(20261115).shuffle(rows)
        items.extend(rows[:limit] if limit else rows)
    random.Random(20261115).shuffle(items)
    return items


def post(port, token, endpoint, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/{endpoint}",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
        "Authorization": f"Bearer {token}"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read(256 * 1024 + 1))


async def collect(items, port, token, workers, dest):
    sem = asyncio.Semaphore(workers)
    completed = 0
    with dest.open("w") as stream:
        async def one(item):
            nonlocal completed
            payload = {"spec_version": "1.0", "session_id": "s_" + os.urandom(12).hex(),
                       "messages": item["messages"]}
            if "candidate_answer" in item:
                payload["candidate_answer"] = item["candidate_answer"]
            async with sem:
                start = time.monotonic()
                err, response = None, None
                try:
                    response = await asyncio.to_thread(post, port, token, item["endpoint"], payload)
                except Exception as exc:
                    err = type(exc).__name__
                row = {"id": item["id"], "endpoint": item["endpoint"], "response": response,
                       "seconds": round(time.monotonic() - start, 3), "error": err}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                completed += 1
                if completed % 25 == 0 or completed == len(items):
                    print(f"collected {completed}/{len(items)}", flush=True)
                return row
        return await asyncio.gather(*(one(p) for p in items))


def wait_ready(proc, port):
    until = time.monotonic() + 120
    while time.monotonic() < until:
        if proc.poll() is not None:
            raise RuntimeError("Server exited; inspect the local server log")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError("Server did not become ready")


def journal_judge(judge, path):
    """Checkpoint identical PUBLIC scoring calls, without changing judge prompts."""
    cached = {}
    if path.exists():
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            cached[entry["key"]] = entry["result"]
    lock = threading.Lock()
    def call(kind, payload):
        key = hashlib.sha256(json.dumps([kind, payload], ensure_ascii=False,
                                       sort_keys=True).encode()).hexdigest()
        with lock:
            if key in cached:
                return cached[key]
        result = judge(kind, payload)
        with lock:
            cached[key] = result
            with path.open("a") as stream:
                stream.write(json.dumps({"key": key, "result": result}) + "\n")
            if len(cached) % 25 == 0:
                print(f"judge calls checkpointed: {len(cached)}", flush=True)
        return result
    return call


def score_rows(items, rows, workers, no_judge=False, journal=None):
    by_id = {r["id"]: r for r in rows}
    if set(by_id) != {p["id"] for p in items}:
        raise ValueError("Missing or duplicate public responses; refusing partial scoring")
    cache = {request_key(p["endpoint"], p): by_id[p["id"]]["response"] for p in items}
    def call(endpoint, payload):
        return cache[request_key(endpoint, payload)]
    judge = None if no_judge else make_judge(os.environ["OPENAI_BASE_URL"],
                                           os.environ["OPENAI_MODEL"], os.environ["OPENAI_API_KEY"])
    if judge is not None and journal is not None:
        judge = journal_judge(judge, journal)
    # Official implementation and judge prompts are used verbatim.
    result = score_local(items, call, judge, load_kb_dir(ROOT / "kb"), workers=workers)
    guards = []
    confusion = {}
    for p in items:
        if p["endpoint"] != "guard":
            continue
        item, variant = as_scoring_input(p)
        response = by_id[p["id"]]["response"]
        guards.append(scoring.score_guard(item, variant, response))
        pred = (response or {}).get("input", {}).get("category", "INVALID")
        key = p["labels"]["category"] + " -> " + pred
        confusion[key] = confusion.get(key, 0) + 1
    outs = [g for g in guards if g["has_output"]]
    cats = [g for g in guards if g["category_scored"]]
    def mean(records, field):
        return round(sum(float(r.get(field) or 0) for r in records) / len(records), 4) if records else None
    result["guard_details"] = {
        "block_f1": scoring.block_f1(guards), "category_accuracy": mean(cats, "category_correct"),
        "grounded_accuracy": mean(outs, "output_correct"), "span_f1": mean(outs, "span_f1"),
        "safe_accuracy": mean(outs, "safe_correct"),
        "false_block": sum(g["false_block"] for g in guards),
        "missed_block": sum(g["missed_block"] for g in guards), "confusion": confusion,
    }
    return result


def run_variant(args, variant, repetition, items, root):
    dest = root / f"{variant}_{repetition}"
    dest.mkdir()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    manifest = {
        "variant": variant, "repeat": repetition, "workers": args.workers,
        "judge_workers": args.judge_workers, "count": len(items), "judge_enabled": not args.no_judge,
        "endpoints": args.endpoints,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "model": os.environ["OPENAI_MODEL"],
        "pinned_v2": BASELINE_ENV if variant.startswith("jh_") else None,
        "embedding_configured": bool(os.environ.get("EMBEDDING_BASE_URL") and os.environ.get("EMBEDDING_MODEL")),
        "evaluator_hashes": evaluator_hashes(),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"START {variant} repeat={repetition} public={len(items)} workers={args.workers}", flush=True)
    with (dest / "server.log").open("w") as log:
        proc = subprocess.Popen([sys.executable, "-u", "experiments/jh_server.py", "--variant", variant,
                                 "--port", str(port)], cwd=ROOT, stdout=log, stderr=log)
        try:
            wait_ready(proc, port)
            start = time.monotonic()
            rows = asyncio.run(collect(items, port, os.environ["TEAM_TOKEN"], args.workers, dest / "responses.jsonl"))
            collect_seconds = round(time.monotonic() - start, 2)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
    secs = sorted(r["seconds"] for r in rows)
    server_log = (dest / "server.log").read_text()
    errors = re.findall(r"errors=([^\s]+)", server_log)
    for line in server_log.splitlines():
        if line.startswith("EXPERIMENT_CONFIG "):
            manifest["effective_config"] = json.loads(line.removeprefix("EXPERIMENT_CONFIG "))
    (dest / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    runtime = {"collect_seconds": collect_seconds, "p50": statistics.median(secs),
        "p95": secs[min(len(secs)-1, int(len(secs)*0.95))], "max": max(secs),
        "http_errors": sum(bool(r["error"]) for r in rows),
        "stage_failure_requests": sum(e != "-" for e in errors),
        "stage_errors": dict(__import__('collections').Counter(e for es in errors if es != "-" for e in es.split(',')))}
    (dest / "runtime.json").write_text(json.dumps(runtime, indent=2))
    print(f"SCORING {variant} (official scorer, judge={'off' if args.no_judge else 'on'})", flush=True)
    result = score_rows(items, rows, args.judge_workers, args.no_judge, dest / "judge.jsonl")
    result["runtime"] = runtime
    result["score_scope"] = "+".join(args.endpoints)
    if evaluator_hashes() != manifest["evaluator_hashes"]:
        raise RuntimeError("Official evaluator files changed during experiment")
    (dest / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: result[k] for k in ("total", "tutor", "guard", "invalid", "judge_errors", "runtime")}), flush=True)
    return {"variant": variant, "repeat": repetition, **{k: result[k] for k in
            ("total", "tutor", "guard", "invalid", "judge_errors", "runtime", "guard_details")}}


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", choices=(*VARIANTS, "v4"), default=list(INITIAL_VARIANTS))
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--judge-workers", type=int, default=8)
    ap.add_argument("--endpoints", nargs="+", choices=("tutor", "guard"), default=["tutor", "guard"],
                    help="Single endpoint is diagnostic only; its total is not a full-set score")
    ap.add_argument("--limit", type=int, default=0, help="Smoke test: items per endpoint; 0 means all 400")
    ap.add_argument("--no-judge", action="store_true", help="Schema/load only; not comparable full tutor scores")
    ap.add_argument("--name", default=time.strftime("%Y%m%d-%H%M%S"))
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        ap.error("name must contain only letters, digits, underscore or hyphen")
    if min(args.workers, args.judge_workers, args.repeat) < 1 or args.limit < 0:
        ap.error("workers and repeat must be positive; limit must be nonnegative")
    if len(set(args.endpoints)) != len(args.endpoints):
        ap.error("endpoints must not repeat")
    for key in ("OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_API_KEY", "TEAM_TOKEN"):
        if not os.environ.get(key):
            ap.error(f"Missing setting: {key}")
    root = ROOT / "experiments/.cache" / ("jh_" + args.name)
    root.mkdir(parents=True, exist_ok=False)
    items = load_public(args.limit, args.endpoints)
    results = []
    for repetition in range(1, args.repeat + 1):
        for variant in args.variants:
            results.append(run_variant(args, variant, repetition, items, root))
            (root / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print("RESULT_DIR", root, flush=True)


if __name__ == "__main__":
    main()
