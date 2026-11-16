"""Evaluate a frozen public guard server over HTTP; never send labels to the server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def request(url, payload, token=None, timeout=30):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode(),
        headers=headers, method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(req, timeout=timeout) as response:
        return response.status, response.read(256 * 1024 + 1), response.headers.get("X-Guard-Request-ID")


def fingerprint(root):
    paths = list(root.glob("*.py")) + list((root / "kb").rglob("*"))
    paths += [root / "work/kb_index.json", root / "public/guard.jsonl"]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_file()}


def one(item, url, token, validate_guard_response):
    payload = {"spec_version": "1.0", "session_id": "public-eval", "messages": item["messages"]}
    candidate = item.get("candidate_answer")
    if candidate is not None:
        payload["candidate_answer"] = candidate
    row = {"id": item["id"], "response": None, "error": None, "status": None, "request_id": None}
    start = time.monotonic()
    try:
        status, body, request_id = request(url, payload, token)
        row.update(status=status, request_id=request_id)
        if status != 200 or len(body) > 256 * 1024:
            raise ValueError("invalid_http_response")
        data = json.loads(body)
        result = validate_guard_response(data, candidate)
        if candidate is None and result.output is not None:
            raise ValueError("unexpected_output")
        if result.output is not None and result.output.grounded != (not result.output.unsupported):
            raise ValueError("inconsistent_grounding")
        row["response"] = data
    except urllib.error.HTTPError as error:
        row.update(status=error.code, error=f"HTTPError:{error.code}",
                   request_id=error.headers.get("X-Guard-Request-ID"))
        error.close()
    except Exception as error:
        row["error"] = type(error).__name__
    row["seconds"] = round(time.monotonic() - start, 6)
    return row


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run(root, dest, public, workers, repeat, score_run, validate_guard_response):
    dest.mkdir(parents=True, exist_ok=False)
    before = fingerprint(root)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}/guard"
    env = dict(os.environ, GUARD_REVIEW_MODE="off", GUARD_SAFETY_REVIEW="off", PYTHONDONTWRITEBYTECODE="1")
    (dest / "traces").mkdir()
    env.update(GUARD_PUBLIC_DIAGNOSTICS="1", GUARD_PUBLIC_TRACE_DIR=str(dest / "traces"), GUARD_PROJECT_ROOT=str(root))
    # Do not inherit module search paths pointing at the mutable production checkout.
    env["PYTHONPATH"] = str(root)
    command = [sys.executable, "-m", "uvicorn", "public_guard_monitor:app", "--host", "127.0.0.1", "--port", str(port)]
    manifest = {"root": str(root), "workers": workers, "repeat": repeat,
                "review_mode": "off", "safety_review": "off", "source_hashes": before,
                "chat_model": os.environ["OPENAI_MODEL"], "embedding_model": os.environ["EMBEDDING_MODEL"],
                "port": port, "scope": "public-only actual HTTP; fresh diagnostic server wrapping current /guard with identical off settings; independent model calls"}
    proc = None
    summary = None
    with (dest / "server.log").open("w") as log:
        try:
            proc = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            manifest["pid"] = proc.pid
            dump(dest / "manifest.json", manifest)
            startup_deadline = time.monotonic() + 30
            while True:
                if proc.poll() is not None:
                    raise RuntimeError("server_start_failed")
                try:
                    request(url, {"messages": []}, timeout=1)
                except urllib.error.HTTPError as error:
                    code = error.code
                    error.close()
                    if code != 401:
                        raise RuntimeError("auth_check_failed")
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() >= startup_deadline:
                        raise TimeoutError("server_start_timeout")
                    time.sleep(0.2)
            token = os.environ["TEAM_TOKEN"]
            try:
                request(url, {"messages": None}, token, timeout=2)
            except urllib.error.HTTPError as error:
                code = error.code
                error.close()
                if code != 422:
                    raise RuntimeError("authenticated_preflight_failed")
            else:
                raise RuntimeError("invalid_request_accepted")
            try:
                request(url, {"messages": [{"role":"user", "content":"NON_PUBLIC_PROBE_8a2b"}]}, token, timeout=2)
            except urllib.error.HTTPError as error:
                code=error.code;error.close()
                if code != 503:raise RuntimeError("public_allowlist_failed")
            else:raise RuntimeError("public_allowlist_failed")
            if list((dest / "traces").glob("*.json")):raise RuntimeError("non_public_trace_written")
            manifest["non_public_rejected_without_trace"]=True
            dump(dest / "manifest.json",manifest)
            print(f"START repeat={repeat} workers={workers} destination={dest}", flush=True)
            started = time.monotonic()
            rows = {}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(one, item, url, token, validate_guard_response) for item in public]
                for future in as_completed(futures):
                    row = future.result()
                    rows[row["id"]] = row
                    with (dest / "responses.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if len(rows) % 20 == 0:
                        print(f"repeat={repeat} completed={len(rows)}/{len(public)}", flush=True)
            records = [rows[item["id"]] for item in public]
            dump(dest / "records.json", records)
            summary = score_run(public, records)
            summary.update(repeat=repeat, workers=workers, mean_seconds=sum(r["seconds"] for r in records) / len(records),
                           total_seconds=round(time.monotonic() - started, 3),
                           http_statuses=dict(Counter(str(r["status"]) for r in records)))
            traces = [json.loads(p.read_text()) for p in (dest / "traces").glob("*.json")]
            summary["traces"] = len(traces)
            trace_ids={t["request_id"] for t in traces}
            summary["unlinked_http_requests"]=[r["id"] for r in records if r["request_id"] not in trace_ids]
            summary["trace_missing_public_ids"]=sorted({p["id"] for p in public}-{t["id"] for t in traces})
            after = fingerprint(root)
            summary["source_files_changed_during_evaluation"] = [p for p in before.keys() | after.keys() if before.get(p) != after.get(p)]
            dump(dest / "summary.json", summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            dump(dest / "cleanup.json", {"pid": proc.pid if proc else None,
                 "stopped": proc.poll() is not None if proc else True})
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 64 or args.repeats < 1:
        parser.error("workers must be 1..64 and repeats >=1")
    root = args.root.resolve()
    sys.path[:0] = [str(root), str(root / "grader")]
    from evaluate_guard_review import score_run
    from guard_schemas import validate_guard_response
    public = [json.loads(line) for line in (root / "public/guard.jsonl").read_text().splitlines() if line.strip()]
    assert len(public) == 200 and all(item.get("endpoint") == "guard" for item in public)
    args.destination.mkdir(parents=True, exist_ok=False)
    summaries = []
    for repeat in range(1, args.repeats + 1):
        summaries.append(run(root, args.destination / f"run-{repeat}", public, args.workers, repeat, score_run, validate_guard_response))
        dump(args.destination / "summary.json", {"runs": summaries})


if __name__ == "__main__":
    main()
