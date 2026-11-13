"""64-concurrency async load test against a running /tutor & /guard server.

Usage:
    python scripts/stress_test.py --base-url http://localhost:8000 --token dev-token
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grader.scoring import parse_guard, parse_tutor


def _load_payloads(public_dir: Path, total: int) -> List[Tuple[str, Dict[str, Any], bool]]:
    """Build a balanced mix of /tutor and /guard payloads from public/*.jsonl."""
    items: List[Tuple[str, Dict[str, Any], bool]] = []
    tutor_path = public_dir / "tutor.jsonl"
    guard_path = public_dir / "guard.jsonl"

    tutor_rows = [json.loads(l) for l in tutor_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    guard_rows = [json.loads(l) for l in guard_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    half = total // 2
    for i in range(half):
        row = tutor_rows[i % len(tutor_rows)]
        payload = {
            "spec_version": "1.0",
            "session_id": f"s_{secrets.token_hex(8)}",
            "messages": row["messages"],
        }
        items.append(("tutor", payload, False))

    for i in range(total - half):
        row = guard_rows[i % len(guard_rows)]
        has_cand = "candidate_answer" in row and row["candidate_answer"] is not None
        payload = {
            "spec_version": "1.0",
            "session_id": f"s_{secrets.token_hex(8)}",
            "messages": row["messages"],
        }
        if has_cand:
            payload["candidate_answer"] = row["candidate_answer"]
        items.append(("guard", payload, has_cand))

    return items


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round((pct / 100.0) * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


async def run_stress_test(
    base_url: str,
    token: str,
    concurrency: int,
    total: int,
    timeout_sec: float,
    public_dir: Path,
) -> int:
    payloads = _load_payloads(public_dir, total)
    sem = asyncio.Semaphore(concurrency)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    limits = httpx.Limits(max_connections=max(100, concurrency + 16), max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(timeout_sec, connect=5.0)

    async def _send(client: httpx.AsyncClient, endpoint: str, body: Dict[str, Any], has_cand: bool) -> Dict[str, Any]:
        url = f"{base_url.rstrip('/')}/{endpoint}"
        async with sem:
            t0 = time.perf_counter()
            status = 0
            valid = False
            err = ""
            try:
                resp = await client.post(url, json=body, headers=headers)
                status = resp.status_code
                if status == 200:
                    data = resp.json()
                    if endpoint == "tutor":
                        valid = parse_tutor(data) is not None
                    else:
                        valid = parse_guard(data, has_candidate=has_cand) is not None
            except Exception as exc:
                err = type(exc).__name__
            dt = time.perf_counter() - t0
            return {"endpoint": endpoint, "status": status, "valid": valid, "latency": dt, "err": err}

    wall_start = time.perf_counter()
    async with httpx.AsyncClient(limits=limits, timeout=timeout, follow_redirects=False) as client:
        results = await asyncio.gather(*(_send(client, ep, body, hc) for ep, body, hc in payloads))
    wall_elapsed = time.perf_counter() - wall_start

    latencies = sorted(r["latency"] for r in results)
    ok_200 = sum(1 for r in results if r["status"] == 200)
    valid_cnt = sum(1 for r in results if r["valid"])
    n = len(results)

    print(f"=== Stress Test Summary ({n} requests, concurrency={concurrency}) ===")
    print(f"Total wall-clock time : {wall_elapsed:.3f}s")
    print(f"HTTP 200 rate         : {ok_200}/{n} ({100.0 * ok_200 / n:.1f}%)")
    print(f"Schema validity rate  : {valid_cnt}/{n} ({100.0 * valid_cnt / n:.1f}%)")
    print(f"Latency p50           : {_percentile(latencies, 50):.3f}s")
    print(f"Latency p95           : {_percentile(latencies, 95):.3f}s")
    print(f"Latency p99           : {_percentile(latencies, 99):.3f}s")
    print(f"Latency max           : {max(latencies) if latencies else 0.0:.3f}s")

    errors = [r for r in results if r["status"] != 200 or not r["valid"]]
    if errors:
        print(f"Non-200 or invalid count: {len(errors)} (sample: {errors[:3]})")
        return 1
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="64-concurrency load test for /tutor and /guard.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Server base URL.")
    parser.add_argument("--token", default=os.environ.get("TEAM_TOKEN", "dev-token"), help="Bearer token.")
    parser.add_argument("--concurrency", type=int, default=64, help="Max concurrent requests.")
    parser.add_argument("--total", type=int, default=64, help="Total requests to send.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds.")
    parser.add_argument("--public-dir", type=Path, default=ROOT / "public", help="Path to public/*.jsonl.")
    args = parser.parse_args(argv)

    return asyncio.run(
        run_stress_test(
            base_url=args.base_url,
            token=args.token,
            concurrency=args.concurrency,
            total=args.total,
            timeout_sec=args.timeout,
            public_dir=args.public_dir,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
