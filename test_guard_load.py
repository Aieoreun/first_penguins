"""Bounded HTTP load test using public guard requests only; stores metadata only."""
import argparse
import asyncio
import json
import math
import os
import secrets
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from guard_schemas import validate_guard_response

ROOT = Path(__file__).resolve().parent


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def call_one(item, url, token, timeout):
    candidate = item.get('candidate_answer')
    payload = {'spec_version': '1.0', 'session_id': 'public-load-' + secrets.token_hex(8), 'messages': item['messages']}
    if candidate is not None:
        payload['candidate_answer'] = candidate
    request = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode('utf-8'), method='POST',
        headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'},
    )
    start = time.monotonic()
    status, error = None, None
    try:
        opener = urllib.request.build_opener(NoRedirect())
        with opener.open(request, timeout=timeout) as response:
            status = response.status
            data = response.read(256 * 1024 + 1)
        if status != 200:
            raise ValueError('unexpected_status')
        if len(data) > 256 * 1024:
            raise ValueError('response_too_large')
        result = validate_guard_response(json.loads(data), candidate)
        if candidate is None and result.output is not None:
            raise ValueError('unexpected_output')
        if result.output is not None and result.output.grounded != (not result.output.unsupported):
            raise ValueError('inconsistent_grounding')
    except urllib.error.HTTPError as exc:
        status, error = exc.code, f'HTTPError:{exc.code}'
        exc.close()
    except Exception as exc:
        # Never serialize exception bodies, requests or responses.
        error = type(exc).__name__
    return {'status': status, 'error': error, 'seconds': round(time.monotonic() - start, 4),
            'has_candidate': candidate is not None}


async def run(items, args, token):
    semaphore = asyncio.Semaphore(args.workers)
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # to_thread uses this explicitly bounded default executor.
        loop.set_default_executor(pool)
        async def one(item):
            async with semaphore:
                return await asyncio.to_thread(call_one, item, args.url, token, args.timeout)
        return await asyncio.gather(*(one(item) for item in items))


def main():
    parser = argparse.ArgumentParser(description='공개 문항으로 가드 HTTP 동시 부하 검사')
    parser.add_argument('--url', default='http://127.0.0.1:8001/guard')
    parser.add_argument('--requests', type=int, default=64)
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--timeout', type=float, default=29)
    args = parser.parse_args()
    if not 1 <= args.workers <= 64 or not 1 <= args.requests <= 64 or not 0 < args.timeout <= 30:
        parser.error('requests/workers=1..64, timeout=(0,30] 필요')
    token = os.environ.get('TEAM_TOKEN')
    if not token:
        parser.error('TEAM_TOKEN 환경변수가 필요합니다.')
    public = [json.loads(line) for line in (ROOT / 'public/guard.jsonl').read_text(
        encoding='utf-8').splitlines() if line.strip()]
    if len(public) < args.requests or any(item.get('endpoint') != 'guard' for item in public):
        parser.error('공개 가드 문항이 부족하거나 파일 형식이 잘못됐습니다.')
    items = public[:args.requests]
    start = time.monotonic()
    records = asyncio.run(run(items, args, token))
    times = sorted(record['seconds'] for record in records)
    summary = {
        'requests': len(records), 'workers': args.workers,
        'candidate_requests': sum(record['has_candidate'] for record in records),
        'total_seconds': round(time.monotonic() - start, 3),
        'latency_p95_seconds': times[math.ceil(len(times) * .95) - 1],
        'latency_max_seconds': times[-1],
        'errors': dict(Counter(record['error'] for record in records if record['error'])),
        'error_count': sum(record['error'] is not None for record in records),
        'http_statuses': dict(Counter(str(record['status']) for record in records)),
        'scope': 'HTTP endpoint using public data; correctness score is evaluated separately',
    }
    destination = ROOT / 'work/evaluations' / (
        'load-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f') + '.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({'summary':summary,'request_metadata':records},
                                    ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'부하 검사 기록: {destination}')


if __name__ == '__main__':
    main()
